from utils import qwen2_generate
from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_retriever
import tomllib
from PIL import Image
import re
from tqdm import tqdm
import json
import torch
import torch.nn.functional as F
import time
import csv

class Instructor(BasicPipeline):
    def __init__(self, student, model, processor, config, retriever=None):
        super().__init__(config)
        if retriever is None:
            retriever = get_retriever(config)
        self.retriever = retriever
        self.model = model
        self.processor = processor
        self.student = student
        self.uncertain_threshold = 0.0
        with open('prompts/instructor.toml', 'rb') as f:
            self.instructor_prompt = tomllib.load(f)

    def instud_generate(self, question: str, image: Image):
        logs = []
        feedback_list = []
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": self.instructor_prompt['system_prompt'].format(question=question, is_first_turn="True")},
                ],
            }
        ]
        
        instructor_response, entropy = qwen2_generate(self.model, self.processor, messages)
        
        logs.append({"instructor_first_response": instructor_response})
        state, plan = self.parse_from_instructor(instructor_response)
        messages.append({
                    "role": "assistant",
                    "content": [{
                        "type": "text", "text": f"state: {state}, plan: {plan}"
                    }]
                })
        # if state != "continue":
        #     print(f"ERROR! The Instructor went on strike! Before he went off, he said: {instructor_response}")
        #     return "", messages
        # else:
        conversation_num, max_turns = 0, 5
        while conversation_num < max_turns:
            print("\n<Instructor>")
            print(f"State: {state}, Plan: {plan}")
            print("</Instructor>")
            if state == "finish":
                break
            # messages.append({'role': 'assistant', 'content': instructor_response})
            print("<Student>")
            feedback, student_logs = self.student.generate(question, image, plan)
            print("</Student>")
            feedback_list.append(feedback)
            logs.append({"feedback": feedback})
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Feedback from user: {feedback}\n\n(Reminder: Use this feedback to move towards solving the MAIN QUESTION: '{question}'. If not fully solved, continue asking. Your response must be JSON.)"}
                ]
            })

            instructor_response, entropy = qwen2_generate(self.model, self.processor, messages)
            # instructor_response = self.check_student_response(feedback_list, plan)
            messages.append({
                "role": "assistant",
                "content": [{
                    "type": "text", "text": instructor_response
                }]
            })
            logs.append({"instructor_response": instructor_response})
            state, plan = self.parse_from_instructor(instructor_response)
            
            conversation_num += 1

        final_answer = plan
        logs.append({"student_logs": student_logs})
        return final_answer, logs
    
    def generate(self, question: str, image: Image):
        pass

    def estimate_uncertainty(self, question, img):
        v_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Describe this image in detail."},
                ],
            }
        ]
        
        v_prompt = self.processor.apply_chat_template(
            v_messages, tokenize=False, add_generation_prompt=True
        )
        
        v_inputs = self.processor(
            text=[v_prompt], 
            images=[img], 
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            v_outputs = self.model.generate(
                **v_inputs,
                max_new_tokens=512,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
        

        v_text = self.processor.batch_decode(v_outputs.sequences, skip_special_tokens=True)[0]
        print(f"\n[Visual Probe Output]: {v_text}")

        v_feat = v_outputs.hidden_states[-1][-1][:, -1, :]

        # --- 2. Text Probe (T分支) ---
        t_messages = [{
            "role": "user", 
            "content": f"To answer the question '{question}', describe what the image should look like. Imagine the visual scene details. Only generate your imagination, don't generate irrelevant words or markdown markups."
        }]
        
        t_prompt = self.processor.apply_chat_template(
            t_messages, tokenize=False, add_generation_prompt=True
        )
        
        t_inputs = self.processor(
            text=[t_prompt], 
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            t_outputs = self.model.generate(
                **t_inputs,
                max_new_tokens=512,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

        t_text = self.processor.batch_decode(t_outputs.sequences, skip_special_tokens=True)[0]
        print(f"[Text Probe Output]  : {t_text}")

        t_feat = t_outputs.hidden_states[-1][-1][:, -1, :]

        similarity = F.cosine_similarity(v_feat, t_feat).item()
        
        print(f"[Uncertainty Score]  : {1 - similarity:.4f} (Sim: {similarity:.4f})")
        return 1 - similarity
    
    def parse_from_instructor(self, instructor_response):
        json_data = None
        json_str = None

        markdown_pattern = r"```json\s*(\{.*?\})\s*```"
        match = re.search(markdown_pattern, instructor_response, re.DOTALL | re.IGNORECASE)
        
        if match:
            json_str = match.group(1)
        else:
            # 1.2 如果没找到代码块，尝试寻找最外层的花括号 { ... }
            # 这能处理模型忘记写 ```json 的情况
            brace_pattern = r"\{.*\}"
            match = re.search(brace_pattern, instructor_response, re.DOTALL)
            if match:
                json_str = match.group(0)
        
        if json_str:
            try:
                json_data = json.loads(json_str, strict=False)
                
            except json.JSONDecodeError as e:
                print(f"Warning: JSON extracted but parse failed: {e}")
                print(f"Bad JSON string: {json_str}")
                
        state_content = "continue" 
        plan_content = instructor_response 

        if json_data and isinstance(json_data, dict):            
            if "state" in json_data:
                state_content = str(json_data["state"]).strip().lower()
            
            if "plan" in json_data:
                plan_content = str(json_data["plan"]).strip()
                
            # if "thought" in json_data:
            #     print(f"[Instructor Thought]: {json_data['thought']}")
                
        else:
            print("WARNING! No valid JSON object found in response.")
            state_content = "continue"
            plan_content = instructor_response

        if state_content not in ["continue", "finish"]:
            print(f"Warning: Unexpected state '{state_content}', defaulting to 'continue'")
            state_content = "continue"
        return state_content, plan_content
    def naive_generate(self, question: str, image: Image):
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": f"You are a helpful assistant. Answer the question based on the image provided. Question: {question}"},
                ],
            }
        ]
        response, uncertainty = qwen2_generate(self.model, self.processor, messages)
        return response, uncertainty
    
    def run(self, dataset, do_eval=True, pred_process_fun=None):
        questions = dataset.question
        ids = dataset.id
        prediction_list = []
        entropy_list = []
        start_time = time.time()
        with open("uncertainty_scores.tsv", "w", newline='', encoding="utf-8") as tsv_f:
            writer = csv.writer(tsv_f, delimiter='\t')
            writer.writerow(["id", "uncertainty_score"])
             
        with open("intermediate_logs.jsonl", "w", encoding="utf-8") as f:
            for i, (question, id) in enumerate(zip(questions, ids)):
                print(f"\n[{i}/{len(questions)}] question: {question}")
                
                try:
                    img_path = f"data/datasets/crag/images/{id}.jpg"
                    img = Image.open(img_path).convert("RGB")
                    
                    uncertainty_score = self.estimate_uncertainty(question, img)
                    print(f"Estimated uncertainty score: {uncertainty_score:.4f}")
                    with open("uncertainty_scores.tsv", "a", newline='', encoding="utf-8") as tsv_f:
                        writer = csv.writer(tsv_f, delimiter='\t')
                        writer.writerow([str(id), uncertainty_score])
                        print(f"id: {str(id)}, uncertainty_score: {uncertainty_score}")
                    # uncertainty_score = 0.1
                    # if uncertainty_score > self.uncertain_threshold:
                    #     final_answer, context = self.instud_generate(question, img)
                    # else:
                    #     final_answer, entropy = self.naive_generate(question, img)
                    # prediction_list.append(final_answer)

                    # logs = {"id": id, "question": question, "prediction": final_answer}
                    # f.write(json.dumps(logs, ensure_ascii=False) + "\n")
                    
                    if i % 10 == 0:
                        f.flush()
    
                except torch.cuda.OutOfMemoryError:
                    print(f"!!! CUDA OOM Error at index {i}, id: {id}. Clearing cache and skipping...")
                    with open("uncertainty_scores.tsv", "a", newline='', encoding="utf-8") as tsv_f:
                        writer = csv.writer(tsv_f, delimiter='\t')
                        writer.writerow([str(id), 0.0])
                        print(f"id: {str(id)}, uncertainty_score: {0.0}")
                    torch.cuda.empty_cache()
                    
                    prediction_list.append("Error: CUDA OOM")
                    
                    error_log = {"id": id, "question": question, "prediction": "I can't answer.", "logs": "CUDA OOM Error - Skipped"}
                    f.write(json.dumps(error_log, ensure_ascii=False) + "\n")
                    
                    continue
    
                except Exception as e:
                    print(f"!!! Unknown Error at index {i}, id: {id}: {str(e)}")
                    prediction_list.append(f"Error: {str(e)}")
                    continue
        
        end_time = time.time()
        total_duration = end_time - start_time
        count = len(questions)
            
        dataset.update_output("pred", prediction_list)
        dataset = self.evaluate(dataset, do_eval=do_eval)
        avg_time = total_duration / count if count > 0 else 0

        print(f"\n[Timing] Total: {total_duration:.2f}s | Count: {count} | Avg per item: {avg_time:.4f}s")
        with open("records.txt", "a", encoding="utf-8") as f:
            f.write(f"\n[Timing] Total: {total_duration:.2f}s | Count: {count} | Avg per item: {avg_time:.4f}s")
        return dataset