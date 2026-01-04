from utils import qwen2_generate
from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_retriever
import tomllib
from PIL import Image
import re
from tqdm import tqdm
import json
import torch
import time

class Instructor(BasicPipeline):
    def __init__(self, student, model, processor, config, retriever=None):
        super().__init__(config)
        if retriever is None:
            retriever = get_retriever(config)
        self.retriever = retriever
        self.model = model
        self.processor = processor
        self.student = student
        with open('prompts/instructor.toml', 'rb') as f:
            self.instructor_prompt = tomllib.load(f)

    def generate(self, question: str, image: Image):
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
                    {"type": "text", "text": self.instructor_prompt['system_prompt'].format(question=question, is_first_turn=True)},
                ],
            }
        ]
        
        instructor_response = qwen2_generate(self.model, self.processor, messages)
        print("\n<Instructor>")
        print(f"Instructor first Response: {instructor_response}")
        print("\n</Instructor>")
        logs.append({"instructor_first_response": instructor_response})
        state, plan = self.parse_from_instructor(instructor_response)
        messages.append({
                    "role": "assistant",
                    "content": [{
                        "type": "text", "text": f"state: {state}, plan: {plan}"
                    }]
                })
        if state != "continue":
            print(f"ERROR! The Instructor went on strike! Before he went off, he said: {instructor_response}")
            return "", messages
        else:
            conversation_num, max_turns = 0, 8
            while conversation_num < max_turns:
                # print("="*30)
                # print(f"INSTRUCTOR Prompt Contexts:\n{messages}")
                # print("="*30)
                print(f"State: {state}, Plan: {plan}")
                if state == "finish":
                    break
                # messages.append({'role': 'assistant', 'content': instructor_response})
                print("\n<Student>")
                feedback, student_logs = self.student.generate(question, image, plan)
                feedback_list.append(feedback)
                logs.append({"feedback": feedback})
                print("<Student>")
                print(f"Student feedback: {feedback}")
                print("</Student>")
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Feedback from user: {feedback}\n\n(Reminder: Use this feedback to move towards solving the MAIN QUESTION: '{question}'. If not fully solved, continue asking.)"}
                    ]
                })

                instructor_response = qwen2_generate(self.model, self.processor, messages)
                # instructor_response = self.check_student_response(feedback_list, plan)
                messages.append({
                    "role": "assistant",
                    "content": [{
                        "type": "text", "text": instructor_response
                    }]
                })
                logs.append({"instructor_response": instructor_response})
                print("\n<Instructor>")
                print(f"Instructor response after student: {instructor_response}")
                print("\n</Instructor>")
                state, plan = self.parse_from_instructor(instructor_response)
                conversation_num += 1

            final_answer = plan
            logs.append({"student_logs": student_logs})
            return final_answer, logs
        
    def check_student_response(self, feedback_list, plan):
        pass

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
                
        state_content = "continue" # 默认值
        plan_content = instructor_response # 默认值为原始文本，防止丢失信息

        if json_data and isinstance(json_data, dict):            
            if "state" in json_data:
                state_content = str(json_data["state"]).strip().lower()
            
            if "plan" in json_data:
                plan_content = str(json_data["plan"]).strip()
                
            # (可选) 打印思维链 Thought，用于调试
            if "thought" in json_data:
                print(f"[Instructor Thought]: {json_data['thought']}")
                
        else:
            # 3.2 解析彻底失败
            print("ERROR! No valid JSON object found in response.")
            # 这种情况下，我们通常假设 state 是 continue，
            # 并把整个回复当作 plan，或者记录错误日志
            state_content = "continue"
            plan_content = instructor_response

        # 最后的安全检查：确保 state 是我们需要的值
        if state_content not in ["continue", "finish"]:
            print(f"Warning: Unexpected state '{state_content}', defaulting to 'continue'")
            state_content = "continue"

        return state_content, plan_content
    
    def run(self, dataset, do_eval=True, pred_process_fun=None):
        questions = dataset.question
        ids = dataset.id
        prediction_list = []
        start_time = time.time()
        with open("intermediate_logs.jsonl", "w", encoding="utf-8") as f:
            for i, (question, id) in enumerate(zip(questions, ids)):
                print(f"[{i}/{len(questions)}] question: {question}")
                
                try:
                    img_path = f"data/datasets/crag/images/{id}.jpg"
                    img = Image.open(img_path).convert("RGB")
                    
                    final_answer, context = self.generate(question, img)
                    prediction_list.append(final_answer)
    
                    logs = {"id": id, "question": question, "prediction": final_answer, "logs": context}
                    f.write(json.dumps(logs, ensure_ascii=False) + "\n")
                    
                    if i % 10 == 0:
                        f.flush()
    
                except torch.cuda.OutOfMemoryError:
                    print(f"!!! CUDA OOM Error at index {i}, id: {id}. Clearing cache and skipping...")
                    
                    torch.cuda.empty_cache()
                    
                    prediction_list.append("Error: CUDA OOM")
                    
                    error_log = {"id": id, "question": question, "prediction": "I can't answer.", "logs": "CUDA OOM Error - Skipped"}
                    f.write(json.dumps(error_log, ensure_ascii=False) + "\n")
                    
                    continue
    
                # except Exception as e:
                #     print(f"!!! Unknown Error at index {i}, id: {id}: {str(e)}")
                #     prediction_list.append(f"Error: {str(e)}")
                #     continue
        
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