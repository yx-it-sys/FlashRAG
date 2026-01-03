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
                    {"type": "text", "text": self.instructor_prompt['system_prompt'].format(question=question)},
                ],
            }
        ]
        print("\n<Instructor>")
        instructor_response = qwen2_generate(self.model, self.processor, messages)
        print(f"Instructor first Response: {instructor_response}")
        logs.append({"instructor_first_response": instructor_response})
        state, plan = self.parse_from_instructor(instructor_response)
        messages.append({
                    "role": "assistant",
                    "content": [{
                        "type": "text", "text": f"state: {state}, plan: {plan}"
                    }]
                })
        if state != "insufficient":
            print(f"ERROR! The Instructor went on strike! Before he went off, he said: {instructor_response}")
            return "", messages
        else:
            conversation_num, max_turns = 0, 8
            while conversation_num < max_turns:
                print("="*30)
                print(f"INSTRUCTOR Prompt Contexts:\n{messages}")
                print("="*30)
                if state == "sufficient":
                    break
                # messages.append({'role': 'assistant', 'content': instructor_response})
                print("\n<Student>")
                feedback, student_logs = self.student.generate(question, image, plan)
                feedback_list.append(feedback)
                logs.append({"feedback": feedback})
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Feedback from student on the current plan {plan}:\n{feedback}"}
                    ]
                })
                print("\n<Instructor>")
                instructor_response = qwen2_generate(self.model, self.processor, messages)
                # instructor_response = self.check_student_response(feedback_list, plan)
                messages.append({
                    "role": "assistant",
                    "content": [{
                        "type": "text", "text": instructor_response
                    }]
                })
                logs.append({"instructor_response": instructor_response})
                print(f"Instructor response after student: {instructor_response}")
                state, plan = self.parse_from_instructor(instructor_response)
                conversation_num += 1

            final_answer = plan
            context = messages
            logs.append({"student_logs": student_logs})
            return final_answer, logs
        
    def check_student_response(self, feedback_list, plan):
        pass

    def parse_from_instructor(self, instructor_response):
        # ==========================================
        # 策略 1: 尝试解析 XML 风格标签 (<state>...</state>)
        # ==========================================
        
        # 1.1 解析 XML State
        # 使用正向前瞻 (?=...) 兼容闭合标签缺失或直接衔接 <plan> 的情况
        state_pattern_xml = r"<state>\s*(.*?)\s*(?=</state>|<plan>|$)"
        state_match_xml = re.search(state_pattern_xml, instructor_response, re.DOTALL | re.IGNORECASE)
        
        state_content = None
        if state_match_xml:
            state_content = state_match_xml.group(1).strip()
    
        # 1.2 解析 XML Plan
        plan_pattern_xml = r"<plan>\s*(.*?)\s*(?=</plan>|$)"
        plan_match_xml = re.search(plan_pattern_xml, instructor_response, re.DOTALL | re.IGNORECASE)
        
        plan_content = None
        if plan_match_xml:
            plan_content = plan_match_xml.group(1).strip()
    
        if state_content is None:
            # print("XML format failed, switching to Text-Key-Value parsing...")
            
            # 2.1 解析 Text State
            # 查找 "state:" 开头，直到遇到 "plan:" 或 字符串结尾
            state_pattern_text = r"state:\s*(.*?)\s*(?=plan:|$)"
            state_match_text = re.search(state_pattern_text, instructor_response, re.DOTALL | re.IGNORECASE)
            
            if state_match_text:
                state_content = state_match_text.group(1).strip()
                
            # 2.2 解析 Text Plan
            # 查找 "plan:" 开头，直到字符串结尾
            plan_pattern_text = r"plan:\s*(.*)"
            plan_match_text = re.search(plan_pattern_text, instructor_response, re.DOTALL | re.IGNORECASE)
            
            if plan_match_text:
                plan_content = plan_match_text.group(1).strip()
    
        
        if state_content is None:
            print("ERROR! State not found in either XML or Text format")
            state_content = "insufficient"
    
        if plan_content is None:
            # 如果两种策略都没找到 Plan，通常意味着模型输出了非结构化的纯文本
            # 为了不中断流程，将整个回复作为 Plan
            print("ERROR! Plan not found, using full response")
            state_content = "insufficient"
            plan_content = instructor_response
    
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