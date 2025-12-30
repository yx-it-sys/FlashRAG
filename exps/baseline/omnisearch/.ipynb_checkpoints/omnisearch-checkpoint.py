from flashrag.evaluator import Evaluator
from flashrag.utils import get_retriever, get_generator
from flashrag.pipeline import BasicMultiModalPipeline
import re
import os
import json
from PIL import Image
import tomllib
from tqdm import tqdm

class OmniSearchPipeline(BasicMultiModalPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template)
        self.config = config
        self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever
        prompt_path = self.config['omni_vqa_prompt_path']
        with open(prompt_path, 'rb') as f:
            self.prompt = tomllib.load(f)['system_prompts']['multimodal_qa_omni']
    
    def iterative_infer(self, question, id):
        img_path = f"data/datasets/crag/images/{id}.jpg"
        img = Image.open(img_path).convert("RGB")
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": self.prompt},
            ]},
            {"role": "user", "content":[
                {"type": "text", "text": f"Input Question: {question}"},
                {"type": "image", "image": img}
            ]}

        ]
        response = self.generator.generate(messages)
        print(f"First Response: {response}")
        messages.append({'role': 'assistant', 'content': response})
        
        conversation_num, max_turns = 0, 5
        while conversation_num < max_turns:
            if "Final Answer" in response or "<Final Answer>" in response:
                break
            need_txt_ret = "Text Retrieval" in response
            need_img_ret = "Retrieval with Input Image" in response
            if need_txt_ret or need_img_ret:
                retrieval_content = ""
                if need_txt_ret:
                    print("Start Text Retrieval...")
                    pattern = r'Text Retrieval[:\s"]*(.*?)(?=<|$)'
                    match = re.search(pattern, response, re.DOTALL)
                    query_txt = ""
                    if match:
                        query_txt = match.group(1).strip()
                        print(f"Query Text: {query_txt}")
                    if query_txt == "":
                        print(f"ERROR!!Query_txt is None")
                        search_text = self.retriever.search_by_text(query_txt)
                        retrieval_content = "\n\n".join([f"Doc{i+1}:\n{text}" for i, text in enumerate(search_text)])
                        # print(f"Retrieval result: {retrieval_content}")
                    else:
                        search_text = self.retriever.search_by_text(query_txt)
                        retrieval_content = "\n\n".join([f"Doc{i+1}:\n{text}" for i, text in enumerate(search_text)])
                        # print(f"Retrieval result: {retrieval_content}")
                else:
                    print("Start Image Retrieval...")
                    search_text = self.retriever.search_by_image(img)
                    retrieval_content = "\n\n".join([f"Doc{i+1}:\n{text}" for i, text in enumerate(search_text)])
                    # print(f"Retrieval result: {retrieval_content}")

                contents = []
                if retrieval_content:
                    contents.append({'type': 'text', 'text': f"Contents of retrieved documents:\n{retrieval_content}"})
                else:
                    contents.append({'type': 'text', 'text': "No relevant information found."})    

                messages.append({'role': 'user', 'content': contents})

                try:
                    response = self.generator.generate(messages)
                    print(f"Response: {response}")
                    messages.append({"role":"assistant", "content": response})
                except Exception as e:
                    print("Inference error, hidden states ignored:", e)
                    return response, messages
            else:
                conversation_num += 1
                break
            conversation_num += 1
        
        pattern = r'(?:<Final Answer>|Final Answer:)\s*(.*?)(?=<|$)'
        final_answer_match = re.search(pattern, response, re.DOTALL)

        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            final_answer = final_answer.replace('\n', '')
            print(f"Final Answer: {final_answer}")
            return final_answer, messages
        else:
            print(f"Warning: reached end of agent loop for item {conversation_num} without a 'Final Answer'. returning last response")
            return response, messages
      
    def safe_write(self, file_path: str, data: dict):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    import time
    def run(self, dataset, do_eval=True, pred_process_fun=None):
        questions = dataset.question
        ids = dataset.id
        prediction_list = []

        # 2. 记录开始时间
        start_time = time.time()

        for question, id in tqdm(zip(questions, ids), total=len(questions)):
            answer, context = self.iterative_infer(question, id)
            prediction_list.append(answer)

        dataset.update_output("pred", prediction_list)
        dataset = self.evaluate(dataset, do_eval=do_eval)
        end_time = time.time()
        total_duration = end_time - start_time
        count = len(questions)
        
        avg_time = total_duration / count if count > 0 else 0

        print(f"\n[Timing] Total: {total_duration:.2f}s | Count: {count} | Avg per item: {avg_time:.4f}s")
        with open("records.txt", "a", encoding="utf-8") as f:
            f.write(f"\n[Timing] Total: {total_duration:.2f}s | Count: {count} | Avg per item: {avg_time:.4f}s")
        return dataset