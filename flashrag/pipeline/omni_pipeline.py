from flashrag.evaluator import Evaluator
from flashrag.utils import get_retriever, get_generator
from flashrag.pipeline import BasicMultiModalPipeline
import re
import os
import json
import base64
from PIL import Image
import tomllib
from tqdm import tqdm
import time
import openai

class OmniSearchPipeline(BasicMultiModalPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template)
        self.config = config
        self.data_dir = self.config['data_dir']
        self.dataset_name = self.config['dataset_name']
        self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever
        prompt_path = self.config['omni_prompt_path']
        with open(prompt_path, 'rb') as f:
            self.prompt = tomllib.load(f)['system_prompt']
        self.output_dir = self.config['output_dir'] if 'output_dir' in self.config and self.config['output_dir'] else self.config['save_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        self.trajectory_path = os.path.join(self.output_dir, "omnisearch_trajectories.jsonl")

    def _normalize_generation_output(self, response):
        if isinstance(response, str):
            return response
        if isinstance(response, list):
            flattened = []
            for item in response:
                normalized = self._normalize_generation_output(item)
                if normalized:
                    flattened.append(normalized)
            return "\n".join(flattened).strip()
        if response is None:
            return ""
        return str(response)

    def _generate_text(self, messages):
        delay = 2
        max_retries = 5
        for attempt in range(max_retries):
            try:
                raw_response = self.generator.generate(messages)
                return raw_response
            except openai.RateLimitError as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Rate limit hit, retrying in {delay}s ...")
                time.sleep(delay)
                delay *= 2
        raw_response = self.generator.generate(messages)
        return self._normalize_generation_output(raw_response)

    def _format_retrieved_doc(self, doc):
        if isinstance(doc, str):
            return doc
        if isinstance(doc, dict):
            for key in ("contents", "text", "content", "body", "passage"):
                value = doc.get(key)
                if value:
                    return str(value)
            return json.dumps(doc, ensure_ascii=False)
        return str(doc)

    def _search_text_docs(self, query_txt):
        return self.retriever.search(query_txt, query_type="text")

    def _search_image_docs(self, img):
        return self.retriever.search(img, query_type="image")

    def _serialize_for_log(self, value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): self._serialize_for_log(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._serialize_for_log(v) for v in value]
        if isinstance(value, Image.Image):
            return {"type": "pil_image", "size": list(value.size), "mode": value.mode}
        return str(value)

    def _extract_action_nodes(self, response):
        if not response:
            return []

        nodes = []
        tag_pattern = re.compile(r"<Thought>|<Sub-Question>|<Search>|<End>")
        matches = list(tag_pattern.finditer(response))
        action_name_map = {
            "<Thought>": "thought",
            "<Sub-Question>": "sub-question",
            "<Search>": "search",
        }

        for idx, match in enumerate(matches):
            tag = match.group(0)
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(response)
            content = response[start:end].strip()

            if tag in action_name_map and content:
                nodes.append({
                    "action": action_name_map[tag],
                    "content": content,
                })
            elif tag == "<End>":
                final_answer_match = re.search(r"Final Answer:\s*(.*?)(?=$)", content, re.DOTALL)
                if final_answer_match:
                    final_answer = final_answer_match.group(1).strip().replace("\n", "")
                    if final_answer:
                        nodes.append({
                            "action": "final_answer",
                            "content": final_answer,
                        })

        if not nodes:
            final_answer_match = re.search(r"Final Answer:\s*(.*?)(?=$)", response, re.DOTALL)
            if final_answer_match:
                final_answer = final_answer_match.group(1).strip().replace("\n", "")
                if final_answer:
                    nodes.append({
                        "action": "final_answer",
                        "content": final_answer,
                    })
        return nodes

    def _record_response_actions(self, trajectory, response):
        trajectory.extend(self._extract_action_nodes(response))

    def _write_trajectory(self, record):
        self.safe_write(self.trajectory_path, self._serialize_for_log(record))
    
    def turn_to_url(self, image_path):
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{base64_image}"

    def iterative_infer(self, question, id, image_id):
        img_path = os.path.join(self.data_dir, self.dataset_name, "images", f"{image_id}.jpg")
        
        if not os.path.exists(img_path):
            return None
        
        img_url = self.turn_to_url(img_path)
        img = Image.open(img_path).convert("RGB")
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": self.prompt},
            ]},
            {"role": "user", "content":[
                {"type": "text", "text": f"Input Question: {question}"},
                {"type":  "image_url", "image_url": {"url": img_url}}
            ]}

        ]
        trajectory = []
        start_time = time.time()
        response = self._generate_text(messages)
        print(f"First Response: {response}")
        self._record_response_actions(trajectory, response)
        messages.append({'role': 'assistant', 'content': response})
        
        conversation_num, max_turns = 0, 5
        while conversation_num < max_turns:
            if "Final Answer" in response or "<Final Answer>" in response:
                break
            need_txt_ret = "Text Retrieval" in response
            need_img_ret = "Image Retrieval" in response
            need_no_ret = "No Retrieval" in response
            if need_txt_ret or need_img_ret or need_no_ret:
                retrieval_content = ""
                query_txt = ""
                action = "text_retrieval" if need_txt_ret else "image_retrieval"
                retrieved_docs = []
                if need_txt_ret:
                    print("Start Text Retrieval...")
                    pattern = r'Text Retrieval[:\s"]*(.*?)(?=<|$)'
                    match = re.search(pattern, response, re.DOTALL)
                    if match:
                        query_txt = match.group(1).strip()
                        print(f"Query Text: {query_txt}")
                    if query_txt == "":
                        print(f"ERROR!!Query_txt is None")
                        retrieved_docs = self._search_text_docs(query_txt)
                        retrieval_content = "\n\n".join(
                            [f"Doc{i+1}:\n{self._format_retrieved_doc(text)}" for i, text in enumerate(retrieved_docs)]
                        )
                        # print(f"Retrieval result: {retrieval_content}")
                    else:
                        retrieved_docs = self._search_text_docs(query_txt)
                        retrieval_content = "\n\n".join(
                            [f"Doc{i+1}:\n{self._format_retrieved_doc(text)}" for i, text in enumerate(retrieved_docs)]
                        )
                        print(f"Retrieval result: {retrieval_content[:100]}")
                elif need_img_ret:
                    print("Start Image Retrieval...")
                    retrieved_content = self._search_image_docs(img)
                    print(f"Retrieval result: {retrieved_content[:100]}")
                elif need_no_ret:
                    retrieval_content = None

                contents = []
                if retrieval_content:
                    contents.append({'type': 'text', 'text': f"Contents of retrieved documents:\n{retrieval_content}"})
                else:
                    contents.append({'type': 'text', 'text': "No relevant information found."})    

                messages.append({'role': 'user', 'content': contents})

                try:
                    response = self._generate_text(messages)
                    print(f"Response: {response}")
                    self._record_response_actions(trajectory, response)
                    messages.append({"role":"assistant", "content": response})
                except Exception as e:
                    print("Inference error, hidden states ignored:", e)
                    self._write_trajectory({
                        "question": question,
                        "id": id,
                        "final_answer": response,
                        "status": "generation_error",
                        "duration_seconds": time.time() - start_time,
                        "trajectory": trajectory,
                    })
                    return response, messages
                       
            conversation_num += 1
        
        pattern = r'(?:<Final Answer>|Final Answer:)\s*(.*?)(?=<|$)'
        final_answer_match = re.search(pattern, response, re.DOTALL)

        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            final_answer = final_answer.replace('\n', '')
            print(f"Final Answer: {final_answer}")
            self._write_trajectory({
                "question": question,
                "id": id,
                "final_answer": final_answer,
                "status": "ok",
                "duration_seconds": time.time() - start_time,
                "trajectory": trajectory,
            })
            return final_answer, messages
        else:
            print(f"Warning: reached end of agent loop for item {conversation_num} without a 'Final Answer'. returning last response")
            self._write_trajectory({
                "question": question,
                "id": id,
                "final_answer": response,
                "status": "missing_final_answer",
                "duration_seconds": time.time() - start_time,
                "trajectory": trajectory,
            })
            return response, messages
      
    def safe_write(self, file_path: str, data: dict):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        questions = dataset.question
        image_ids = dataset.image_id
        ids = dataset.id
        prediction_list = []

        # 2. 记录开始时间
        start_time = time.time()

        for question, id , image_id in tqdm(zip(questions, ids, image_ids), total=len(questions)):
            answer, context = self.iterative_infer(question, id, image_id)
            prediction_list.append(answer)

        dataset.update_output("pred", prediction_list)
        dataset = self.evaluate(dataset, do_eval=do_eval)
        end_time = time.time()
        total_duration = end_time - start_time
        count = len(questions)
        
        avg_time = total_duration / count if count > 0 else 0

        print(f"\n[Timing] Total: {total_duration:.2f}s | Count: {count} | Avg per item: {avg_time:.4f}s")
        with open(os.path.join(self.output_dir, "records.txt"), "a", encoding="utf-8") as f:
            f.write(f"\n[Timing] Total: {total_duration:.2f}s | Count: {count} | Avg per item: {avg_time:.4f}s")
        return dataset
