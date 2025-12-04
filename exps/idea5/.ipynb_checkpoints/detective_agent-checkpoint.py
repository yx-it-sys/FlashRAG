from object_detector import ObjectDetector
import tomllib
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from object_processor import ObjectProcessor
import os
from PIL import Image
import json
import re

class Detective:
    def __init__(self, model, processor, retriever, config, max_loop=5):
        self.config = config
        self.model = model
        self.processor = processor
        self.max_loop = max_loop
        with open('prompts/detective.toml', 'rb') as f:
            self.prompt = tomllib.load(f)
        self.retriever = retriever
        self.object_detector = ObjectDetector(self.model, self.processor)
        self.object_processor = ObjectProcessor(self.model, self.processor, self.retriever)

    def generate(self, messages):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=32768,
            do_sample=False,  
            temperature=0.0   
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = output_text[0]
        return response
    
    def inference(self, objects, object_feature_expansion, object_retrieval_expansion, question, image):
        all_logs = {}
        all_logs["query"] = question
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": self.prompt['system_prompt'].format(objects=objects)}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt["user_prompt"].format(question=question)}
                ]
            }
        ]
        loop = 0
        logs = []
        while loop <= self.max_loop:
            log = {}
            response = self.generate(messages)
            messages.append({"role": "assistant", "content": response})
            print(f"Loop {loop} response:\n {response}")
            log["raw_response"]: response
            if "Final Answer" in response:
                final_answer = response.split("Final Answer:")[-1].strip()
                log["final_answer"] = final_answer
                logs.append(log)
                all_logs['logs'] = logs
                return final_answer, all_logs
            
            need_txt_ret = "Text Retrieval" in response
            need_evidence_info = "Evidence Retrieval" in response

            contents = []
            search_text = None
            if need_txt_ret or need_evidence_info:
                if need_txt_ret:
                    pattern = r"<search>(.*?)</search>"
                    matches = re.findall(pattern, response, re.DOTALL)[0]    
                    query_text = matches.split("Text Retrieval:")[-1].strip()
                    search_text = [text['contents'] for text in self.retriever.search(query_text)]
                    search_text = "\n".join(search_text)
                    print(f"Contents of retrieved documents:\n{search_text}")
                    log["need_txt_ret"] = search_text
                else:
                    query_text_list = []
                    pattern = r"<evidence_retrieval>(.*?)</evidence_retrieval>"
                    matches = re.findall(pattern, response, re.DOTALL)                    
                    for i, m in enumerate(matches):
                        m = m.strip().split("Evidence Retrieval:")[-1].strip()
                        query_text_list.append(m)
                    object_features_list = []
                    object_retrieved_list = []
                    for m in query_text_list:           
                        object_features_list.append(object_feature_expansion[m])
                        object_retrieved_list.append(object_retrieval_expansion[m])
                    search_evidence_list = []
                    for m, (n, p) in zip(query_text_list, zip(object_features_list, object_retrieved_list)):
                        search_evidence_list.append(f"Feature of {m}:\n{n}\nMore information on {m}:\n{p}")
                    evidence = "\n".join(search_evidence_list)
                    print(f"Contents of evidence from image:\n{evidence}")
                    log["need_evidence_ret"] = evidence

                if search_text:
                    contents.append("Contents of retrieved documents:")
                    contents.extend(search_text)
                elif search_evidence_list:
                    contents.append("Contents of evidence from image:")
                    contents.extend(search_evidence_list)
                else:
                    contents.append("No relevant information found.")
            print(f"Contents:{contents}")
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "\n".join(contents)}
                    ]
                }
            )
            loop += 1
            logs.append(log)
        
        all_logs['logs'] = logs
        return response, all_logs

    def run(self, dataset, do_eval=True, pred_process_func=None):
        pred_answers_list = []
        for item in dataset:
            question = item.question
            id = item.id
            img_path = f"{self.config['data_dir']}/{self.config['dataset_name']}/images/{id}.jpg"
            image = Image.open(img_path)

            objects = self.object_detector.find_objects(question, image)
            object_feature_expansion, object_retrieval_expansion = self.object_processor.object_expansion(objects)

            response, logs = self.inference(objects, object_feature_expansion, object_retrieval_expansion, question, image)

            result_data = {}
            result_data['id'] = item.id
            result_data['question'] = item.question
            result_data['golden_answers'] = item.golden_answers
            result_data['prediction'] = response
            result_data['logs'] = logs
            pred_answers_list.append(response)
            file_path = os.path.join(self.config['save_dir'], "output.jsonl")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
        dataset.update_output("pred", pred_answers_list)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_func=pred_process_func)
        return dataset


