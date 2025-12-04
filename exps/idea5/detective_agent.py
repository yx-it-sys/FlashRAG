from agents.object_detector import ObjectDetector
import tomllib
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from retriever import search
from object_processor import object_expansion

class Detective:
    def __init__(self, model, processor, max_loop=5):
        self.model = model
        self.processor = processor
        self.max_loop = max_loop
        with open('prompts/detective.toml', 'rb') as f:
            self.prompt = tomllib.load(f)
        self.object_detector = ObjectDetector(self.model, self.processor)

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
        while loop <= self.max_loop:
            response = self.generate(messages)
            print(f"Loop {loop} response:\n {response}")
            if "Final Answer" in response:
                final_answer = response.split("Final Answer:")[-1].strip()
                return final_answer
            
            need_txt_ret = "Text Retrieval" in response
            need_evidence_info = "Evidence Retrieval" in response

            if need_txt_ret or need_evidence_info:
                if need_txt_ret:
                    query_text = (
                        response.split("Text Retrieval")[-1]
                        .replace(":", "")
                        .replace('"', "")
                        .replace(">", "")
                    )
                    search_text = search(query_text, type="dense")
                    print(f"Contents of retrieved documents:\n{search_text}")
                else:
                    query_text = (
                        response.split("Evidence Retrieval")[-1]
                        .replace(":", "")
                        .replace('"', "")
                        .replace(">", "")
                    )
                    object_features = object_feature_expansion[query_text]
                    object_retrieved = object_retrieval_expansion[query_text]
                    search_evidence = f"Feature of {query_text}:\n{object_features}\nMore information on {query_text}:\n{object_retrieved}"
                    print(f"Contents of evidence from image:\n{search_evidence}")

                contents = []
                if search_text:
                    contents.append("Contents of retrieved documents:")
                    contents.extend(search_text)
                elif search_evidence:
                    contents.append("Contents of evidence from image:")
                    contents.extend(search_evidence)
                else:
                    contents.append("No relevant information found.")
            
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "\n".join(contents)}
                    ]
                }
            )
            loop += 1

        return response


