import tomllib
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from typing import List
import re
import json


class ObjectDetector:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        with open('prompts/object_detector.toml', 'rb') as f:
            self.prompt = tomllib.load(f)

    def find_objects(self, question, image):
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt['user_prompt'].format(question=question)}
                ]
            }
        ]

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
        final_list = []
        try:
            pattern = r'(?:json)?\s*([\s\S]*?)\s*'
            matches = re.findall(pattern, response, re.DOTALL)

            for i, json_str in enumerate(matches):
                try:
                    data = json.loads(json_str)

                    if isinstance(data, list):
                        final_list.extend(data)
                    elif isinstance(data, dict):
                        final_list.append(data)
                    else:
                        print(f"ERROR: {data}")
                except json.JSONDecodeError as e:
                    print(f"代码块 {i+1}: 解析失败，可能不是有效的 JSON。错误: {e}")
                    continue
        except:
            print(f"ERROR: Fail to parse {response}")
            return None
        
        return final_list       

def main():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")    
    
    planner = ObjectDetector(model, processor)

    dataset = load_dataset("lmms-lab/OK-VQA", split='val2014')
    example = dataset[4]
    image = example['image'] 
    question = example['question']
    print(f"Answer: {example['answers']}")
    response = planner.find_objects(question=question, image=image)
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
