from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForVision2Seq
import os

# # --- 定义模型名称和本地保存路径 ---
# model_name = 'Qwen/Qwen2.5-VL-7B-Instruct'
# # 你可以自定义这个文件夹名称
# local_model_path = '/mnt/data/models' 


# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
# model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
#             {"type": "text", "text": "What animal is on the candy?"}
#         ]
#     },
# ]
# inputs = processor.apply_chat_template(
# 	messages,
# 	add_generation_prompt=True,
# 	tokenize=True,
# 	return_dict=True,
# 	return_tensors="pt",
# ).to(model.device)

# outputs = model.generate(**inputs, max_new_tokens=40)
# print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="dev")