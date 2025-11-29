import json
from datasets import load_dataset
import os
from tqdm import tqdm

num_samples = 100
seed = 42
full_dataset = load_dataset("lmms-lab/OK-VQA", split='val2024')
full_dataset = full_dataset.shuffle(seed=seed)

dummy_subset = full_dataset.select(range(num_samples))

annotations_dir = os.path.dirname("baseline/data/datasets/okvqa")
img_dir = os.path.join(annotations_dir, "images")

os.makedirs(annotations_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

processed_data = []
for item in full_dataset:
    image = item['image']
    img_filename = f"{item['question_id']}.jpg"
    img_filepath = os.path.join(img_dir, img_filename)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image.save(img_filepath, 'JPEG')

    clean_item = {
        "id": item["id"],
        "question": item["question"],
        "golden_answers": item["golden_answers"]
    }
    processed_data.append(clean_item)

print("所有图片保存完成！")
output_filename = "test.jsonl"
output_path = os.path.join(annotations_dir, output_filename)

with open(output_path, 'w', encoding='utf-8') as f:
    for entry in tqdm(processed_data, desc="Writing jsonl"):
        if not isinstance(entry, dict):
            try:
                entry = dict(entry)
            except Exception:
                continue
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Saved {len(processed_data)} entries to {output_path}")
