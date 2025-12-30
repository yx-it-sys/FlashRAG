from datasets import load_dataset
import json
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm

ds = load_dataset("izhx/UMRB-EncyclopediaVQA", "corpus", split="corpus")
img_dir = "images"
os.makedirs(img_dir, exist_ok=True)

df = pd.read_csv('test.csv', nrows=3751)
answer_pairs = {}
for q, a in zip(df['question'], df['answer']):
    answer_pairs[q] = a

processed_data = []
for item in ds:
    image = item['image']
    img_filename = f"{item['id']}.jpg"
    img_filepath = os.path.join(img_dir, img_filename)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image.save(img_filepath, 'JPEG')

    answer = answer_pairs[item['text']]
    clean_item = {
        "id": item["id"],
        "question": item["text"],
        "golden_answers": answer
    }
    processed_data.append(clean_item)

output_filename = "test.jsonl"

with open(output_filename, 'w', encoding='utf-8') as f:
    for entry in tqdm(processed_data, desc="Writing jsonl"):
        if not isinstance(entry, dict):
            try:
                entry = dict(entry)
            except Exception:
                continue
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Saved {len(processed_data)} entries to {output_filename}")