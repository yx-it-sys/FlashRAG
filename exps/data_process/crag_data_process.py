import json
from datasets import load_dataset
import os
from tqdm import tqdm
from io import BytesIO
import requests
from PIL import Image

def load_image_from_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        response.raise_for_status()
        
        image_data = BytesIO(response.content)
        
        image = Image.open(image_data)
        return image

    except Exception as e:
        print(f"获取图片失败: {e}")
        return None
        
def main():
    full_dataset = load_dataset("crag-mm-2025/crag-mm-single-turn-public", split='validation')
    # start_index = 1326
    # total_len = len(full_dataset)
    
    # subset = full_dataset.select(range(start_index, total_len))
    # num_samples = 100
    # seed = 42
    # full_dataset = full_dataset.shuffle(seed=seed)  
    # dummy_subset = full_dataset.select(range(num_samples))
    
    annotations_dir = os.path.dirname("E:/实验/data/datasets/crag")
    img_dir = os.path.join(annotations_dir, "images")
    
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    processed_data = []
    for item in tqdm(full_dataset):
        image = item['image']
        target_url = item['image_url']
        img_filename = f"{item['session_id']}.jpg"
        img_filepath = os.path.join(img_dir, img_filename)
        if image is None:
            image = load_image_from_url(target_url)
        if image.mode != 'RGB':
            image = image.convert('RGB')
    
        image.save(img_filepath, 'JPEG')
    
        clean_item = {
            "id": item["session_id"],
            "question": item["turns"]["query"][0],
            "golden_answers": item["answers"]["ans_full"]
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

if __name__ == "__main__":
    main()
