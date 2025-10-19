from PIL import Image
import numpy as np
from flashrag.utils import get_generator
import json
import warnings
import os

class DisturbImage():
    def __init__(self, config, prompt_template, method, threshold, generator=None):
        self.config = config
        self.prompt_template = prompt_template
        if method in ['average', 'cluster']:
            self.method = method
        else:
            print(f"{method} is not supported by this tools, set to default: 'cluster'.")
            self.method = "cluster"
        
        self.threshold = threshold
        self.generator = get_generator(config) if generator is None else generator
    
    def add_salt_and_pepper_noise_numpy(image_pil, step=5):
        image_np = np.array(image_pil)
        if not (step <= 10):
            warnings.warn(f"step: {step} has out of numerical boundary, set to default: 5")
            step = 5
        image_pil_list = []
        for i in range(step):
            k = i / step
            noisy_image = image_np.copy()
            height, width = noisy_image[:2]
            num_noise_pixels = int (k * height * width)

            salt_coords = [np.random.randint(0, j - 1, int(num_noise_pixels / 2)) for j in (height, width)]
            noisy_image[salt_coords[0], salt_coords[1], :] = 255

            pepper_coords = [np.random.randint(0, i - 1, int(num_noise_pixels / 2)) for i in (height, width)]
            noisy_image[pepper_coords[0], pepper_coords[1], :] = 0

            noisy_image = Image.fromarray(noisy_image)
            image_pil_list.append(noisy_image)

            return image_pil_list
    
    def consistency(self, answers_list, method, threshold):
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        bert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        embeddings = bert_model.encode(answers_list)
        similarity_matrix = cosine_similarity(embeddings)
        scores = similarity_matrix[np.triu_indices(len(answers_list), k=1)]
        if method == "average":
            return np.mean(scores) if len(scores) > 0 else 1.0
        elif method == "cluster":
            num_consist = 0
            for score in scores:
                if score >= threshold:
                    num_consist += 1
            return num_consist / len(answers_list)

    def generate_answers(self, dataset, step=5, disturb_type=None):
        if disturb_type is None:
            data_items_list = []
            data_list = list(dataset)
            for data in data_list:
                data_items_list.append({
                    "id": data.id,
                    "question": data.question,
                    "answers": data.golden_answers,
                })
            question_d_i_list = []
            for data in data_items_list:
                id = data['id']
                image_path = os.path.join(self.config["image_path"], f"{data['id']}.jpg")
                image_pil = Image.open(image_path).convert('RGB')
                image_pil_list = self.add_salt_and_pepper_noise_numpy(image_pil, step)
                question = data['question']
                golden_answers = data['answers']
                question_d_i_list.append({
                    'id': id,
                    'question': question,
                    'golden_answers': golden_answers,
                    'disturbed_images': image_pil_list
                })
            
            question_d_i_list_with_confidence_score = []
            for q_d_i_pair in question_d_i_list:
                pred_answers_list = []
                for i, image in enumerate(q_d_i_pair['disturbed_images']):
                    response_dict = self.generator.generate([self.prompt_template.get_string(question, image)], uncertainty_type=None)
                    response = response_dict[0]['output_text'][0]
                    pred_answers_list.append(response)
                    question_d_i_list_with_confidence_score.append({
                        'id': q_d_i_pair['id'],
                        'question': q_d_i_pair['question'],
                        'golden_answers': q_d_i_pair['golden_answers'],
                        'consistent': self.consistency(pred_answers_list, method=self.method, k=self.threshold)
                    })
            # 保存至output.jsonl
            result_data = {}
            for item in question_d_i_list_with_confidence_score:
                result_data["id"] = item["id"]
                result_data['question'] = item['question']
                result_data['golden_answers'] = item['golden_answers']
                result_data['confidence_score'] = item['consistent']
                file_path = os.path.join(self.config["save_dir"], "output.jsonl")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
        print(f"Successfully calculating confidence scores via disturbing images, see outputs in {file_path}.")
