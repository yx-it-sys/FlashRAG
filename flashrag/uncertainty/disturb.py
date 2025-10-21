from PIL import Image, ImageFilter, ImageDraw
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
    
    def Guasian_blurring(self, image_pil, step=5):
        blurred_images = []
        for i in range(step):
            blur_radius = i * 2
            blurred_image = image_pil.filter(
                ImageFilter.GaussianBlur(radius=blur_radius)
            )
            blurred_images.append(blurred_image)
        # save blurred images to disk
        out_dir = os.path.join('/mnt/data/results_disturb_images', 'blurred')
        os.makedirs(out_dir, exist_ok=True)
        for idx, img in enumerate(blurred_images, start=1):
            path = os.path.join(out_dir, f'blurred_{idx}.jpg')
            try:
                img.save(path, format='JPEG')
            except Exception:
                img.convert('RGB').save(path, format='JPEG')
        return blurred_images
    
    def add_salt_and_pepper_noise_numpy(self, image_pil, step=5):
        image_np = np.array(image_pil)
        if not (step <= 10):
            warnings.warn(f"step: {step} has out of numerical boundary, set to default: 5")
            step = 5
        image_pil_list = []
        for i in range(step):
            k = i / step
            noisy_image = image_np.copy()
            height, width = noisy_image.shape[:2]
            num_noise_pixels = int (k * height * width)

            salt_coords = [np.random.randint(0, j - 1, int(num_noise_pixels / 2)) for j in (height, width)]
            noisy_image[salt_coords[0], salt_coords[1], :] = 255

            pepper_coords = [np.random.randint(0, i - 1, int(num_noise_pixels / 2)) for i in (height, width)]
            noisy_image[pepper_coords[0], pepper_coords[1], :] = 0

            noisy_image = Image.fromarray(noisy_image)
            image_pil_list.append(noisy_image)
        
        # save blurred images to disk
        out_dir = os.path.join('/mnt/data/results_disturb_images', 'pepper')
        os.makedirs(out_dir, exist_ok=True)
        for idx, img in enumerate(image_pil_list, start=1):
            path = os.path.join(out_dir, f'pepper_{idx}.jpg')
            try:
                img.save(path, format='JPEG')
            except Exception:
                img.convert('RGB').save(path, format='JPEG')
        return image_pil_list
    
    def consistency(self, answers_list, method, threshold):
        from sentence_transformers import SentenceTransformer, util
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        bert_model = SentenceTransformer('/mnt/data/models')
        embeddings = bert_model.encode(answers_list)
        similarity_matrix = cosine_similarity(embeddings)
        scores = similarity_matrix[np.triu_indices(len(answers_list), k=1)]
        print(f"scores: {scores}")
        if method == "average":
            return np.mean(scores) if len(scores) > 0 else 1.0
        elif method == "cluster":
            clusters = util.community_detection(embeddings, threshold=threshold, min_community_size=1)
            if not clusters:
                largest_cluster_size = 1
            else:
                largest_cluster = max(clusters, key=len)
                largest_cluster_size = len(largest_cluster)
            total_elements = len(answers_list)
            ratio = largest_cluster_size / total_elements
            print(f"ratio: {ratio}")
            return ratio


    def generate_answers(self, list_dataset, step=5, disturb_type=None):
        if disturb_type is None:
            data_items_list = []
            for data in list_dataset:
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
                # image_pil_list = self.Guasian_blurring(image_pil=image_pil, step=step)
                question = data['question']
                golden_answers = data['answers']
                question_d_i_list.append({
                    'id': id,
                    'question': question,
                    'golden_answers': golden_answers,
                    'disturbed_images': image_pil_list
                })
            
            for q_d_i_pair in question_d_i_list:
                print(f"now answering quastion: {q_d_i_pair['id']}")
                pred_answers_list = []
                for i, image in enumerate(q_d_i_pair['disturbed_images']):
                    prompt = self.prompt_template.add_question_and_image(input_question=q_d_i_pair['question'], pil_image=image)
                    response_dict = self.generator.generate([prompt], uncertainty_type=None)
                    response = response_dict[0]['output_text'][0]
                    pred_answers_list.append(response)
                    del prompt
                    del response_dict
                    del response
                print(f"pred answers list: {pred_answers_list}")
                consistency_score = self.consistency(pred_answers_list, method=self.method, threshold=self.threshold),
                question_d_i_list_with_confidence_score = {
                    'id': q_d_i_pair['id'],
                    'question': q_d_i_pair['question'],
                    'golden_answers': q_d_i_pair['golden_answers'],
                    'consistent': consistency_score,
                    'prediction': pred_answers_list[0]
                }
                # 保存至output.jsonl
                file_path = os.path.join('results_disturb_images/gausian_blur', "output.jsonl")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(question_d_i_list_with_confidence_score, ensure_ascii=False) + "\n")
        print(f"Successfully calculating confidence scores via disturbing images, see outputs in {file_path}.")
