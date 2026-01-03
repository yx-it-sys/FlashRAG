from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from cragmm_search.search import UnifiedSearchPipeline
import torch
import re

def qwen2_generate(model, processor, messages):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs, 
        padding=True,
        return_tensors="pt",
    )
    
    del image_inputs
    del video_inputs

    inputs = inputs.to("cuda")
    
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.7,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )
    
    input_len = inputs.input_ids.shape[1]
        
    generated_ids_trimmed = [
        out_ids[input_len:] for out_ids in generated_ids
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]
    
class CRAGSearch():
    def __init__(self, top_k):
        self.search_pipeline = UnifiedSearchPipeline(
            image_model_name="openai/clip-vit-large-patch14-336",
            image_hf_dataset_id="crag-mm-2025/image-search-index-validation",
            text_model_name="BAAI/bge-large-en-v1.5",
            web_hf_dataset_id="crag-mm-2025/web-search-index-validation",
        )
        self.top_k = top_k
        
    def clean_wiki_text(self, text):
        """清理 MediaWiki 格式，转换为自然语言"""
        if not isinstance(text, str):
            return str(text)
        
        text = re.sub(r'<br\s*/?>', ', ', text)
        
        text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
        
        def convert_repl(match):
            parts = match.group(1).split('|')
            valid_parts = [p for p in parts if p and '=' not in p]
            return " ".join(valid_parts[:2])
        text = re.sub(r'\{\{convert\|(.*?)\}\}', convert_repl, text)
    
        text = re.sub(r'\{\{.*?\}\}', '', text)
        
        return text.strip()
    
    def process_graph_results(self, results):
        results_list = []
        seen_entities = set()
        
        exclude_keys = {
            'image', 'image_size', 'coordinates', 'mapframe_wikidata', 
            'website', 'url', 'score', 'index'
        }
    
        if results:
            for item in results:
                entities = item.get('entities', [])
                for i, entity in enumerate(entities):
                    entity_name = self.clean_wiki_text(entity.get('entity_name', 'Unknown'))
                    if entity_name in seen_entities:
                        continue
                    seen_entities.add(entity_name)
                    
                    attr_lines = []
                    attributes = entity.get('entity_attributes', {})
                    if attributes is not None:
                        for key, value in attributes.items():
                            if key not in exclude_keys:
                                clean_val = self.clean_wiki_text(value)
                                if clean_val:
                                    readable_key = key.replace('_', ' ').capitalize()
                                    attr_lines.append(f"- {readable_key}: {clean_val}")
                        
                    if attr_lines:
                        entity_block = f"### Entity: {entity_name}\n" + "\n".join(attr_lines)
                        results_list.append(entity_block)
        return results_list 
        
    def search_by_image(self, image):
        '''
        {'index': 17030, 'score': 0.906402587890625, 'url': 'https://upload.wikimedia.org/wikipedia/commons/3/34/The_Beekman_tower_from_the_East_River_%286215420371%29.jpg', 'entities': [{'entity_name': '8 Spruce Street', 'entity_attributes': {'name': '8 Spruce Street<br />(New York by Gehry)', 'image': '8 Spruce Street (01030p).jpg', 'image_size': '200px', 'address': '8 Spruce Street<br />[[Manhattan]], New York, U.S. 10038', 'mapframe_wikidata': 'yes', 'coordinates': '{{coord|40|42|39|N|74|00|20|W|region:US-NY_type:landmark|display|=|inline,title}}', 'status': 'Complete', 'start_date': '2006', 'completion_date': '2010', 'opening': 'February 2011', 'building_type': '[[Mixed-use development|Mixed-use]]', 'architectural_style': '[[Deconstructivism]]', 'roof': '{{convert|870|ft|m|0|abbr|=|on}}', 'top_floor': '{{convert|827|ft|abbr|=|on}}', 'floor_count': '76', 'floor_area': '{{convert|1000000|sqft|m2|abbr|=|on}}', 'architect': '[[Frank Gehry]]', 'structural_engineer': '[[WSP Group|WSP Cantor Seinuk]]', 'main_contractor': 'Kreisler Borg Florman', 'developer': '[[Forest City Ratner]]', 'engineer': '[[Jaros, Baum & Bolles]] (MEP)', 'owner': '8 Spruce (NY) Owner LLC', 'management': 'Beam Living', 'website': '{{URL|https://live8spruce.com/}}'}}]}

        {'index': 16196, 'score': 0.8643585443496704, 'url': 'https://upload.wikimedia.org/wikipedia/commons/1/12/New_York_by_Gehry_-_New_York_-_USA_-_panoramio.jpg', 'entities': [{'entity_name': '8 Spruce Street', 'entity_attributes': {'name': '8 Spruce Street<br />(New York by Gehry)', 'image': '8 Spruce Street (01030p).jpg', 'image_size': '200px', 'address': '8 Spruce Street<br />[[Manhattan]], New York, U.S. 10038', 'mapframe_wikidata': 'yes', 'coordinates': '{{coord|40|42|39|N|74|00|20|W|region:US-NY_type:landmark|display|=|inline,title}}', 'status': 'Complete', 'start_date': '2006', 'completion_date': '2010', 'opening': 'February 2011', 'building_type': '[[Mixed-use development|Mixed-use]]', 'architectural_style': '[[Deconstructivism]]', 'roof': '{{convert|870|ft|m|0|abbr|=|on}}', 'top_floor': '{{convert|827|ft|abbr|=|on}}', 'floor_count': '76', 'floor_area': '{{convert|1000000|sqft|m2|abbr|=|on}}', 'architect': '[[Frank Gehry]]', 'structural_engineer': '[[WSP Group|WSP Cantor Seinuk]]', 'main_contractor': 'Kreisler Borg Florman', 'developer': '[[Forest City Ratner]]', 'engineer': '[[Jaros, Baum & Bolles]] (MEP)', 'owner': '8 Spruce (NY) Owner LLC', 'management': 'Beam Living', 'website': '{{URL|https://live8spruce.com/}}'}}]}

        '''
        results = self.search_pipeline(image, k=self.top_k)
        formatted_texts = self.process_graph_results(results)
        return formatted_texts

    def search_by_text(self, text):
        results = self.search_pipeline(text, k=2)
        if results is None:
            return ["No results founds"]
        else:
            results_list = []
            for result in results:
                results_list.append(f"{result.get('page_name', '')}\n{result.get('page_snippet', '')}")
            return results_list


        





    