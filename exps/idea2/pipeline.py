from transformers import AutoTokenizer, AutoModelForCausalLM
import tomllib
from flashrag.utils import get_retriever
from typing import List
from utils import extract_json_for_assessment

class Pipeline():
    def __init__(self, config, model_name, max_loops, ret_thresh, retriever=None):
        self.model_name = model_name
        self.config = config
        self.max_loops = max_loops
        self.top_k = self.config['retrieval_topk']
        self.ret_thresh = ret_thresh
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        with open('prompts/assess.toml', "rb") as f:
            self.assessment_prompt = tomllib.load(f)
        with open('prompts/refine.toml', 'rb') as f:
            self.refine_prompt = tomllib.load(f)
        with open('prompts/rag_generate.toml', 'rb') as f:
             self.rag_prompt = tomllib.load(f)
        with open('prompts/internal_debate_generate.toml', 'rb') as f:
             self.debate_prompt = tomllib.load(f)
        if retriever is None:
            retriever = get_retriever(config)
        self.retriever = retriever

    def run_with_question_only(self, question):
        current_query = question
        loop_count = 0
        collected_useful_fragments = []

        while loop_count < self.max_loops:
            retrieved_docs, scores = self.retriever.search(query=current_query, num=self.top_k, return_score=True)
            retrieved_results = []

            for doc, score in zip(retrieved_docs, scores):
                if score >= self.ret_thresh:
                    retrieved_results.append({'doc': doc, 'score': score})
            
            print(f"Retrieved Results: {retrieved_results}")
            
            assessment_result = self.assess(current_query, [doc['doc'] for doc in retrieved_results])
            
            print(f"Assessment Result: {assessment_result}")

            assessment = assessment_result.get('judgment', '')
            useful_fragments = assessment_result.get('useful_fragments', '')
            missing_information = assessment_result.get('missing_information', '')

            collected_useful_fragments.extend(useful_fragments)
            print(f"collected_useful_fragments: {collected_useful_fragments}")
            if assessment == "sufficient":
                final_answer = self.rag_generate(question, list(dict.fromkeys(collected_useful_fragments)))
                print(f"Sufficient case, RAG anser: {final_answer}")
                return final_answer

            elif assessment == "insufficient":
                if loop_count > self.max_loops:
                    break
                loop_count += 1
                current_query = self.refine(question, current_query, collected_useful_fragments, missing_information)
                print(f"refined Query: {current_query}")
        
        # supervised_answer = self.internal_debate_generate(question, list({v['id']: v for v in collected_useful_fragments}.values()))
        final_answer = self.rag_generate(question, list(dict.fromkeys(collected_useful_fragments)))
        print(f"Mocked Debate Answer: {final_answer}")
        return final_answer, None
    
    def assess(self, query: str, docs: List[str]):
            messages = [                
                {"role": "system", "content": self.assessment_prompt['system_prompt']},
                {"role": "user", "content": self.assessment_prompt['user_prompt'].format(user_query=query, documents_list=docs)}
            ]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            outputs = self.model.generate(**inputs, max_new_tokens=2048)
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            print(f"Assess response: {response}")
            assessment_result = extract_json_for_assessment(response)
            return assessment_result

    def refine(self, initial_query: str, current_query: str, collected_useful_fragments: List[str], missing_information: str):
            messages = [                
                {"role": "system", "content": self.refine_prompt['system_prompt']},
                {"role": "user", "content": self.refine_prompt['user_prompt'].format(original_user_query=initial_query, last_attempted_query=current_query, collected_useful_fragments=collected_useful_fragments, missing_info_from_assess =missing_information)}
            ]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            outputs = self.model.generate(**inputs, max_new_tokens=2048)
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            return response

    def rag_generate(self, question: str, supporting_docs: List[str]):
            messages = [                
                {"role": "system", "content": self.rag_prompt['system_prompt']},
                {"role": "user", "content": self.rag_prompt['user_prompt'].format(reference=supporting_docs, question=question)}
            ]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            outputs = self.model.generate(**inputs, max_new_tokens=2048)
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            return response
    
    def internal_debate_generate(self, question: str, supporting_docs: List[str]):
        pass

            


            
        
        