from transformers import AutoTokenizer, AutoModelForCausalLM
import tomllib
from flashrag.utils import get_retriever
from typing import List
from utils import extract_json_for_assessment

class Pipeline():
    def __init__(self, config, model_name, max_loops, ret_thresh):
        self.model_name = model_name
        self.config = config
        self.max_loops = max_loops
        self.top_k = self.config['retrieval_topk']
        self.ret_thresh = ret_thresh
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        with open('prompts/assess.toml', "rb") as f:
            self.assessment_prompt = tomllib.load(f)
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
            
            assessment_result = self.assess(current_query, [doc['doc'] for doc in retrieved_results])
            
            assessment = assessment_result.get('judgment', '')
            useful_fragments = assessment_result.get('useful_fragments', '')
            missing_information = assessment_result.get('missing_information', '')

            collected_useful_fragments.extend(useful_fragments)

            if assessment == "sufficient":
                final_answer = self.rag_generate(question, collected_useful_fragments)
                return final_answer

            elif assessment == "insufficient":
                current_query = self.refine(current_query, missing_information)
                loop_count += 1
        
        internal_answer = self.internal_generate(question, list({v['id']: v for v in collected_useful_fragments}.values()))
        supervised_answer = self.debate_generate(internal_answer, collected_useful_fragments)
        
        return supervised_answer
    
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

            outputs = self.model.generate(**inputs, max_new_tokens=40)
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
            assessment_result = extract_json_for_assessment(response)
            return assessment_result

    def refine(self, query: str, missing_information: str):
        pass

    def rag_generate(self, question: str, supporting_docs: List[str]):
        pass
    
    def internal_generate(self, question: str, supporting_docs: List[str]):
        pass

    def debate_generate(self, question: str, supporting_docs: List[str]):
        pass


            


            
        
        