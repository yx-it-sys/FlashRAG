from transformers import AutoTokenizer, AutoModelForCausalLM
import tomllib
from flashrag.utils import get_retriever
from typing import List
from utils import extract_json_for_assessment, chat_with_qwen

class Pipeline():
    def __init__(self, config, model, tokenizer, device, max_loops, ret_thresh, retriever=None):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.max_loops = max_loops
        self.top_k = self.config['retrieval_topk']
        self.ret_thresh = ret_thresh
        with open('prompts/rectify.toml', "rb") as f:
             self.rectify_prompt = tomllib.load(f)
        with open('prompts/assess.toml', "rb") as f:
            self.assessment_prompt = tomllib.load(f)
        with open('prompts/refine_misinformation.toml', 'rb') as f:
            self.refine_prompt = tomllib.load(f)
        with open('prompts/rag_generate.toml', 'rb') as f:
             self.rag_prompt = tomllib.load(f)
        with open('prompts/internal_debate_generate.toml', 'rb') as f:
             self.debate_prompt = tomllib.load(f)
        if retriever is None:
            retriever = get_retriever(config)
        self.retriever = retriever

    def run_with_question_only(self, question: str):
        # current_query = self.rectify(question, context)
        current_query = question
        loop_count = 0
        collected_useful_fragments = []
        records = []
        while loop_count < self.max_loops:
            retrieved_docs, scores = self.retriever.search(query=current_query, num=self.top_k, return_score=True)
            retrieved_results = []

            for doc, score in zip(retrieved_docs, scores):
                if score >= self.ret_thresh:
                    retrieved_results.append({'doc': doc['contents'], 'score': score})
            
            # print(f"Retrieved Results: {retrieved_results}")
            records.append({"state": "retrieve", "result": [doc['doc'] for doc in retrieved_results]})
            if len(collected_useful_fragments) == 0:
                assessment_result = self.assess(question, [doc['doc'] for doc in retrieved_results])
            else:
                assessment_result = self.assess(question, collected_useful_fragments)
            # print(f"Assessment Result: {assessment_result}")
            # 保存assessment result
            records.append({"state": "assess", "result": assessment_result})
            # 当LLM生成失败，则跳过assess的步骤，直接return，转到下一个数据点
            if assessment_result is None:
                print("Error in LLM Generating, skipping to the next question")
                records.append({"state": "assess", "result": "Error in LLM Generating, skipping to the next question"})
                return None, assessment_result
            
            assessment = assessment_result.get('judgment', '')
            useful_fragments = assessment_result.get('useful_fragments', '')
            missing_information = assessment_result.get('missing_information', '')

            collected_useful_fragments.extend(useful_fragments)
            print(f"collected_useful_fragments: {collected_useful_fragments}")
            if assessment == "sufficient":
                final_answer = self.rag_generate(question, list(dict.fromkeys(collected_useful_fragments)))
                print(f"Sufficient case, RAG answer: {final_answer}")
                records.append({"state": "rag_generate", "result": final_answer})
                log = {'sub_question': question, "records": records}
                return final_answer, log

            elif assessment == "insufficient":
                if loop_count > self.max_loops:
                    break
                loop_count += 1
                current_query = self.refine(current_query, missing_information)
                print(f"refined Query: {current_query}")
                records.append({"state": "refine", "result": current_query})
        
        # supervised_answer = self.internal_debate_generate(question, collected_useful_fragments)
        final_answer = self.rag_generate(question, list(dict.fromkeys(collected_useful_fragments)))
        print(f"Mocked Debate Answer: {final_answer}")
        records.append({"state": "internal_generate", "result": final_answer})
        log = {'sub_question': question, "records": records}
        return final_answer, log
    
    def rectify(self, question: str, context: List[str]):
        messages = [                
            {"role": "system", "content": self.rectify_prompt['system_prompt']},
            {"role": "user", "content": self.rectify_prompt['user_prompt'].format(user_query=question, context=context)}
        ]

        response = chat_with_qwen(self.model, self.tokenizer, messages, "qwen3", "thinking")['content']
        print(f"Rectify response: {response}")
        return response

    def assess(self, query: str, docs: List[str]):
            messages = [                
                {"role": "system", "content": self.assessment_prompt['system_prompt']},
                {"role": "user", "content": self.assessment_prompt['user_prompt'].format(user_query=query, documents_list=docs)}
            ]
            response = chat_with_qwen(self.model, self.tokenizer, messages, "qwen3", "thinking")['content']
            # print(f"Assess response: {response}")
            assessment_result = extract_json_for_assessment(response)
            return assessment_result

    def refine(self, current_query: str,  missing_information: str):
            messages = [                
                {"role": "system", "content": self.refine_prompt['system_prompt']},
                {"role": "user", "content": self.refine_prompt['user_prompt'].format(current_query=current_query, missing_info_from_assess=missing_information)}
            ]
            response = chat_with_qwen(self.model, self.tokenizer, messages, "qwen3", "thinking")['content']
            return response

    def rag_generate(self, question: str, supporting_docs: List[str]):
            messages = [                
                {"role": "system", "content": self.rag_prompt['system_prompt']},
                {"role": "user", "content": self.rag_prompt['user_prompt'].format(reference=supporting_docs, question=question)}
            ]
            response = chat_with_qwen(self.model, self.tokenizer, messages, "qwen3", "thinking")['content']
            return response
    
    def internal_debate_generate(self, question: str, supporting_docs: List[str]):
        pass

            


            
        
        