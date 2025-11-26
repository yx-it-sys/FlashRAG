from transformers import AutoTokenizer, AutoModelForCausalLM
import tomllib
from flashrag.utils import get_retriever
from typing import List
from utils import extract_json_for_assessment, chat_with_qwen

class Pipeline():
    def __init__(self, config, model, tokenizer, entity_extractor, device, max_loops, ret_thresh, retriever=None):
        self.device = device
        self.entity_extractor = entity_extractor
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
        current_query = [question]
        loop_count = 0
        collected_useful_fragments = []
        records = []
        while loop_count < self.max_loops:
            retrieved_docs, scores = self.retriever.batch_search(query=current_query, num=self.top_k, return_score=True)
            print(f"retrieved_docs: {retrieved_docs}")
            print(f"scores: {scores}")
            retrieved_results = []

            for doc, score in zip(retrieved_docs, scores):
                if score >= self.ret_thresh:
                    retrieved_results.append({'doc': doc['contents'], 'score': score})
            
            # print(f"Retrieved Results: {retrieved_results}")
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
        
        # 循环次数太多，考虑Replan
        print("Loop in Retrieval-Assess-Refine.")
        for record in records:
            print(f"state: {record['state']}")
            print(f"result: {record['result']}")
        # supervised_answer = self.internal_debate_generate(question, collected_useful_fragments)
        final_answer = self.rag_generate(question, list(dict.fromkeys(collected_useful_fragments)))
        print(f"Mocked Debate Answer: {final_answer}")
        records.append({"state": "internal_generate", "result": final_answer})
        log = {'sub_question': question, "records": records}
        return final_answer, log
    
    def assess(self, query: str, docs: List[str]):
            messages = [                
                {"role": "system", "content": self.assessment_prompt['system_prompt']},
                {"role": "user", "content": self.assessment_prompt['user_prompt'].format(user_query=query, documents_list=docs)}
            ]
            response = chat_with_qwen(self.model, self.tokenizer, messages, "qwen2", enable_thinking=False)
            # print(f"Assess response: {response}")
            assessment_result = extract_json_for_assessment(response)
            return assessment_result

    def refine(self, missing_information: str):
            # GLiNER提取missing_information的实体，得到实体列表
            labels = ["person", "award", "date", "competitions", "teams"]
            entity_list = self.entity_extractor.predict_entities(missing_information, labels)
            # 将current_query与实体列表中的元素一一配对，新的查询列表
            query_list = [f"Find '{missing_information}' related to {entity}" for entity in entity_list]
            # 返回新的查询列表
            return query_list

    def rag_generate(self, question: str, supporting_docs: List[str]):
            messages = [                
                {"role": "system", "content": self.rag_prompt['system_prompt']},
                {"role": "user", "content": self.rag_prompt['user_prompt'].format(reference=supporting_docs, question=question)}
            ]
            response = chat_with_qwen(self.model, self.tokenizer, messages, "qwen2", enable_thinking=False)
            return response
    
    def internal_debate_generate(self, question: str, supporting_docs: List[str]):
        pass

            


            
        
        