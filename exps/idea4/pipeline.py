import tomllib
from flashrag.utils import get_retriever
from typing import List
import re
from utils import extract_refine, chat_with_qwen

class Pipeline():
    def __init__(self, config, model, tokenizer, device, max_loops, ret_thresh, retriever=None):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.max_loops = max_loops
        self.top_k = self.config['retrieval_topk']
        self.ret_thresh = ret_thresh
        with open('prompts/refine.toml', 'rb') as f:
            self.refine_prompt = tomllib.load(f)
        with open('prompts/rag_generate.toml', 'rb') as f:
             self.rag_prompt = tomllib.load(f)
        if retriever is None:
            retriever = get_retriever(config)
        self.retriever = retriever
    def run_rag(self, question: str):
        query_list = [question]
        retrieved_docs, scores = self.retriever.batch_search(query=query_list, num=self.top_k, return_score=True)
        retrieved_results = []
        for docs, scores in zip(retrieved_docs, scores):
            for doc, score in zip(docs, scores):
                if score >= self.ret_thresh:
                    retrieved_results.append(doc['contents'])
        references = "\n".join(retrieved_results)
        messages = [                
            {"role": "system", "content": self.rag_prompt['system_prompt']},
            {"role": "user", "content": self.rag_prompt['user_prompt'].format(question=question, reference=references)}
        ]
        response = chat_with_qwen(self.model, self.tokenizer, messages, "qwen2", enable_thinking=False)
        llm_output = response['content']
        # print(f"Assess response: {response}")
        # parse outputs in xml format
        patterns = {
            "think": r"<think>(.*?)</think>",
            "missing": r"<missing>(.*?)</missing>",
            "answer": r"<answer>(.*?)</answer>"
        }

        parts = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, llm_output, re.DOTALL)
            if match:
                content = match.group(1).strip()
                parts[key] = content if content else None
            else:
                print(f"ERROR! fail to parse assess&generate.")
                parts[key] = None
        is_sufficient = True if parts["missing"] is None else False
        if not is_sufficient:
            # Retriever Pro
            
    
    
    def run_with_question_only(self, question: str):
        query_list = [question]
        loop_count = 0
        collected_useful_fragments = []
        records = []
        while loop_count < self.max_loops:
            #print(f"Current Query: {query_list}")
            if query_list is None or len(query_list) == 0:
                print("No more queries to process. Exiting loop.")
                break
            retrieved_docs, scores = self.retriever.batch_search(query=query_list, num=self.top_k, return_score=True)
            retrieved_results = []

            for docs, scores in zip(retrieved_docs, scores):
                for doc, score in zip(docs, scores):
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
            #print(f"collected_useful_fragments: {collected_useful_fragments}")
            if assessment == "sufficient":
                final_answer = self.rag_generate(question, list(dict.fromkeys(collected_useful_fragments)))
                #print(f"Sufficient case, RAG answer: {final_answer}")
                records.append({"state": "rag_generate", "result": final_answer})
                log = {'sub_question': question, "records": records}
                return final_answer, log

            elif assessment == "insufficient":
                if loop_count > self.max_loops:
                    break
                loop_count += 1
                query_list = self.refine(query_list[-1], missing_information)
                #print(f"refined Query: {query_list}")
                records.append({"state": "refine", "result": query_list})
        
        if loop_count >= self.max_loops:
            print("Loop in Retrieval-Assess-Refine.")
            for record in records:
                print(f"state: {record['state']}")
                print(f"result: {record['result']}")
            # supervised_answer = self.internal_debate_generate(question, collected_useful_fragments)
            # final_answer = self.rag_generate(question, list(dict.fromkeys(collected_useful_fragments)))
            # print(f"Mocked Debate Answer: {final_answer}")
            final_answer = None
            # records.append({"state": "give_up", "result": final_answer})
        log = {'sub_question': question, "records": records}
        return final_answer, log
    
    def assess_generate(self, query: str, docs: List[str]):
            messages = [                
                {"role": "system", "content": self.rag_prompt['system_prompt']},
                {"role": "user", "content": self.rag_prompt['user_prompt'].format(question=query, reference=docs)}
            ]
            response = chat_with_qwen(self.model, self.tokenizer, messages, "qwen2", enable_thinking=False)
            llm_output = response['content']
            # print(f"Assess response: {response}")
            # parse outputs in xml format
            patterns = {
                "think": r"<think>(.*?)</think>",
                "missing": r"<missing>(.*?)</missing>",
                "answer": r"<answer>(.*?)</answer>"
            }

            parts = {}
            for key, pattern in patterns.items():
                match = re.search(pattern, llm_output, re.DOTALL)
                if match:
                    content = match.group(1).strip()
                    parts[key] = content if content else None
                else:
                    print(f"ERROR! fail to parse assess&generate.")
                    parts[key] = None
            is_sufficient = True if parts["missing"] is None else False
            if not is_sufficient:
                # Retriever Pro

            return assessment_result

    def refine(self, current_query, missing_information: str):
        messages = [
                {"role": "system", "content": self.refine_prompt['system_prompt']},
                {"role": "user", "content": self.refine_prompt['user_prompt'].format(last_attempted_query=current_query, missing_info_from_assess=missing_information)}
            ]
        response = chat_with_qwen(self.model, self.tokenizer, messages, "qwen2", enable_thinking=False)['content']
        entities, refined_query = extract_refine(response)
        entities.append(refined_query)
        return entities
        

    def rag_generate(self, question: str, supporting_docs: List[str]):
            messages = [                
                {"role": "system", "content": self.rag_prompt['system_prompt']},
                {"role": "user", "content": self.rag_prompt['user_prompt'].format(reference=supporting_docs, question=question)}
            ]
            response = chat_with_qwen(self.model, self.tokenizer, messages, "qwen2", enable_thinking=False)
            return response['content']
    
    def internal_debate_generate(self, question: str, supporting_docs: List[str]):
        pass

            


            
        
        