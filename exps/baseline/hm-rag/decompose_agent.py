from typing import List
from utils import qwen_generate
from transformers import AutoModelForCausalLM, AutoTokenizer

class DecomposeAgent:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


    def count_intents(self, query: str) -> int:
        """
        Determine the number of intents in the input query.
        Use LLM to analyze the number of intents contained in the input text.
        Args:
            query (str): The input query text.
        Returns:
            int: The number of intents.
        """
        sys_prompt = "Please calculate how many independent intents are contained in the following query. Return only an integer."
        user_prompt = f"{query}\nNumber of intents: "
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = qwen_generate(messages, self.model, self.tokenizer)
        try:
            return int(response.strip())
        except ValueError:
            return 1
        
    def decompose(self, query: str) -> List[str]:
        """
        Decompose the query. If the number of intents is greater than 1, perform intent decomposition.
        Args:
            query (str): The input query text.
        Returns:
            List[str]: A list of decomposed sub-queries.
        """
        intent_count = self.count_intents(query)
        intent_count = min(intent_count, 3)  # Limit the number of intents to a maximum of 5
        if intent_count > 1:
            return self._split_query(query)
        # return [query]
        return query

    def _split_query(self, query: str) -> List[str]:
        """
        The method that actually performs query decomposition.
        Args:
            query (str): The input query text.
        Returns:
            List[str]: A list of decomposed sub-queries.
        """
        sys_prompt = "Split the following query into multiple independent sub-queries, separated by '||', without additional explanations."
        user_prompt = f":\n{query}\nList of sub-queries:"

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = qwen_generate(messages, self.model, self.tokenizer)

        return [q.strip() for q in response.split("||") if q.strip()]



# ---------------------- Run tests ----------------------
if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)    
    agent = DecomposeAgent(model, tokenizer)
    query = "Check today's weather in Shanghai, then summarize the latest scientific research news from Fudan University, and finally compare the advantages and disadvantages of Python and Java."
    subqueries = agent.decompose(query)
    print("Decomposed sub-queries:")
    for i, subq in enumerate(subqueries, 1):
        print(f"{i}. {subq}")