import re
import json
import time
import numpy as np
import warnings
from collections import Counter
from flashrag.evaluator.utils import normalize_answer
from tqdm import tqdm

class BaseMetric:
    """`BaseMetric` serves as the base object of all metrics. Implemented metric should
    inherit this class.
    """

    metric_name = "base"

    def __init__(self, config):
        self.config = config
        self.dataset_name = config["dataset_name"]

    def calculate_metric(self, data):
        """Get the total score of this metric and score for each sample.

        Args:
            data object: it contains basic information and generated information.

        Returns:
            (metric_score: dict, metric_score_list: list)
            metric_score: such as ``{'em': 0.53}``.
            metric_score_list: score for each sample.

        """
        return {}, []

    def _canonicalize_answer(self, answer):
        if answer is None:
            return ""
        if isinstance(answer, dict):
            if "wikidata" in answer:
                return str(answer["wikidata"])
            if "text" in answer:
                return str(answer["text"])
            if "value" in answer:
                return str(answer["value"])
            return json.dumps(answer, ensure_ascii=False)
        return str(answer)

    def _canonicalize_answer_list(self, answers):
        if isinstance(answers, list):
            return [self._canonicalize_answer(answer) for answer in answers]
        return [self._canonicalize_answer(answers)]

    def get_dataset_answer(self, data):
        if any(choice == [] for choice in data.choices):
            golden_answers_list = data.answer_eval if hasattr(data, "answer_eval") else data.answer
            if any(not answers for answers in golden_answers_list):
                answer_eval_list = getattr(data, "answer_eval", None)
                if answer_eval_list is not None:
                    golden_answers_list = [
                        answers if answers else (answer_eval if isinstance(answer_eval, list) else [answer_eval])
                        for answers, answer_eval in zip(golden_answers_list, answer_eval_list)
                    ]
                if any(not answers for answers in golden_answers_list):
                    answer_list = getattr(data, "answer", None)
                    if answer_list is not None:
                        golden_answers_list = [
                            answers if answers else (answer if isinstance(answer, list) else [answer])
                            for answers, answer in zip(golden_answers_list, answer_list)
                        ]
        else:
            # multi-choice dataset
            all_choices_list = data.choices
            golden_choice_idx_list = data.golden_answers
            golden_answers_list = [
                [choices[idx] for idx in idx_list]
                for choices, idx_list in zip(all_choices_list, golden_choice_idx_list)
            ]

        golden_answers_list = [self._canonicalize_answer_list(answers) for answers in golden_answers_list]

        return golden_answers_list


class F1_Score(BaseMetric):
    """Token-level F1 score"""

    metric_name = "f1"

    def __init__(self, config):
        super().__init__(config)

    def token_level_scores(self, prediction: str, ground_truths: list):
        final_metric = {"f1": 0, "precision": 0, "recall": 0}
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        for ground_truth in ground_truths:
            normalized_prediction = normalize_answer(prediction)
            normalized_ground_truth = normalize_answer(ground_truth)

            if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
                continue
            if (
                normalized_ground_truth in ["yes", "no", "noanswer"]
                and normalized_prediction != normalized_ground_truth
            ):
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ["f1", "precision", "recall"]:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric

    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = self.get_dataset_answer(data)

        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["f1"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        f1 = sum(metric_score_list) / len(metric_score_list)
        return {"f1": f1}, metric_score_list


class Recall_Score(F1_Score):
    """Token-level Recall score"""

    metric_name = "recall"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = self.get_dataset_answer(data)
        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["recall"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        precision = sum(metric_score_list) / len(metric_score_list)
        return {"recall": precision}, metric_score_list


class Precision_Score(F1_Score):
    """Token-level Precision score"""

    metric_name = "precision"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = self.get_dataset_answer(data)
        metric_score_list = [
            self.token_level_scores(pred, golden_answers)["precision"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        precision = sum(metric_score_list) / len(metric_score_list)
        return {"precision": precision}, metric_score_list

class ExactMatch(BaseMetric):
    r"""Exact match measure whether the predicted answer is completely consistent
    with the standard answer.

    """

    metric_name = "em"

    def __init__(self, config):
        super().__init__(config)
        self.is_regex = self.dataset_name == "curatedtrec"

    def calculate_em(self, prediction: str, golden_answers: list) -> float:
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_answer(prediction)
        score = 0.0
        for golden_answer in golden_answers:
            if self.is_regex:
                print("Consider answer as regex!")
                golden_answer = re.compile(golden_answer, re.IGNORECASE)
                match = re.fullmatch(golden_answer, normalized_prediction)
                if match is not None:
                    score = 1.0
                    break
            else:
                golden_answer = normalize_answer(golden_answer)
                if golden_answer == normalized_prediction:
                    score = 1.0
                    break
        return score

    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = self.get_dataset_answer(data)

        metric_score_list = [
            self.calculate_em(pred, golden_answers) for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        em_score = sum(metric_score_list) / len(metric_score_list)

        return {"em": em_score}, metric_score_list


class Sub_ExactMatch(BaseMetric):
    r"""Sub-Exact match measure whether the predicted answer contains the standard answer."""

    metric_name = "acc"

    def __init__(self, config):
        super().__init__(config)
        self.is_regex = self.dataset_name == "curatedtrec"

    def calculate_sub_em(self, prediction: str, golden_answers: list) -> float:
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_answer(prediction)
        score = 0.0
        for golden_answer in golden_answers:
            if self.is_regex:
                print("Consider answer as regex!")
                golden_answer = re.compile(golden_answer, re.IGNORECASE)
                match = re.search(golden_answer, normalized_prediction)
                if match is not None:
                    score = 1.0
                    break
            else:
                golden_answer = normalize_answer(golden_answer)
                if golden_answer in normalized_prediction:
                    score = 1.0
                    break
        return score

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        pred_list = data.pred

        metric_score_list = [
            self.calculate_sub_em(pred, golden_answers) for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        sub_em_score = sum(metric_score_list) / len(metric_score_list)

        return {"acc": sub_em_score}, metric_score_list

class GPTAcc(BaseMetric):
    metric_name = "gpt"

    def __init__(self, config):
        super().__init__(config)
        from openai import OpenAI

        self.sys_prompt = (
            "You are a strict answer judge.\n"
            "Task: compare the Candidate's Answer against the Reference Answers and decide whether the candidate answer is semantically correct.\n"
            "Rules:\n"
            "1. Return 'yes' if the candidate answer matches the reference meaning, even if wording differs.\n"
            "2. Return 'no' if the answer is wrong, unsupported, incomplete for the asked target, or irrelevant.\n"
            "3. If the candidate answer is empty, a reasoning trace, a search plan, or does not directly answer the question, return 'no'.\n"
            "4. Output exactly one token: yes or no.\n"
            "5. Do not output any explanation, punctuation, or extra words."
        )
        gpt_acc_setting = config["gpt_acc_setting"]
        self.api_model = gpt_acc_setting["model_name"]
        api_key = gpt_acc_setting["api_key"]
        base_url = gpt_acc_setting["base_url"]
        self.max_retries = 5
        self.retry_sleep_seconds = 5

        assert api_key, "GPTAcc requires gpt_acc_setting.api_key for API judge."
        assert base_url, "GPTAcc requires gpt_acc_setting.base_url for API judge."

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def _parse_judge_text(self, judge_text: str):
        judge_text = (judge_text or "").strip().lower()
        if judge_text == "yes":
            return 1.0
        if judge_text == "no":
            return 0.0

        tokens = re.findall(r"\b(?:yes|no)\b", judge_text)
        if len(tokens) == 1:
            return 1.0 if tokens[0] == "yes" else 0.0
        return None

    def calculate_acc(self, prediction: str, golden_answers: list) -> float:
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_answer(prediction)
        reference_answers_list = "\n".join(golden_answers)
        messages = [
            {"role": "system", "content": self.sys_prompt},
            {
                "role": "user",
                "content": (
                    f"Reference Answers:\n{reference_answers_list}\n\n"
                    f"Candidate's Answer:{normalized_prediction}\n\n"
                    "Your judge response:"
                ),
            },
        ]
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.api_model,
                    messages=messages,
                    stream=False,
                    temperature=0,
                    max_tokens=4,
                )
                judge_text = (response.choices[0].message.content or "").strip()
                parsed_score = self._parse_judge_text(judge_text)
                if parsed_score is not None:
                    return parsed_score

                last_error = ValueError(f"Response cannot be parsed: {judge_text}")
                print(
                    f"ERROR judge with GPT API! Response cannot be parsed "
                    f"(attempt {attempt}/{self.max_retries}): {judge_text}"
                )
            except Exception as exc:
                last_error = exc
                print(
                    f"ERROR judge with GPT API! Request failed "
                    f"(attempt {attempt}/{self.max_retries}): {type(exc).__name__}: {exc}"
                )

            if attempt < self.max_retries:
                time.sleep(self.retry_sleep_seconds * attempt)

        raise last_error
    
    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        pred_list = data.pred

        metric_score_list = [
            self.calculate_acc(pred, golden_answers) for pred, golden_answers in tqdm(zip(pred_list, golden_answers_list), total=len(pred_list), desc="Calculating GPTAcc")
        ]
        gpt_acc_score = sum(metric_score_list) / len(metric_score_list)

        return {"gpt_acc": gpt_acc_score}, metric_score_list



class Retrieval_Recall(BaseMetric):
    r"""The recall of the top-k retreived passages, we measure if any of the passage contain the answer string."""

    metric_name = "retrieval_recall"

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["metric_setting"]["retrieval_recall_topk"]

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        retrieve_docs = data.retrieval_result
        recall_score_list = []
        for doc_list, golden_answers in zip(retrieve_docs, golden_answers_list):
            if len(doc_list) < self.topk:
                warnings.warn(f"Length of retrieved docs is smaller than topk ({self.topk})")
            doc_list = [doc["contents"] for doc in doc_list[: self.topk]]
            hit_list = []
            for doc in doc_list:
                for golden_answer in golden_answers:
                    if normalize_answer(golden_answer) in normalize_answer(doc):
                        hit_list.append(True)
                        break
                else:
                    hit_list.append(False)
            score = 1 if any(hit_list) else 0
            recall_score_list.append(score)
        recall_score = sum(recall_score_list) / len(recall_score_list)

        return {f"retrieval_recall_top{self.topk}": recall_score}, recall_score_list


class Retrieval_Precision(BaseMetric):
    r"""The precision of the top-k retreived passages, we measure if any of the passage contain the answer string."""

    metric_name = "retrieval_precision"

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["metric_setting"]["retrieval_recall_topk"]

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        retrieve_docs = data.retrieval_result
        precision_score_list = []
        for doc_list, golden_answers in zip(retrieve_docs, golden_answers_list):
            if len(doc_list) < self.topk:
                warnings.warn(f"Length of retrieved docs is smaller than topk ({self.topk})")
            doc_list = [doc["contents"] for doc in doc_list[: self.topk]]
            hit_list = []
            for doc in doc_list:
                for golden_answer in golden_answers:
                    if normalize_answer(golden_answer) in normalize_answer(doc):
                        hit_list.append(True)
                        break
                else:
                    hit_list.append(False)
            score = sum(hit_list) / len(hit_list)
            precision_score_list.append(score)
        precision_score = sum(precision_score_list) / len(precision_score_list)

        return {f"retrieval_precision_top{self.topk}": precision_score}, precision_score_list


class Rouge_Score(BaseMetric):
    metric_name = "rouge_score"
    cached_scores = {}
    
    def __init__(self, config):
        super().__init__(config)
        from rouge import Rouge

        self.scorer = Rouge()

    def calculate_rouge(self, pred, golden_answers):
        if (pred, tuple(golden_answers)) in self.cached_scores:
            return self.cached_scores[(pred, tuple(golden_answers))]
        output = {}
        for answer in golden_answers:
            scores = self.scorer.get_scores(pred, answer)
            for key in ["rouge-1", "rouge-2", "rouge-l"]:
                if key not in output:
                    output[key] = []
                output[key].append(scores[0][key]["f"])
        for k, v in output.items():
            output[k] = max(v)

        self.cached_scores[(pred, tuple(golden_answers))] = output
        return output




class Rouge_1(Rouge_Score):
    metric_name = "rouge-1"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        pred_list = data.pred

        metric_score_list = [
            self.calculate_rouge(pred, golden_answers)["rouge-1"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"rouge-1": score}, metric_score_list


class Rouge_2(Rouge_Score):
    metric_name = "rouge-2"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        pred_list = data.pred

        metric_score_list = [
            self.calculate_rouge(pred, golden_answers)["rouge-2"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"rouge-2": score}, metric_score_list


class Rouge_L(Rouge_Score):
    metric_name = "rouge-l"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        pred_list = data.pred

        metric_score_list = [
            self.calculate_rouge(pred, golden_answers)["rouge-l"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"rouge-l": score}, metric_score_list



class ZH_Rouge_Score(BaseMetric):
    metric_name = "zh_rouge_score"
    cached_scores = {}
    
    def __init__(self, config):
        super().__init__(config)
        from rouge_chinese import Rouge

        self.scorer = Rouge()

    def calculate_rouge(self, pred, golden_answers):
        import jieba
        if (pred, tuple(golden_answers)) in self.cached_scores:
            return self.cached_scores[(pred, tuple(golden_answers))]
        output = {}
        pred = ' '.join(jieba.cut(pred))
        for answer in golden_answers:
            answer = ' '.join(jieba.cut(answer))
            scores = self.scorer.get_scores(pred, answer)
            for key in ["rouge-1", "rouge-2", "rouge-l"]:
                if key not in output:
                    output[key] = []
                output[key].append(scores[0][key]["f"])
        for k, v in output.items():
            output[k] = max(v)

        self.cached_scores[(pred, tuple(golden_answers))] = output
        return output




class ZH_Rouge_1(ZH_Rouge_Score):
    metric_name = "zh_rouge-1"

    def __init__(self, config):
        super().__init__(config)
        

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        pred_list = data.pred

        metric_score_list = [
            self.calculate_rouge(pred, golden_answers)["rouge-1"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"zh_rouge-1": score}, metric_score_list


class ZH_Rouge_2(ZH_Rouge_Score):
    metric_name = "zh_rouge-2"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        pred_list = data.pred

        metric_score_list = [
            self.calculate_rouge(pred, golden_answers)["rouge-2"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"zh_rouge-2": score}, metric_score_list


class ZH_Rouge_L(ZH_Rouge_Score):
    metric_name = "zh_rouge-l"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        golden_answers_list = self.get_dataset_answer(data)
        pred_list = data.pred

        metric_score_list = [
            self.calculate_rouge(pred, golden_answers)["rouge-l"]
            for pred, golden_answers in zip(pred_list, golden_answers_list)
        ]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"zh_rouge-l": score}, metric_score_list




class BLEU(BaseMetric):
    metric_name = "bleu"

    def __init__(self, config):
        super().__init__(config)
        from ._bleu import Tokenizer13a

        self.tokenizer = Tokenizer13a()
        self.max_order = config["metric_setting"].get("bleu_max_order", 4)
        self.smooth = config["metric_setting"].get("bleu_smooth", False)

    def calculate_metric(self, data):
        from ._bleu import compute_bleu

        golden_answers_list = self.get_dataset_answer(data)
        pred_list = data.pred

        pred_list = [self.tokenizer(pred) for pred in pred_list]
        golden_answers_list = [
            [self.tokenizer(ans) for ans in golden_answers] for golden_answers in golden_answers_list
        ]
        score = compute_bleu(
            reference_corpus=golden_answers_list,
            translation_corpus=pred_list,
            max_order=self.max_order,
            smooth=self.smooth,
        )
        (total_bleu, precisions, bp, ratio, translation_length, reference_length) = score

        score_list = []
        for pred, golden_answers in zip(pred_list, golden_answers_list):
            pred = [pred]
            golden_answers = [golden_answers]
            score = compute_bleu(
                reference_corpus=golden_answers,
                translation_corpus=pred,
                max_order=self.max_order,
                smooth=self.smooth,
            )
            (bleu, precisions, bp, ratio, translation_length, reference_length) = score
            score_list.append(bleu)

        return {"bleu": total_bleu}, score_list


class LLMJudge(BaseMetric):
    metric_name = "llm_judge"
    JUDGE_PROMPT = """
    You will be given a user_question and system_answer couple.
    Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
    Give your answer as a float on a scale of 0 to 10, where 0 means that the system_answer is not helpful at all, and 10 means that the answer completely and helpfully addresses the question.

    Provide your feedback as follows:

    Feedback:::
    Total rating: (your rating, as a float between 0 and 10)

    Now here are the question and answer.

    Question: {question}
    Answer: {answer}

    Feedback:::
    Total rating: """

    def __init__(self, config):
        super().__init__(config)
        if "llm_judge_setting" in config["metric_setting"]:
            llm_setting = config["metric_setting"]["llm_judge_setting"]
        else:
            assert False, "No available LLM settings!"
        # TODO: integrate generator class
        llm_name = llm_setting["model_name"]
        if "model_path" not in llm_setting:
            model_path = config["model2path"].get(llm_name, None)
        else:
            model_path = llm_setting["model_path"]
        if model_path is None:
            assert False, "None model path "

        from transformers import pipeline

        self.llm_pipeline = pipeline("text2text-generation", model=model_path, device=0)

    def extract_judge_score(answer: str, split_str: str = "Total rating:") -> int:
        try:
            if split_str in answer:
                rating = answer.split(split_str)[1]
            else:
                rating = answer
            digit_groups = [el.strip() for el in re.findall(r"\d+(?:\.\d+)?", rating)]
            return float(digit_groups[0])
        except Exception as e:
            print(e)
            return 0

    def calculate_metric(self, data):
        question_list = data.question
        pred_list = data.pred

        judge_input_prompt = [self.JUDGE_PROMPT.format(question=q, answer=a) for q, a in zip(question_list, pred_list)]
        judge_output = self.llm_pipeline(judge_input_prompt, max_new_tokens=100, batch_size=8)
        judge_output = [item["generated_text"] for item in judge_output]

        metric_score_list = [self.extract_judge_score(o) for o in judge_output]
        # rescale score
        metric_score_list = [score / 10 + 1 for score in metric_score_list]

        score = sum(metric_score_list) / len(metric_score_list)

        return {"llm_judge_score": score}, metric_score_list


class CountToken(BaseMetric):
    metric_name = "input_tokens"

    def __init__(self, config):
        super().__init__(config)
        tokenizer_name = config["metric_setting"].get("tokenizer_name", None)
        is_hf_tokenizer = True
        from flashrag.utils.constants import OPENAI_MODEL_DICT

        if tokenizer_name is None or tokenizer_name in OPENAI_MODEL_DICT:
            # use gpt4 tokenizer
            import tiktoken

            if tokenizer_name is None:
                tokenizer_name = "gpt-4"
            tokenizer = tiktoken.encoding_for_model(tokenizer_name)
            is_hf_tokenizer = False
        else:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.tokenizer = tokenizer
        self.is_hf_tokenizer = is_hf_tokenizer

    def _prompt_to_text(self, prompt):
        if isinstance(prompt, str):
            return prompt

        if isinstance(prompt, list):
            text_chunks = []
            for message in prompt:
                if not isinstance(message, dict):
                    continue
                content = message.get("content", "")
                if isinstance(content, str):
                    text_chunks.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_chunks.append(item.get("text", ""))
            return "\n".join([chunk for chunk in text_chunks if chunk])

        return str(prompt)

    def calculate_metric(self, data):
        try:
            input_prompts = data.prompt
        except AttributeError:
            return {"avg_input_tokens": 0.0}, [0.0 for _ in data]

        prompt_texts = [self._prompt_to_text(prompt) for prompt in input_prompts]
        if self.is_hf_tokenizer:
            token_counts = [len(self.tokenizer.tokenize(text)) for text in prompt_texts]
        else:
            token_counts = [len(self.tokenizer.encode(text)) for text in prompt_texts]
        avg_tokens = sum(token_counts) / len(token_counts) if len(token_counts) > 0 else 0.0

        return {"avg_input_tokens": avg_tokens}, token_counts

class GAOKAOMM_Accuracy(BaseMetric):
    metric_name = 'gaokao_acc'
    def __init__(self, config):
        super().__init__(config)
    
    def calculate_metric(self, data):
        metric_dict = {}
        acc_list = []
        for item in data:
            golden_answers = item.golden_answers
            golden_answers = [i.lower() for i in golden_answers]
            golden_answer = "".join(golden_answers)
            pred = item.pred.lower()
            subject = item.subject

            question_type = item.question_type
            if question_type == 'single_choice':
                acc = 1.0 if pred == golden_answer else 0.0
            else:
                if pred == golden_answer:
                    acc = 1.0
                elif pred in golden_answer:
                    acc = 0.5
                else:
                    acc = 0.0
            acc_list.append(acc)
            if subject not in metric_dict:
                metric_dict[subject] = []
            metric_dict[subject].append(acc)
        for key, value in metric_dict.items():
            metric_dict[key] = np.mean(value)
        
        metric_dict['avg_score'] = np.mean(acc_list)
        return metric_dict, acc_list 
                
