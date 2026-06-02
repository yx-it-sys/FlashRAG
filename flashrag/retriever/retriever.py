import json
import os
import time
import requests
from collections import defaultdict
from io import BytesIO

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
from typing import List, Dict, Union
import functools
from tqdm import tqdm
import faiss
import copy
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from flashrag.utils import get_reranker, get_device
from flashrag.retriever.utils import load_corpus, load_docs, convert_numpy, judge_image, judge_zh
from flashrag.retriever.encoder import Encoder, STEncoder, ClipEncoder
import torch
from PIL import Image

if get_device() == "cpu":
    faiss.omp_set_num_threads(1)

def cache_manager(func):
    """
    Decorator used for retrieving document cache.
    With the decorator, The retriever can store each retrieved document as a file and reuse it.
    """

    @functools.wraps(func)
    def wrapper(self, query=None, num=None, return_score=False):
        if num is None:
            num = self.topk
        if self.use_cache:
            if isinstance(query, str):
                new_query_list = [query]
            else:
                new_query_list = query

            no_cache_query = []
            cache_results = []
            for new_query in new_query_list:
                if new_query in self.cache:
                    cache_res = self.cache[new_query]
                    if len(cache_res) < num:
                        warnings.warn(f"The number of cached retrieval results is less than topk ({num})")
                    cache_res = cache_res[:num]
                    # separate the doc score
                    doc_scores = [item["score"] for item in cache_res]
                    cache_results.append((cache_res, doc_scores))
                else:
                    cache_results.append(None)
                    no_cache_query.append(new_query)

            if no_cache_query != []:
                # use batch search without decorator
                no_cache_results, no_cache_scores = self._batch_search_with_rerank(no_cache_query, num, True)
                no_cache_idx = 0
                for idx, res in enumerate(cache_results):
                    if res is None:
                        assert new_query_list[idx] == no_cache_query[no_cache_idx]
                        cache_results[idx] = (
                            no_cache_results[no_cache_idx],
                            no_cache_scores[no_cache_idx],
                        )
                        no_cache_idx += 1

            results, scores = (
                [t[0] for t in cache_results],
                [t[1] for t in cache_results],
            )

        else:
            results, scores = func(self, query=query, num=num, return_score=True)

        if self.save_cache:
            # merge result and score
            save_results = results.copy()
            save_scores = scores.copy()
            if isinstance(query, str):
                query = [query]
                if "batch" not in func.__name__:
                    save_results = [save_results]
                    save_scores = [save_scores]
            for new_query, doc_items, doc_scores in zip(query, save_results, save_scores):
                for item, score in zip(doc_items, doc_scores):
                    item["score"] = score
                self.cache[new_query] = doc_items

        if return_score:
            return results, scores
        else:
            return results

    return wrapper


def rerank_manager(func):
    """
    Decorator used for reranking retrieved documents.
    """

    @functools.wraps(func)
    def wrapper(self, query, num=None, return_score=False):
        results, scores = func(self, query=query, num=num, return_score=True)
        if self.use_reranker:
            results, scores = self.reranker.rerank(query, results)
            if "batch" not in func.__name__:
                results = results[0]
                scores = scores[0]
        if return_score:
            return results, scores
        else:
            return results

    return wrapper


class BaseRetriever:
    """Base object for all retrievers."""

    def __init__(self, config):
        self._config = config
        self.update_config()

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config_data):
        self._config = config_data
        self.update_config()

    def update_config(self):
        self.update_base_setting()
        self.update_additional_setting()

    def update_base_setting(self):
        self.retrieval_method = self._config["retrieval_method"]
        self.text_retrieval_topk = self._config["text_retrieval_topk"]
        self.image_retrieval_topk = self._config["image_retrieval_topk"]
        self.index_path = self._config["index_path"]
        self.corpus_path = self._config["corpus_path"]

        self.save_cache = self._config["save_retrieval_cache"]
        self.use_cache = self._config["use_retrieval_cache"]
        self.cache_path = self._config["retrieval_cache_path"]

        self.use_reranker = self._config["use_reranker"]
        if self.use_reranker:
            self.reranker = get_reranker(self._config)
        else:
            self.reranker = None

        if self.save_cache:
            self.cache_save_path = os.path.join(self._config["save_dir"], "retrieval_cache.json")
            self.cache = {}
        if self.use_cache:
            assert self.cache_path is not None
            with open(self.cache_path, "r") as f:
                self.cache = json.load(f)
        self.silent = self._config["silent_retrieval"] if "silent_retrieval" in self._config else False

    def update_additional_setting(self):
        pass

    @staticmethod
    def _normalize_dedup_value(value):
        if value is None:
            return ""
        return str(value).strip()

    def _build_result_dedup_key(self, item):
        if isinstance(item, str):
            return ("text", self._normalize_dedup_value(item))

        if isinstance(item, dict):
            content_fields = [
                "title",
                "text",
                "contents",
                "page_name",
                "page_snippet",
                "url",
                "image_url",
            ]
            content_key = tuple(self._normalize_dedup_value(item.get(field, "")) for field in content_fields)
            if any(content_key):
                return ("dict", content_key)
            return ("dict_json", json.dumps(item, sort_keys=True, ensure_ascii=False))

        return ("raw", self._normalize_dedup_value(item))

    def _dedup_topk_results(self, results, scores=None, num=None):
        deduped_results = []
        deduped_scores = [] if scores is not None else None
        seen_keys = set()

        for idx, result in enumerate(results):
            dedup_key = self._build_result_dedup_key(result)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            deduped_results.append(result)
            if scores is not None:
                deduped_scores.append(scores[idx])
            if num is not None and len(deduped_results) >= num:
                break

        if scores is not None:
            return deduped_results, deduped_scores
        return deduped_results

    def _save_cache(self):
        self.cache = convert_numpy(self.cache)

        def custom_serializer(obj):
            if isinstance(obj, np.float32):
                return float(obj)
            raise TypeError(f"Type {type(obj)} not serializable")

        with open(self.cache_save_path, "w") as f:
            json.dump(self.cache, f, indent=4, default=custom_serializer)

    def _search(self, query: str, num: int, return_score: bool) -> List[Dict[str, str]]:
        r"""Retrieve topk relevant documents in corpus.

        Return:
            list: contains information related to the document, including:
                contents: used for building index
                title: (if provided)
                text: (if provided)

        """

        pass

    def _batch_search(self, query, num, return_score):
        pass

    def search(self, *args, **kwargs):
        return self._search(*args, **kwargs)

    def batch_search(self, *args, **kwargs):
        return self._batch_search(*args, **kwargs)


class BaseTextRetriever(BaseRetriever):
    """Base text retriever."""

    def __init__(self, config):
        super().__init__(config)

    @cache_manager
    @rerank_manager
    def search(self, *args, **kwargs):
        return self._search(*args, **kwargs)

    @cache_manager
    @rerank_manager
    def batch_search(self, *args, **kwargs):
        return self._batch_search(*args, **kwargs)

    @rerank_manager
    def _batch_search_with_rerank(self, *args, **kwargs):
        return self._batch_search(*args, **kwargs)

    @rerank_manager
    def _search_with_rerank(self, *args, **kwargs):
        return self._search(*args, **kwargs)


class BM25Retriever(BaseTextRetriever):
    r"""BM25 retriever based on pre-built pyserini index."""

    def __init__(self, config, corpus=None):
        super().__init__(config)
        self.load_model_corpus(corpus)

    def update_additional_setting(self):
        self.backend = self._config["bm25_backend"]

    def load_model_corpus(self, corpus):
        if self.backend == "pyserini":
            # Warning: the method based on pyserini will be deprecated
            from pyserini.search.lucene import LuceneSearcher

            self.searcher = LuceneSearcher(self.index_path)
            self.contain_doc = self._check_contain_doc()
            if not self.contain_doc:
                if corpus is None:
                    self.corpus = load_corpus(self.corpus_path)
                else:
                    self.corpus = corpus
            self.max_process_num = 8
               
        elif self.backend == "bm25s":
            import Stemmer
            import bm25s

            self.corpus = load_corpus(self.corpus_path)
            is_zh = judge_zh(self.corpus[0]["contents"])

            self.searcher = bm25s.BM25.load(self.index_path, mmap=True, load_corpus=False)
            if is_zh:
                self.tokenizer = bm25s.tokenization.Tokenizer(stopwords="zh")
                self.tokenizer.load_stopwords(self.index_path)
                self.tokenizer.load_vocab(self.index_path)
            else:
                stemmer = Stemmer.Stemmer("english")
                self.tokenizer = bm25s.tokenization.Tokenizer(stopwords="en", stemmer=stemmer)
                self.tokenizer.load_stopwords(self.index_path)
                self.tokenizer.load_vocab(self.index_path)

            self.searcher.corpus = self.corpus
            self.searcher.backend = "numba"

        else:
            assert False, "Invalid bm25 backend!"

    def _check_contain_doc(self):
        r"""Check if the index contains document content"""
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score=False) -> List[Dict[str, str]]:
        if num is None:
            num = self.topk
        if self.backend == "pyserini":
            is_zh = judge_zh(query)
            if is_zh:
                self.searcher.set_language("zh")
            hits = self.searcher.search(query, num)
            if len(hits) < 1:
                if return_score:
                    return [], []
                else:
                    return []

            scores = [hit.score for hit in hits]
            if len(hits) < num:
                warnings.warn("Not enough documents retrieved!")
            else:
                hits = hits[:num]

            if self.contain_doc:
                all_contents = [json.loads(self.searcher.doc(hit.docid).raw())["contents"] for hit in hits]
                results = [
                    {
                        "id": hit.docid, 
                        "title": content.split("\n")[0].strip('"'),
                        "text": "\n".join(content.split("\n")[1:]),
                        "contents": content,
                    }
                    for content, hit in zip(all_contents, hits)
                ]
            else:
                results = load_docs(self.corpus, [hit.docid for hit in hits])
        elif self.backend == "bm25s":
            import bm25s

            # query_tokens = self.tokenizer.tokenize([query], return_as="tuple", update_vocab=False)
            # Original :query_tokens = bm25s.tokenize([query])
            query_tokens = bm25s.tokenize(query)
            results, scores = self.searcher.retrieve(query_tokens, k=num)
            results = list(results[0])
            scores = list(scores[0])
        else:
            assert False, "Invalid bm25 backend!"

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query, num: int = None, return_score=False):
        if self.backend == "pyserini":
            # TODO: modify batch method
            results = []
            scores = []
            for _query in query:
                item_result, item_score = self._search(_query, num, True)
                results.append(item_result)
                scores.append(item_score)
        elif self.backend == "bm25s":
            import bm25s

            # query_tokens = self.tokenizer.tokenize(query, return_as="tuple", update_vocab=False)
            query_tokens = bm25s.tokenize(query)
            results, scores = self.searcher.retrieve(query_tokens, k=num)
        else:
            assert False, "Invalid bm25 backend!"
        results = results.tolist() if isinstance(results, np.ndarray) else results
        scores = scores.tolist() if isinstance(scores, np.ndarray) else scores
        if return_score:
            return results, scores
        else:
            return results


class DenseRetriever(BaseTextRetriever):
    r"""Dense retriever based on pre-built faiss index."""

    def __init__(self, config: dict, corpus=None):
        super().__init__(config)

        self.load_corpus(corpus)
        self.load_index()
        self.load_model()

    def load_corpus(self, corpus):
        if corpus is None:
            self.corpus = load_corpus(self.corpus_path)
        else:
            self.corpus = corpus

    def load_index(self):
        if self.index_path is None or not os.path.exists(self.index_path):
            raise Warning(f"Index file {self.index_path} does not exist!")
        self.index = faiss.read_index(self.index_path)
        if self.use_faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

    def update_additional_setting(self):
        self.query_max_length = self._config["retrieval_query_max_length"]
        self.pooling_method = self._config["retrieval_pooling_method"]
        self.use_fp16 = self._config["retrieval_use_fp16"]
        self.batch_size = self._config["retrieval_batch_size"]
        self.instruction = self._config["instruction"]

        self.retrieval_model_path = self._config["retrieval_model_path"]
        self.use_st = self._config["use_sentence_transformer"]
        self.use_faiss_gpu = self._config["faiss_gpu"]

    def load_model(self):
        if self.use_st:
            self.encoder = STEncoder(
                model_name=self.retrieval_method,
                model_path=self._config["retrieval_model_path"],
                max_length=self.query_max_length,
                use_fp16=self.use_fp16,
                instruction=self.instruction,
                silent=self.silent,
            )
        else:
            # check pooling method
            self._check_pooling_method(self.retrieval_model_path, self.pooling_method)
            self.encoder = Encoder(
                model_name=self.retrieval_method,
                model_path=self.retrieval_model_path,
                pooling_method=self.pooling_method,
                max_length=self.query_max_length,
                use_fp16=self.use_fp16,
                instruction=self.instruction,
            )

    def _check_pooling_method(self, model_path, pooling_method):
        try:
            # read pooling method from 1_Pooling/config.json
            pooling_config = json.load(open(os.path.join(model_path, "1_Pooling/config.json")))
            for k, v in pooling_config.items():
                if k.startswith("pooling_mode") and v == True:
                    detect_pooling_method = k.split("pooling_mode_")[-1]
                    if detect_pooling_method == "mean_tokens":
                        detect_pooling_method = "mean"
                    elif detect_pooling_method == "cls_token":
                        detect_pooling_method = "cls"
                    else:
                        # raise warning: not implemented pooling method
                        warnings.warn(f"Pooling method {detect_pooling_method} is not implemented.", UserWarning)
                        detect_pooling_method = "mean"
                    break
        except:
            detect_pooling_method = None

        if detect_pooling_method is not None and detect_pooling_method != pooling_method:
            warnings.warn(
                f"Pooling method in model config file is {detect_pooling_method}, but the input is {pooling_method}. Please check carefully."
            )

    def _search(self, query: str, num: int = None, return_score=False):
        if num is None:
            num = self.text_retrieval_topk
        query_emb = self.encoder.encode(query)
        candidate_k = min(max(num * 5, num), self.index.ntotal)
        scores, idxs = self.index.search(query_emb, k=candidate_k)
        scores = scores.tolist()
        idxs = idxs[0]
        scores = scores[0]

        results = load_docs(self.corpus, idxs)
        results, scores = self._dedup_topk_results(results, scores, num)
        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query: List[str], num: int = None, return_score=False):
        if isinstance(query, str):
            query = [query]
        if num is None:
            num = self.text_retrieval_topk
        batch_size = self.batch_size
        candidate_k = min(max(num * 5, num), self.index.ntotal)

        results = []
        scores = []
        emb = self.encoder.encode(query, batch_size=batch_size, is_query=True)
        scores, idxs = self.index.search(emb, k=candidate_k)
        scores = scores.tolist()
        idxs = idxs.tolist()

        flat_idxs = sum(idxs, [])
        results = load_docs(self.corpus, flat_idxs)
        results = [results[i * candidate_k : (i + 1) * candidate_k] for i in range(len(idxs))]
        deduped_results = []
        deduped_scores = []
        for query_results, query_scores in zip(results, scores):
            query_results, query_scores = self._dedup_topk_results(query_results, query_scores, num)
            deduped_results.append(query_results)
            deduped_scores.append(query_scores)
        results = deduped_results
        scores = deduped_scores

        if return_score:
            return results, scores
        else:
            return results


class MultiModalRetriever(BaseRetriever):
    r"""Multi-modal retriever based on pre-built faiss index."""

    def __init__(self, config: dict, corpus=None):
        super().__init__(config)
        self.mm_index_dict = config[
            "multimodal_index_path_dict"
        ]  # {"text": "path/to/text_index", "image": "path/to/image_index"}
        self.index_dict = {"text": None, "image": None}
        for modal in ["text", "image"]:
            idx_path = self.mm_index_dict[modal]
            if idx_path is not None:
                self.index_dict[modal] = faiss.read_index(idx_path)
            if config["faiss_gpu"]:
                co = faiss.GpuMultipleClonerOptions()
                co.useFloat16 = True
                co.shard = True
                self.index_dict[modal] = faiss.index_cpu_to_all_gpus(self.index_dict[modal], co=co)
        if corpus is None:
            self.corpus = load_corpus(self.corpus_path)
        else:
            self.corpus = corpus
        self.text_retrieval_topk = config["text_retrieval_topk"]
        self.image_retrieval_topk = config["image_retrieval_topk"]
        self.batch_size = config["retrieval_batch_size"]

        self.encoder = ClipEncoder(
            model_name=self.retrieval_method, model_path=config["retrieval_model_path"], silent=self.silent
        )

    def _judge_input_modal(self, query):
        if not isinstance(query, str):
            return "image"
        else:
            if query.startswith("http") or query.endswith(".jpg") or query.endswith(".png"):
                return "image"
            else:
                return "text"

    def _search(self, query, target_modal: str = "text", num: int = None, return_score=False):
        if num is None:
            num = self.text_retrieval_topk if target_modal == "text" else self.image_retrieval_topk
        assert target_modal in ["image", "text"]

        query_modal = (
            self._judge_input_modal(query) if not isinstance(query, list) else self._judge_input_modal(query[0])
        )
        if query_modal == "image" and isinstance(query, str):
            from PIL import Image

            if os.path.exists(query):
                query = Image.open(query)
            else:
                import requests

                query = Image.open(requests.get(query, stream=True).raw)

        query_emb = self.encoder.encode(query, modal=query_modal)

        scores, idxs = self.index_dict[target_modal].search(query_emb, k=num)
        scores = scores.tolist()
        idxs = idxs[0]
        scores = scores[0]

        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores
        else:
            return results

class MultiRetrieverRouter:
    def __init__(self, config):
        self.merge_method = config["multi_retriever_setting"].get("merge_method", "concat")  # concat/rrf/rerank
        self.final_topk = config["multi_retriever_setting"].get("topk", 5)
        self.retriever_list = self.load_all_retriever(config)
        self.config = config

        if self.merge_method == "rerank":
            config["multi_retriever_setting"]["rerank_topk"] = self.final_topk
            config["multi_retriever_setting"]["device"] = config["device"]
            self.reranker = get_reranker(config["multi_retriever_setting"])

    def load_all_retriever(self, config):
        retriever_config_list = config["multi_retriever_setting"]["retriever_list"]
        # use the same corpus for efficient memory usage
        all_corpus_dict = {}
        retriever_list = []
        for retriever_config in retriever_config_list:
            retrieval_method = retriever_config["retrieval_method"]
            print(f"Loading {retrieval_method} retriever...")
            retrieval_model_path = retriever_config["retrieval_model_path"]
            corpus_path = retriever_config["corpus_path"]

            if retrieval_method == "mcsearch":
                retriever = MCSearchRetriever(retriever_config)
            elif retrieval_method == "bm25":
                if corpus_path is None:
                    corpus = None
                else:
                    if corpus_path in all_corpus_dict:
                        corpus = all_corpus_dict[corpus_path]
                    else:
                        corpus = load_corpus(corpus_path)
                        all_corpus_dict[corpus_path] = corpus
                retriever = BM25Retriever(retriever_config, corpus)
            else:
                if corpus_path in all_corpus_dict:
                    corpus = all_corpus_dict[corpus_path]
                else:
                    corpus = load_corpus(corpus_path)
                    all_corpus_dict[corpus_path] = corpus

                # judge modality
                from transformers import AutoConfig

                try:
                    model_config = AutoConfig.from_pretrained(retrieval_model_path)
                    arch = model_config.architectures[0]
                    print("arch: ", arch)
                    if "clip" in arch.lower():
                        retriever = MultiModalRetriever(retriever_config, corpus)
                    else:
                        retriever = DenseRetriever(retriever_config, corpus)
                except:
                    retriever = DenseRetriever(retriever_config, corpus)

            retriever_list.append(retriever)

        return retriever_list

    def add_source(self, result: Union[list, tuple], retriever):
        retrieval_method = retriever.retrieval_method
        corpus_path = retriever.corpus_path
        is_multimodal = isinstance(retriever, (MultiModalRetriever, MCSearchRetriever))
        # for naive search, result is a list of dict, each repr a doc
        # for batch search, result is a list of list, each repr a doc list(per query)
        for item in result:
            if isinstance(item, list):
                for _item in item:
                    _item["source"] = retrieval_method
                    _item["corpus_path"] = corpus_path
                    _item["is_multimodal"] = is_multimodal
            else:
                item["source"] = retrieval_method
                item["corpus_path"] = corpus_path
                item["is_multimodal"] = is_multimodal
        return result

    def _search_or_batch_search(self, query: Union[str, list], target_modal, num, return_score, method, retriever_list):
        if num is None:
            num = self.final_topk

        result_list = []
        score_list = []

        def process_retriever(retriever):
            is_multimodal = isinstance(retriever, (MultiModalRetriever, MCSearchRetriever))
            params = {"query": query, "return_score": return_score}

            if is_multimodal:
                params["target_modal"] = target_modal

            if method == "search":
                output = retriever.search(**params)
            else:
                output = retriever.batch_search(**params)

            if return_score:
                result, score = output
            else:
                result = output
                score = None

            result = self.add_source(result, retriever)
            return result, score

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_retriever = {
                executor.submit(process_retriever, retriever): retriever for retriever in retriever_list
            }
            for future in as_completed(future_to_retriever):
                try:
                    result, score = future.result()
                    result_list.extend(result)
                    if score is not None:
                        score_list.extend(score)
                except Exception as e:
                    print(f"Error processing retriever {future_to_retriever[future]}: {e}")
        result_list, score_list = self.reorder(result_list, score_list, retriever_list)
        result_list, score_list = self.post_process_result(query, result_list, score_list, num)
        if return_score:
            return result_list, score_list
        else:
            return result_list

    def reorder(self, result_list, score_list, retriever_list):
        """
        batch_search:
        original result like: [[bm25-q1-d1, bm25-q1-d2],[bm25-q2-d1, bm25-q2-d2], [e5-q1-d1, e5-q1-d2], [e5-q2-d1, e5-q2-d2]]
        reorder to: [[bm25-q1-d1, bm25-q1-d2, e5-q1-d1, e5-q1-d2], [bm25-q2-d1,bm25-q2-d2, e5-q2-d1, e5-q2-d2]]

        navie search:
        original result like: [bm25-d1, bm25-d2, e5-d1, e5-d2]

        """

        retriever_num = len(retriever_list)
        query_num = len(result_list) // retriever_num
        assert query_num * retriever_num == len(result_list)

        if isinstance(result_list[0], dict):
            return result_list, score_list

        final_result = []
        final_score = []
        for q_idx in range(query_num):
            final_result.append(sum([result_list[q_idx + r_idx * query_num] for r_idx in range(retriever_num)], []))
            if score_list != []:
                final_score.append(sum([score_list[q_idx + r_idx * query_num] for r_idx in range(retriever_num)], []))
        return final_result, final_score

    def post_process_result(self, query: Union[str, list], result_list, score_list, num):
        # based on self.merge_method
        if self.merge_method == "concat":
            # remove duplicate doc
            if isinstance(result_list[0], dict):
                exist_id = set()
                for idx, doc in enumerate(result_list):
                    if doc["id"] not in exist_id:
                        exist_id.add(doc["id"])
                    else:
                        result_list.remove(doc)
                        if score_list != []:
                            score_list.remove(idx)
            else:
                for query_idx, query_doc_list in enumerate(result_list):
                    exist_id = set()
                    for doc_idx, doc in enumerate(query_doc_list):
                        if doc["id"] not in exist_id:
                            exist_id.add(doc["id"])
                        else:
                            query_doc_list.remove(doc)
                            if score_list != []:
                                score_list[query_idx].remove(doc_idx)
            return result_list, score_list
        elif self.merge_method == "rrf":
            if (isinstance(result_list[0], dict) and len(set([doc["corpus_path"] for doc in result_list])) > 1) or (
                isinstance(result_list[0], list) and len(set([doc["corpus_path"] for doc in result_list[0]])) > 1
            ):
                warnings.warn(
                    "Using multiple corpus may lead to conflicts in DOC IDs, which may result in incorrect rrf results!"
                )
            if isinstance(result_list[0], dict):
                result_list, score_list = self.rrf_merge([result_list], num, k=60)
                result_list = result_list[0]
                score_list = score_list[0]
            else:
                result_list, score_list = self.rrf_merge(result_list, num, k=60)
            return result_list, score_list
        elif self.merge_method == "rerank":
            if isinstance(result_list[0], dict):
                query, result_list, score_list = [query], [result_list], [score_list]
            # parse the result of multimodal corpus
            for item_result in result_list:
                for item in item_result:
                    if item["is_multimodal"]:
                        item["contents"] = item["text"]
            # rerank all docs
            print(result_list)
            result_list, score_list = self.reranker.rerank(query, result_list, topk=num)
            if isinstance(query, str):
                result_list, score_list = result_list[0], score_list[0]
            return result_list, score_list
        else:
            raise NotImplementedError

    def rrf_merge(self, results, topk=10, k=60):
        """
        Perform Reciprocal Rank Fusion (RRF) on retrieval results.

        Args:
            results (list of list of dict): Retrieval results for multiple queries.
            topk (int): Number of top results to return per query.
            k (int): RRF hyperparameter to adjust rank contribution.

        Returns:
            list of list of dict: Fused results with topk highest scores per query.
        """
        fused_results = []
        fused_scores = []
        for query_results in results:
            # Initialize a score dictionary to accumulate RRF scores
            score_dict = {}
            retriever_result_dict = {}
            id2item = {}
            for item in query_results:
                source = item["source"]
                if source not in retriever_result_dict:
                    retriever_result_dict[source] = []
                retriever_result_dict[source].append(item["id"])
                id2item[item["id"]] = item

            # Calculate RRF scores for each document
            for retriever, retriever_result in retriever_result_dict.items():
                for rank, doc_id in enumerate(retriever_result, start=1):
                    if doc_id not in score_dict:
                        score_dict[doc_id] = 0
                    # Add RRF score for the document
                    score_dict[doc_id] += 1 / (k + rank)

            # Sort by accumulated RRF score
            sorted_results = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

            # Keep only the topk results
            top_ids = [i[0] for i in sorted_results[:topk]]
            top_scores = [i[1] for i in sorted_results[:topk]]

            fused_results.append([id2item[id] for id in top_ids])
            fused_scores.append(top_scores)

        return fused_results, fused_scores

    def search(self, query, target_modal="text", num: Union[list, int, None] = None, return_score=False):
        # query: str or PIL.Image
        # judge query type: text or image
        if judge_image(query):
            retriever_list = [
                retriever for retriever in self.retriever_list if isinstance(retriever, MultiModalRetriever)
            ]
        else:
            retriever_list = self.retriever_list
        if target_modal == "image":
            # remove text retriever
            retriever_list = [retriever for retriever in retriever_list if isinstance(retriever, MultiModalRetriever)]

        return self._search_or_batch_search(
            query, target_modal, num, return_score, method="search", retriever_list=retriever_list
        )

    def batch_search(self, query, target_modal="text", num: Union[list, int, None] = None, return_score=False):
        # judge query type: text or image
        if not isinstance(query, list):
            query = [query]
        if target_modal == "image":
            self._retriever_list = [
                retriever for retriever in self.retriever_list if isinstance(retriever, MultiModalRetriever)
            ]
        else:
            self._retriever_list = self.retriever_list
        query_type_list = [judge_image(q) for q in query]
        if all(query_type_list):
            # all query is image
            if self.merge_method == "rerank":
                warnings.warn("merge_method is rerank, but all query is image, use default method `concat` instead")
                self.merge_method = "concat"
            retriever_list = [
                retriever for retriever in self._retriever_list if isinstance(retriever, MultiModalRetriever)
            ]

            return self._search_or_batch_search(
                query, target_modal, num, return_score, method="batch_search", retriever_list=retriever_list
            )
        elif all([not t for t in query_type_list]):
            # all query is text
            # if exist text retriever, don't use mm retriever for text-text search
            if any([isinstance(retriever, BaseTextRetriever) for retriever in self._retriever_list]):
                self._retriever_list = [
                    retriever for retriever in self._retriever_list if not isinstance(retriever, MultiModalRetriever)
                ]
            return self._search_or_batch_search(
                query, target_modal, num, return_score, method="batch_search", retriever_list=self._retriever_list
            )
        else:
            # query list is the mix of image and text
            if self.merge_method == "rerank":
                warnings.warn("merge_method is rerank, but some query is image, use default method `concat` instead")
                self.merge_method = "concat"
            image_query_idx = [i for i, t in enumerate(query_type_list) if t]
            image_query_list = [query[i] for i in image_query_idx]
            text_query_list = [q for q in query if q not in image_query_list]

            text_output = self._search_or_batch_search(
                text_query_list,
                target_modal,
                num,
                return_score,
                method="batch_search",
                retriever_list=self._retriever_list,
            )
            retriever_list = [
                retriever for retriever in self._retriever_list if isinstance(retriever, MultiModalRetriever)
            ]
            image_output = self._search_or_batch_search(
                text_query_list, target_modal, num, return_score, method="batch_search", retriever_list=retriever_list
            )

            # merge text output and image output
            if return_score:
                text_result, text_score = text_output
                image_result, image_score = image_output
                final_result = []
                final_score = []
                text_idx = 0
                image_idx = 0
                for idx in range(len(query)):
                    if idx not in image_query_idx:
                        final_result.append(text_result[text_idx])
                        final_score.append(text_score[text_idx])
                        text_idx += 1
                    else:
                        final_result.append(image_result[image_idx])
                        final_score.append(image_score[image_idx])
                        image_idx += 1
                return final_result, final_score
            else:
                final_result = []
                text_idx = 0
                image_idx = 0
                for idx in range(len(query)):
                    if idx not in image_query_idx:
                        final_result.append(text_result[text_idx])
                        text_idx += 1
                    else:
                        final_result.append(image_result[image_idx])
                        image_idx += 1
                return final_result


class SparseRetriever(BaseTextRetriever):
    """Sparse embedding retriever supporting only SPLADE with Seismic backend for now."""

    def __init__(self, config):
        super().__init__(config)

        import multiprocessing
        self.cores = str(multiprocessing.cpu_count())
        os.environ["RAYON_NUM_THREADS"] = self.cores

        self.progress_bar = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.corpus = load_corpus(config["corpus_path"])
        self.tokenizer, self.model = self._load_sparse_model()

        self.id = 0

        self.update_additional_setting()

        self.seismic_query_cut = self.config["seismic_query_cut"]
        self.seismic_heap_factor = self.config["seismic_heap_factor"]
        self.index_max_tokens = self.config["seismic_max_tokens_length"]
        self._init_seismic_index()

    def update_additional_setting(self):
        """Load config shared for all the models supported"""
        self.query_max_length = self._config["retrieval_query_max_length"]
        self.use_fp16 = self._config["retrieval_use_fp16"]
        self.batch_size = self._config["retrieval_batch_size"]
        self.retrieval_model_path = self._config["retrieval_model_path"]
        self.pooling_method = self._config["retrieval_pooling_method"]

    def _init_seismic_index(self):
        """Initialize Seismic index."""
        from seismic import SeismicIndex  # Assuming this is available
        self.seismic_index = SeismicIndex.load(self.index_path)
        self.string_type = f'U{self.index_max_tokens}'  # For Seismic string dtype

    def _load_sparse_model(self):
        """Load tokenizer and model based on sparse type."""
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(self.retrieval_model_path)
        model = AutoModelForMaskedLM.from_pretrained(self.retrieval_model_path)

        if self.use_fp16:
            model = model.half()

        # Use more gpus if available
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, device_ids=self.config['gpu_id'].split(','))

        model = model.to(self.device)
        model.eval()
        return tokenizer, model

    def _encode(self, query):
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.query_max_length,
            add_special_tokens=True
        ).to(self.model.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits  # [batch_size, seq_len, vocab_size]
            attention_mask = inputs["attention_mask"].unsqueeze(-1)  # [batch, seq_len, 1]

            scores = torch.log1p(torch.relu(logits)) * attention_mask
            v_repr = torch.max(scores, dim=1)[0]  # [batch_size, vocab_size]

            # Move to CPU (it seems much faster)
            v_repr = v_repr.cpu()
            nonzero_mask = v_repr > 1e-4

            # Get sparse values and indices in batch
            batch_indices, token_indices = torch.nonzero(nonzero_mask, as_tuple=True)
            token_scores = v_repr[batch_indices, token_indices]

            # Convert once all token IDs to strings (batched)
            unique_token_ids = torch.unique(token_indices)
            token_id_to_token = {
                idx.item(): tok for idx, tok in zip(
                    unique_token_ids, self.tokenizer.convert_ids_to_tokens(unique_token_ids.tolist())
                )
            }

            # Build final embeddings
            from collections import defaultdict
            embeddings = defaultdict(dict)
            for b_idx, t_idx, score in zip(batch_indices, token_indices, token_scores):
                embeddings[b_idx.item()][token_id_to_token[t_idx.item()]] = round(score.item(), 4)

            # Convert to list for each document
            return [embeddings[i] for i in range(len(query))]

    def search(self, query: list, num: int = None, return_score=False) -> (List[Dict], List[float]):
        """Search using sparse vector."""
        num = num or self.topk

        query_vec = self._encode(query)
        results, scores = self._seismic_search(query_vec, num)

        if return_score:
            return results, scores
        else:
            return results

    def batch_search(self, query, num=None, return_score=False):
        """Search using sparse vector."""
        if isinstance(query, str):
            query = [query]

        if self.pooling_method != 'max':
            print(
                f'Pooling method: {self.pooling_method.upper()} not supported on sparse neural retrieval models. fallback to: MAX.')

        num = num or self.topk

        embeddings = []
        batch = []
        
        # Encode
        for i in range(len(query)):
            # Process batch
            batch.append(query[i])
            if len(batch) >= self.batch_size:
                query_vec = self._encode(batch)
                embeddings.extend(query_vec)
                batch = []

        if batch:
            query_vec = self._encode(batch)
            embeddings.extend(query_vec)

        # Search
        search_results = self._seismic_batch_search(embeddings, num)

        results = []
        scores = []
        for result in sorted(search_results, key=lambda e: int(e[0][0])):
            tmp_results = []
            tmp_scores = []

            for query_id, score, doc_id in result:
                tmp_results.append(self.corpus[int(doc_id)])
                tmp_scores.append(score)

            results.append(tmp_results)
            scores.append(tmp_scores)
        
        if return_score:
            return results, scores
        else:
            return results

    def _seismic_search(self, query_vec: List[Dict[str, float]], k: int) -> (List[Dict], List[float]):
        """Search using Seismic backend."""
        # Convert query to Seismic format
        results, scores = self.index_search(k, query_vec)
        return results[0], scores[0]

    def _seismic_batch_search(self, query_vecs: List[Dict[str, float]], k: int) -> (
            List[List[Dict]], List[List[float]]):
        """Batch search using Seismic backend (one query at a time)."""
        return self.index_search(k, query_vecs)

    def index_search(self, k, query_vec):
        max_len = max(len(query) for query in query_vec)
        pad_token = ""  # or whatever default is appropriate

        query_components = []
        query_values = []
        ids = []

        for query in query_vec:
            keys = list(query.keys())
            values = list(query.values())

            # Pad to max_len
            padded_keys = keys + [pad_token] * (max_len - len(keys))
            padded_values = values + [0.0] * (max_len - len(values))

            query_components.append(np.array(padded_keys, dtype='U30'))
            query_values.append(np.array(padded_values, dtype=np.float32))
            ids.append(self.id)
            self.id += 1

        ids = np.array(ids, dtype='U30')
        # Execute search
        search_results = self.seismic_index.batch_search(
            queries_ids=ids,  # Placeholder ID
            query_components=query_components,
            query_values=query_values,
            query_cut=self.seismic_query_cut,
            heap_factor=self.seismic_heap_factor,
            k=k,
            sorted=True,  # specified even if default value
            num_threads=int(self.cores)
        )
        return search_results

class SerperRetriever(BaseRetriever):
    """Retriever based on Google Serper API for web search."""

    def __init__(self, config):
        super().__init__(config)
        
        # Serper API specific configuration
        self.api_key = config["serper_api_key"]
        if not self.api_key:
            raise ValueError("serper_api_key is required in config")
        
        self.api_url = "https://google.serper.dev/search"
        self.search_type = config["serper_search_type"] if config["serper_search_type"] else "search"  # search, news, images, etc.
        self.location = config["serper_location"] if config["serper_location"] else None  # e.g., "United States"
        self.gl = config["serper_gl"] if config["serper_gl"] else None  # Country code, e.g., "us"
        self.hl = config["serper_hl"] if config["serper_hl"] else "en"  # Language, e.g., "en"
        
    def _search(self, query: str, num: int) -> List[Dict[str, str]]:
        """
        Retrieve top-k relevant documents using Google Serper API.
        
        Args:
            query: Search query string
            num: Number of results to return
            return_score: Whether to return relevance scores
            
        Returns:
            List of dictionaries containing search results with keys:
                - contents: The snippet/description
                - title: Page title
                - text: Full text (same as contents for web search)
                - url: Page URL
                - score: Relevance score (if return_score=True)
        """
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': query,
            'num': num,
            'hl': self.hl
        }
        
        if self.location:
            payload['location'] = self.location
        if self.gl:
            payload['gl'] = self.gl
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            
            # Parse organic results
            organic_results = data.get('organic', [])
            for idx, item in enumerate(organic_results[:num]):
                result = {
                    'title': item.get('title', ''),
                    'text': item.get('snippet', ''),
                    'url': item.get('link', ''),
                }
                
                results.append(result)
            
            return results
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling Serper API: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error in _search: {e}")
            return []
    
    def search(self, query: str, num: int = None) -> List[Dict[str, str]]:
        """
        Single search wrapper for SerperRetriever.
        """
        if num is None:
            num = self.topk
        return self._search(query, num)
    
    def _batch_search(self, query_list: List[str], num: int) -> List[List[Dict[str, str]]]:
        """
        Batch search for multiple queries.
        
        Args:
            query_list: List of query strings
            num: Number of results per query
            return_score: Whether to return relevance scores
            
        Returns:
            List of result lists, one for each query
        """
        results = []
        if num is None:
            num = self.topk
        
        for query in query_list:
            result = self._search(query, num)
            results.append(result)
            # Add a small delay to avoid rate limiting
            time.sleep(0.1)
        
        return results

    def batch_search(self, query_list: List[str], num: int = None):
        return self._batch_search(query_list, num)

import requests
from PIL import Image
from io import BytesIO
from cragmm_search.search import UnifiedSearchPipeline
from huggingface_hub import snapshot_download

class CRAGRetriever(BaseRetriever):
    """Retriever based on CRAG model for code search."""
    from cragmm_search.search import UnifiedSearchPipeline
    def __init__(self, config):
        super().__init__(config)
        hf_home = config["crag_hf_home"] if "crag_hf_home" in config else None
        if hf_home:
            os.environ["HF_HOME"] = hf_home
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))
            os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))
            os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "hub"))

        if "crag_offline" in config and config["crag_offline"]:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        crag_search_device = str(config["crag_search_device"]).lower()
        original_cuda_available = torch.cuda.is_available
        if crag_search_device == "cpu":
            torch.cuda.is_available = lambda: False
        try:
            # The upstream CragMockWeb eagerly scans metadata for ~900k rows during init,
            # which makes every startup look hung. Patch it to fetch metadata lazily per result.
            import cragmm_search.search as crag_search_module
            from huggingface_hub import snapshot_download
            import chromadb
            from cragmm_search.web_search_mock_api.api.web_search import index_web_data

            class LazyCragMockWeb:
                def __init__(self, emb_model, tokenizer, text_index_path, web_hf_dataset_tag=None):
                    self.vector_db = index_web_data(hf_path=text_index_path, revision=web_hf_dataset_tag)
                    self.emb_model = emb_model
                    self.tokenizer = tokenizer
                    self._metadata_cache = {}

                def _get_metadata(self, idx):
                    idx = str(idx)
                    if idx not in self._metadata_cache:
                        chunk = self.vector_db.get(ids=[idx], include=["metadatas"])
                        metadata = chunk["metadatas"][0] if chunk["metadatas"] else {}
                        self._metadata_cache[idx] = metadata
                    return self._metadata_cache[idx]

                def get_page_name(self, idx):
                    return self._get_metadata(idx).get("page_name")

                def get_page_snippet(self, idx):
                    return self._get_metadata(idx).get("page_snippet")

                def get_page_url(self, idx):
                    return self._get_metadata(idx).get("page_url")

            class LazyCragImageKG:
                def __init__(self, emb_model, processor, hf_dataset_id, image_hf_dataset_tag=None):
                    print(f"Loading image index from huggingface {hf_dataset_id}")
                    dataset_local_path = snapshot_download(
                        repo_id=hf_dataset_id,
                        repo_type="dataset",
                        revision=image_hf_dataset_tag,
                    )

                    n_threads = os.cpu_count() or 1
                    client = chromadb.PersistentClient(path=dataset_local_path)
                    self.vector_db = client.get_collection(name="image_embeddings")
                    self.vector_db.modify(metadata={"hnsw:num_threads": n_threads})
                    self.emb_model = emb_model
                    self.processor = processor
                    self._metadata_cache = {}
                    self._entity_cache = {}

                def _get_metadata(self, image_id):
                    image_id = int(image_id)
                    if image_id not in self._metadata_cache:
                        chunk = self.vector_db.get(ids=[str(image_id)], include=["metadatas"])
                        metadata = chunk["metadatas"][0] if chunk["metadatas"] else {}
                        self._metadata_cache[image_id] = metadata
                    return self._metadata_cache[image_id]

                def get_image_url(self, image_id):
                    return self._get_metadata(image_id).get("image_url")

                def get_entity_name(self, image_id):
                    metadata = self._get_metadata(image_id)
                    entities = metadata.get("entities", "[]")
                    return json.loads(entities)

                def get_entity(self, entity_name):
                    if entity_name not in self._entity_cache:
                        for metadata in self._metadata_cache.values():
                            info = metadata.get("info")
                            if not info:
                                continue
                            for name, entity in json.loads(info).items():
                                if name not in self._entity_cache:
                                    self._entity_cache[name] = entity
                        self._entity_cache.setdefault(entity_name, {})
                    return self._entity_cache[entity_name]

            crag_search_module.CragMockWeb = LazyCragMockWeb
            crag_search_module.CragImageKG = LazyCragImageKG

            image_model_name = (
                config["crag_image_model_name"]
                if "crag_image_model_name" in config and config["crag_image_model_name"] is not None
                else config["retrieval_model_path"]
                if "retrieval_model_path" in config and config["retrieval_model_path"] is not None
                else "openai/clip-vit-large-patch14-336"
            )
            text_model_name = (
                config["crag_text_model_name"]
                if "crag_text_model_name" in config and config["crag_text_model_name"] is not None
                else "BAAI/bge-large-en-v1.5"
            )
            self.search_pipeline = UnifiedSearchPipeline(
                image_model_name=image_model_name,
                image_hf_dataset_id="crag-mm-2025/image-search-index-validation",
                text_model_name=text_model_name,
                web_hf_dataset_id="crag-mm-2025/web-search-index-validation",
            )
        finally:
            torch.cuda.is_available = original_cuda_available
        self.text_retrieval_topk = config["text_retrieval_topk"]
        self.image_retrieval_topk = config["image_retrieval_topk"]
    def build_entity_evidence(self, retrieval_results):
        print(f"Raw retrieval results: {retrieval_results}")
        entity_map = {}

        for item in retrieval_results:
            score = item.get("score", 0.0)
            url = item.get("url", "")
            for ent in item.get("entities", []):
                name = ent.get("entity_name", "Unknown")
                attrs = ent.get("entity_attributes", {})

                if name not in entity_map:
                    entity_map[name] = {
                        "entity_name": name,
                        "score_max": score,
                        "match_count": 0,
                        "urls": [],
                        "attributes": attrs.copy() if attrs else {},
                    }

                entity_map[name]["score_max"] = max(entity_map[name]["score_max"], score)
                entity_map[name]["match_count"] += 1
                if url:
                    entity_map[name]["urls"].append(url)

        return list(entity_map.values())

    def entity_evidence_to_text(self, entity_list):
        chunks = []
        for i, ent in enumerate(entity_list, 1):
            attrs = ent["attributes"]
            lines = [
                f"{i}. {ent['entity_name']}",
                f"- best_score: {ent['score_max']:.4f}",
                f"- match_count: {ent['match_count']}",
            ]
            for k in ["building_name", "location", "country", "status",
                    "religious_affiliation", "architect",
                    "architecture_style", "established", "year_completed"]:
                if k in attrs and attrs[k]:
                    lines.append(f"- {k}: {attrs[k]}")
            chunks.append("\n".join(lines))
        return "\n\n".join(chunks)

    @staticmethod
    def _truncate_text(value, max_chars):
        text = str(value)
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + " ..."

    def image_results_to_text(self, retrieval_results):
        max_attr_chars = int(self.config["crag_image_attr_max_chars"])
        max_entity_chars = int(self.config["crag_image_entity_max_chars"])
        chunks = []
        for i, item in enumerate(retrieval_results, 1):
            lines = [f"{i}."]

            entities = item.get("entities", [])
            if not entities:
                lines.append("- entities: []")
            else:
                for entity_idx, ent in enumerate(entities, 1):
                    lines.append(f"- entity_{entity_idx}_name: {ent.get('entity_name', 'Unknown')}")
                    attrs = ent.get("entity_attributes", {}) or {}
                    for key in sorted(attrs.keys()):
                        value = attrs[key]
                        if value is None or value == "":
                            continue
                        lines.append(f"  {key}: {self._truncate_text(value, max_attr_chars)}")

            entity_text = "\n".join(lines)
            chunks.append(self._truncate_text(entity_text, max_entity_chars))

        return "\n\n".join(chunks)

    def search(self, query, num: int = None, query_type: str = None, target_modal: str = "text") -> List[Dict[str, str]]:
        if query_type is None:
            query_type = "image" if not isinstance(query, str) else "text"
        if num is None:
            num = self.text_retrieval_topk if query_type == "text" else self.image_retrieval_topk
        candidate_k = max(num * 5, num)
        if query_type == 'text':
            results = self.search_pipeline(query, k=candidate_k)
            final_results = [f"{result.get('page_name')}\n{result.get('page_snippet')}" for result in results]
            final_results = self._dedup_topk_results(final_results, num=num)
            return final_results
        elif query_type == 'image':
            results = self.search_pipeline(query, k=candidate_k)
            results = self._dedup_topk_results(results, num=num)
            return results
        else:
            raise NotImplementedError("CRAGRetriever currently only supports text query and image query.")

class MCSearchRetriever(BaseRetriever):
    """Retriever for local MC-Search KB embeddings and metadata."""

    def __init__(self, config):
        super().__init__(config)
        self._load_mcsearch_assets()
        self._load_model()

    def update_additional_setting(self):
        self.batch_size = self._config.get("retrieval_batch_size", 8)
        self.score_threshold = self._config.get("mcsearch_score_threshold", 0.0)
        self.image_search_topk = self._config.get("mcsearch_image_search_topk", 5)
        self.mcsearch_visual_bge_model_name = self._config.get(
            "mcsearch_visual_bge_model_name",
            "BAAI/bge-base-en-v1.5",
        )
        self.retrieval_model_path = self._config.get(
            "mcsearch_visual_bge_model_path",
            self._config.get("retrieval_model_path", None),
        )
        self.mcsearch_data_root = self._config.get(
            "mcsearch_data_root",
            "/home/you/FlashRAG/exps/idea10/data/datasets/mcsearch/kb",
        )
        self.mcsearch_model_weight = self._config.get(
            "mcsearch_model_weight",
            "/home/you/FlashRAG/exps/idea10/data/datasets/mcsearch/Visualized_base_en_v1.5.pth",
        )

    @staticmethod
    def _l2_normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        return mat / norms

    @staticmethod
    def _load_ordered_id_list(mapping_path: str) -> List[str]:
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        if isinstance(mapping, list):
            return mapping
        return [sid for _, sid in sorted(((int(k), v) for k, v in mapping.items()), key=lambda x: x[0])]

    @classmethod
    def _build_kb_index(cls, emb_path: str, mapping_path: str) -> tuple[faiss.IndexFlatIP, List[str]]:
        id_list = cls._load_ordered_id_list(mapping_path)
        mat = np.load(emb_path, mmap_mode="r").astype("float32")
        if mat.ndim != 2:
            raise ValueError(f"Expected 2D embedding matrix in {emb_path}, got shape {mat.shape}")
        if len(id_list) != mat.shape[0]:
            raise ValueError(f"Row count mismatch: embeddings {mat.shape[0]} vs mapping {len(id_list)}")
        mat = cls._l2_normalize(mat)
        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(mat)
        return index, id_list

    @staticmethod
    def _load_mcsearch_docs(all_docs_path: str) -> Dict[str, Dict[str, str]]:
        with open(all_docs_path, "r", encoding="utf-8") as f:
            first = f.read(1)
            f.seek(0)
            if first == "[":
                raw_docs = json.load(f)
            else:
                raw_docs = [json.loads(line) for line in f if line.strip()]

        docs = {}
        for item in raw_docs:
            snippet_id = str(item.get("snippet_id", "")).strip()
            if not snippet_id:
                continue
            title = str(item.get("title", "")).strip()
            fact = str(item.get("fact", "")).strip()
            docs[snippet_id] = {
                "id": snippet_id,
                "title": title,
                "text": fact,
                "contents": fact,
                "url": str(item.get("url", "")).strip(),
            }
        return docs

    @staticmethod
    def _load_image_infos(all_image_infos_path: str) -> Dict[int, Dict[str, str]]:
        with open(all_image_infos_path, "r", encoding="utf-8") as f:
            infos = json.load(f)
        return {
            int(item["image_id"]): {
                "id": str(item["image_id"]),
                "title": str(item.get("title", "")).strip(),
                "text": str(item.get("title", "")).strip(),
                "contents": str(item.get("title", "")).strip(),
                "image_url": str(item.get("imgUrl", "")).strip(),
            }
            for item in infos
            if item.get("image_id") is not None
        }

    def _load_mcsearch_assets(self):
        emb_dir = os.path.join(self.mcsearch_data_root, "knowledge_base_emb")
        self.doc_index, self.doc_sid_list = self._build_kb_index(
            os.path.join(emb_dir, "docs_embeddings.npy"),
            os.path.join(emb_dir, "doc_index2snippet.json"),
        )
        self.doc_sid_lookup = np.array(self.doc_sid_list)

        self.img_index, self.img_id_list = self._build_kb_index(
            os.path.join(emb_dir, "img_embeddings.npy"),
            os.path.join(emb_dir, "img_index2imageid.json"),
        )
        self.img_id_lookup = np.array(self.img_id_list)

        self.cap_index, self.cap_id_list = self._build_kb_index(
            os.path.join(emb_dir, "cap_embeddings.npy"),
            os.path.join(emb_dir, "cap_index2imageid.json"),
        )
        self.cap_id_lookup = np.array(self.cap_id_list)

        self.sid2doc = self._load_mcsearch_docs(os.path.join(self.mcsearch_data_root, "all_docs.json"))
        self.image_info = self._load_image_infos(os.path.join(self.mcsearch_data_root, "all_image_infos.json"))

    def _resolve_mcsearch_device(self):
        if not torch.cuda.is_available():
            return "cpu"
        # GPU selection should be decided once at process startup from config
        # via CUDA_VISIBLE_DEVICES. Inside the process, always use the default
        # logical CUDA device instead of reinterpreting gpu_id again.
        return "cuda"

    def _load_model(self):
        from visual_bge.modeling import Visualized_BGE

        device = self._resolve_mcsearch_device()
        from_pretrained = self.retrieval_model_path
        if from_pretrained is not None and "bge-base-en-v1.5" not in str(from_pretrained):
            warnings.warn(
                "MCSearchRetriever expects a BGE base encoder compatible with "
                "`Visualized_base_en_v1.5.pth`. The configured `mcsearch_visual_bge_model_path` "
                f"is `{from_pretrained}`, which does not look like a base model path. "
                "Falling back to loading tokenizer/config by model name only."
            )
            from_pretrained = None

        self.encoder = Visualized_BGE(
            model_name_bge=self.mcsearch_visual_bge_model_name,
            model_weight=self.mcsearch_model_weight,
            from_pretrained=from_pretrained,
        ).to(device)
        self.encoder.device = torch.device(device)
        print(f"MCSearchRetriever loads Visualized_BGE on device: {device}")
        self.encoder.eval()
        torch.set_grad_enabled(False)

    def _encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        with torch.no_grad():
            emb = self.encoder.encode(text=text)
        emb = emb.cpu().numpy().astype("float32")
        if emb.ndim == 1:
            emb = emb[None, :]
        return self._l2_normalize(emb)

    def _load_query_image(self, query):
        if isinstance(query, Image.Image):
            image_bytes = BytesIO()
            query.convert("RGB").save(image_bytes, format="JPEG")
            image_bytes.seek(0)
            return image_bytes
        if isinstance(query, str):
            if os.path.exists(query):
                return query
            response = requests.get(query, stream=True, timeout=15)
            response.raise_for_status()
            image_bytes = BytesIO(response.content)
            image_bytes.seek(0)
            return image_bytes
        raise TypeError(f"Unsupported image query type: {type(query)}")

    def _encode_image(self, image_query) -> np.ndarray:
        image = self._load_query_image(image_query)
        with torch.no_grad():
            emb = self.encoder.encode(image=image)
        emb = emb.cpu().numpy().astype("float32")
        if emb.ndim == 1:
            emb = emb[None, :]
        return self._l2_normalize(emb)

    def _format_doc_result(self, snippet_id: str) -> Dict[str, str]:
        return dict(self.sid2doc.get(str(snippet_id), {"id": str(snippet_id), "title": "", "text": "", "contents": ""}))

    def _format_image_result(self, image_id: Union[str, int]) -> Dict[str, str]:
        image_id = int(image_id)
        item = dict(self.image_info.get(image_id, {"id": str(image_id), "title": "", "text": "", "contents": "", "image_url": ""}))
        item["image_id"] = image_id
        return item

    def _search_text_index(self, query: str, num: int) -> tuple[List[Dict[str, str]], List[float]]:
        query_emb = self._encode_text(query)
        candidate_k = min(max(num * 5, num), self.doc_index.ntotal)
        scores, idxs = self.doc_index.search(query_emb, k=candidate_k)
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()

        deduped_results = []
        deduped_scores = []
        seen_doc_keys = set()

        for idx, score in zip(idxs, scores):
            if idx < 0:
                continue
            result = self._format_doc_result(self.doc_sid_lookup[idx])
            if not any(str(result.get(field, "")).strip() for field in ("title", "text", "contents")):
                continue
            doc_key = (
                str(result.get("title", "")).strip(),
                str(result.get("text", "")).strip(),
            )
            if doc_key in seen_doc_keys:
                continue
            seen_doc_keys.add(doc_key)
            deduped_results.append(result)
            deduped_scores.append(score)
            if len(deduped_results) >= num:
                break

        return deduped_results, deduped_scores

    def _search_image_by_image(self, query, num: int) -> tuple[List[Dict[str, str]], List[float]]:
        query_emb = self._encode_image(query)
        candidate_k = min(max(num * 5, num), self.img_index.ntotal)
        scores, idxs = self.img_index.search(query_emb, k=candidate_k)
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()
        results = [self._format_image_result(self.img_id_lookup[idx]) for idx in idxs if idx >= 0]
        scores = [score for idx, score in zip(idxs, scores) if idx >= 0]
        results, scores = self._dedup_topk_results(results, scores, num)
        return results, scores

    def _search_image_by_text(self, query: str, num: int) -> tuple[List[Dict[str, str]], List[float]]:
        query_emb = self._encode_text(query)
        k = max(num, self.image_search_topk)
        img_scores, img_idxs = self.img_index.search(query_emb, k=k)
        cap_scores, cap_idxs = self.cap_index.search(query_emb, k=k)

        bucket = defaultdict(list)
        for idx, score in zip(img_idxs[0].tolist(), img_scores[0].tolist()):
            if idx >= 0:
                bucket[int(self.img_id_lookup[idx])].append(float(score))
        for idx, score in zip(cap_idxs[0].tolist(), cap_scores[0].tolist()):
            if idx >= 0:
                bucket[int(self.cap_id_lookup[idx])].append(float(score))

        ranked = sorted(
            ((image_id, sum(scores) / len(scores)) for image_id, scores in bucket.items()),
            key=lambda x: x[1],
            reverse=True,
        )[:num]
        results = [self._format_image_result(image_id) for image_id, _ in ranked]
        scores = [score for _, score in ranked]
        results, scores = self._dedup_topk_results(results, scores, num)
        return results, scores

    def _search(self, query, target_modal: str = "text", num: int = None, return_score=False):
        if target_modal == "text":
            num = self.text_retrieval_topk if num is None else num
            if judge_image(query):
                raise NotImplementedError("MCSearchRetriever does not support image-to-text retrieval.")
            results, scores = self._search_text_index(query, num)
        elif target_modal == "image":
            num = self.image_retrieval_topk if num is None else num
            if judge_image(query):
                results, scores = self._search_image_by_image(query, num)
            else:
                results, scores = self._search_image_by_text(query, num)
        else:
            raise ValueError("target_modal must be `text` or `image`.")

        if self.score_threshold > 0:
            filtered = [(item, score) for item, score in zip(results, scores) if score >= self.score_threshold]
            results = [item for item, _ in filtered]
            scores = [score for _, score in filtered]
        if return_score:
            return results, scores
        return results

    def _batch_search(self, query: List[Union[str, Image.Image]], target_modal: str = "text", num: int = None, return_score=False):
        if isinstance(query, (str, Image.Image)):
            query = [query]
        results = []
        scores = []
        for item in tqdm(query, desc="Retrieval process: ", disable=self.silent):
            item_result, item_score = self._search(item, target_modal=target_modal, num=num, return_score=True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        return results

    def _batch_search(self, query: List[str], target_modal: str = "text", num: int = None, return_score=False):
        if isinstance(query, str):
            query = [query]
        if num is None:
            num = self.text_retrieval_topk if target_modal == "text" else self.image_retrieval_topk
        batch_size = self.batch_size
        assert target_modal in ["image", "text"]

        query_modal = self._judge_input_modal(query[0])
        if query_modal == "image" and isinstance(query[0], str):
            from PIL import Image
            import requests

            if os.path.exists(query[0]):
                query = [Image.open(q) for q in query]
            else:
                query = [Image.open(requests.get(q, stream=True).raw) for q in query]

        results = []
        scores = []

        for start_idx in tqdm(range(0, len(query), batch_size), desc="Retrieval process: ", disable=self.silent):
            query_batch = query[start_idx : start_idx + batch_size]
            batch_emb = self.encoder.encode(query_batch, modal=query_modal)
            batch_scores, batch_idxs = self.index_dict[target_modal].search(batch_emb, k=num)

            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            batch_results = [batch_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))]

            scores.extend(batch_scores)
            results.extend(batch_results)

        if return_score:
            return results, scores
        else:
            return results


def main():
    # Example configuration
    from flashrag.config import Config
    config = Config("/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen2_5_7b/config_stage_crag.yaml")
    config['gpu_id']="1"
    text = "playwright"
    retriever = CRAGRetriever(config)

    # Text search
    batch_results = retriever.search(text, query_type="text")
    print(batch_results)

if __name__ == "__main__":
    main()
