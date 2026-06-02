from flashrag.evaluator import Evaluator
from flashrag.dataset.utils import split_dataset, merge_dataset
from flashrag.utils import get_retriever, get_generator, get_refiner, get_judger
from flashrag.prompt import PromptTemplate


class BasicPipeline:
    """Base object of all pipelines. A pipeline includes the overall process of RAG.
    If you want to implement a pipeline, you should inherit this class.
    """

    def __init__(self, config, prompt_template=None):
        self.config = config
        self.device = config["device"]
        self.retriever = None
        self.evaluator = Evaluator(config)
        self.save_retrieval_cache = config["save_retrieval_cache"]
        if prompt_template is None:
            prompt_template = PromptTemplate(config)
        self.prompt_template = prompt_template

    def run(self, dataset):
        """The overall inference process of a RAG framework."""
        pass

    def _normalize_generation_stats(self, stats, sample_count):
        if not stats or sample_count <= 0:
            return None

        if isinstance(stats, dict):
            summary = dict(stats)
        elif isinstance(stats, list):
            total_samples = 0
            total_input_tokens = 0.0
            total_output_tokens = 0.0
            total_tokens = 0.0
            total_latency = 0.0

            for item in stats:
                if not isinstance(item, dict):
                    continue
                batch_size = int(item.get("batch_size", item.get("total_samples", 1)) or 1)
                total_samples += batch_size
                total_input_tokens += float(item.get("input_tokens", 0.0))
                total_output_tokens += float(item.get("output_tokens", 0.0))
                total_tokens += float(item.get("total_tokens", total_input_tokens + total_output_tokens))
                total_latency += float(item.get("latency_seconds", item.get("total_latency_seconds", 0.0)))

            if total_samples <= 0:
                total_samples = len(stats)
            summary = {
                "total_samples": int(total_samples),
                "total_input_tokens": int(total_input_tokens),
                "total_output_tokens": int(total_output_tokens),
                "total_tokens": int(total_tokens),
                "total_latency_seconds": float(total_latency),
            }
        else:
            return None

        total_samples = int(summary.get("total_samples", sample_count) or sample_count)
        total_input_tokens = float(summary.get("total_input_tokens", summary.get("input_tokens", 0.0)))
        total_output_tokens = float(summary.get("total_output_tokens", summary.get("output_tokens", 0.0)))
        total_tokens = float(summary.get("total_tokens", total_input_tokens + total_output_tokens))
        total_latency = float(summary.get("total_latency_seconds", summary.get("latency_seconds", 0.0)))

        compact_summary = {
            "total_samples": total_samples,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "latency_seconds": total_latency,
            "avg_input_tokens_per_sample": (total_input_tokens / total_samples) if total_samples > 0 else 0.0,
            "avg_output_tokens_per_sample": (total_output_tokens / total_samples) if total_samples > 0 else 0.0,
            "avg_total_tokens_per_sample": (total_tokens / total_samples) if total_samples > 0 else 0.0,
            "avg_latency_seconds_per_sample": (total_latency / total_samples) if total_samples > 0 else 0.0,
            "input_tokens_per_second": (total_input_tokens / total_latency) if total_latency > 0 else 0.0,
            "output_tokens_per_second": (total_output_tokens / total_latency) if total_latency > 0 else 0.0,
            "total_tokens_per_second": (total_tokens / total_latency) if total_latency > 0 else 0.0,
        }
        return [compact_summary for _ in range(sample_count)]

    def _attach_generation_stats(self, dataset):
        generator = getattr(self, "generator", None)
        if generator is None:
            return

        stats = None
        for method_name in ("get_generation_stats", "get_performance_stats", "cost_stats"):
            if not hasattr(generator, method_name):
                continue
            method = getattr(generator, method_name)
            try:
                stats = method(reset=True)
            except TypeError:
                stats = method()
            except Exception:
                stats = None
            if stats:
                break

        normalized_stats = self._normalize_generation_stats(stats, len(dataset))
        if normalized_stats is not None:
            dataset.update_output("generation_stats", normalized_stats)

    def evaluate(self, dataset, do_eval=True, pred_process_fun=None):
        """The evaluation process after finishing overall generation"""

        if pred_process_fun is not None:
            dataset = pred_process_fun(dataset)

        self._attach_generation_stats(dataset)

        if do_eval:
            # evaluate & save result
            eval_result = self.evaluator.evaluate(dataset)
            print(eval_result)

        # save retrieval cache
        if self.save_retrieval_cache:
            self.retriever._save_cache()

        return dataset


class SequentialPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        """
        inference stage:
            query -> pre-retrieval -> retriever -> post-retrieval -> generator
        """

        super().__init__(config, prompt_template)
        if generator is None:
            self.generator = get_generator(config)
        else:
            self.generator = generator

        if retriever is None:
            self.retriever = get_retriever(config)
        else:
            self.retriever = retriever

        # TODO: add rewriter module

        self.use_fid = config["use_fid"]

        if config["refiner_name"] is not None:
            self.refiner = get_refiner(config, self.retriever, self.generator)
        else:
            self.refiner = None

    def naive_run(self, dataset, do_eval=True, pred_process_fun=None):
        # direct generation without RAG
        input_prompts = [self.prompt_template.get_string(question=q) for q in dataset.question]
        dataset.update_output("prompt", input_prompts)

        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        input_query = dataset.question
        retrieval_results = self.retriever.batch_search(input_query)
        dataset.update_output("retrieval_result", retrieval_results)

        if self.refiner:
            input_prompt_flag = self.refiner.input_prompt_flag
            if "llmlingua" in self.refiner.name and input_prompt_flag:
                # input prompt
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(dataset.question, dataset.retrieval_result)
                ]
                dataset.update_output("prompt", input_prompts)
                input_prompts = self.refiner.batch_run(dataset)
            else:
                # input retrieval docs
                refine_results = self.refiner.batch_run(dataset)
                dataset.update_output("refine_result", refine_results)
                input_prompts = [
                    self.prompt_template.get_string(question=q, formatted_reference=r)
                    for q, r in zip(dataset.question, refine_results)
                ]

        else:
            if not self.use_fid:
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(dataset.question, dataset.retrieval_result)
                ]

        if self.use_fid:
            print("Use FiD generation")
            input_prompts = []
            for item in dataset:
                q = item.question
                docs = item.retrieval_result
                input_prompts.append([q + " " + doc['contents'] for doc in docs])
        dataset.update_output("prompt", input_prompts)

        # delete used refiner to release memory
        if self.refiner:
            del self.refiner
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset


class ConditionalPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        """
        inference stage:
            query -> judger -> sequential pipeline or naive generate
        """

        super().__init__(config, prompt_template)

        self.judger = get_judger(config)
        if generator is None:
            self.generator = get_generator(config)
        if retriever is None:
            self.retriever = get_retriever(config)
        self.generator = generator
        self.retriever = retriever

        self.sequential_pipeline = SequentialPipeline(
            config, prompt_template, retriever=self.retriever, generator=self.generator
        )

        self.zero_shot_templete = PromptTemplate(
            config=config,
            system_prompt="Answer the question based on your own knowledge. \
                            Only give me the answer and do not output any other words.",
            user_prompt="Question: {question}",
        )

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        # judge_result: list of bool element, representing whether to use retrieval
        judge_result = self.judger.judge(dataset)
        dataset.update_output("judge_result", judge_result)

        # split dataset based on judge_result
        dataset_split = split_dataset(dataset, judge_result)
        pos_dataset, neg_dataset = dataset_split[True], dataset_split[False]

        pos_dataset = self.sequential_pipeline.run(pos_dataset, do_eval=False)
        self.sequential_pipeline.prompt_template = self.zero_shot_templete
        neg_dataset = self.sequential_pipeline.naive_run(neg_dataset, do_eval=False)

        # merge datasets into original format
        dataset = merge_dataset(dataset_split, judge_result)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset


class AdaptivePipeline(BasicPipeline):
    def __init__(
        self,
        config,
        norag_template=None,
        single_hop_prompt_template=None,
        multi_hop_prompt_template=None,
        retriever = None,
        generator = None
    ):
        super().__init__(config)
        # load adaptive classifier as judger
        self.judger = get_judger(config)

        if generator is None:
            generator = get_generator(config)
        if retriever is None:
            retriever = get_retriever(config)

        self.generator = generator
        self.retriever = retriever

        # Load three pipeline for three types of query: naive/single-hop/multi-hop
        from flashrag.pipeline import IRCOTPipeline

        if norag_template is None:
            norag_templete = PromptTemplate(
                config=config,
                system_prompt="Answer the question based on your own knowledge. Only give me the answer and do not output any other words.",
                user_prompt="Question: {question}",
            )
        self.norag_pipeline = SequentialPipeline(
            config,
            prompt_template=norag_templete,
            retriever=retriever,
            generator=generator,
        )

        self.single_hop_pipeline = SequentialPipeline(
            config,
            prompt_template=single_hop_prompt_template,
            retriever=retriever,
            generator=generator,
        )

        self.multi_hop_pipeline = IRCOTPipeline(
            config, prompt_template=multi_hop_prompt_template, retriever=retriever, generator=generator, max_iter=5
        )

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        # judge_result: choice result representing which pipeline to use(e.g. A, B, C)
        judge_result = self.judger.judge(dataset)
        dataset.update_output("judge_result", judge_result)

        # split dataset based on judge_result
        dataset_split = split_dataset(dataset, judge_result)
        for symbol, symbol_dataset in dataset_split.items():
            if symbol == "A":
                symbol_dataset = self.norag_pipeline.naive_run(symbol_dataset, do_eval=False)
            elif symbol == "B":
                symbol_dataset = self.single_hop_pipeline.run(symbol_dataset, do_eval=False)
            elif symbol == "C":
                symbol_dataset = self.multi_hop_pipeline.run(symbol_dataset, do_eval=False)
            else:
                assert False, "Unknown symbol!"

        # merge datasets into original format
        dataset = merge_dataset(dataset_split, judge_result)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset
