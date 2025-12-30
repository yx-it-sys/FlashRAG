# Guidance for AI coding agents (Copilot / Chat)

This repository is FlashRAG — a Python toolkit for efficient RAG research. Keep instructions short and concrete.

- Big picture: Python package (see `setup.py`, `flashrag/version.py`). Core modules: retrievers, generators, pipelines (see `flashrag/pipeline/mm_pipeline.py`).
- Install (developer):
  - pip editable: `pip install -e .`
  - to install all optional features: `pip install -e .[full]` or use extras in `setup.py` (retriever/generator/multimodal).
  - Faiss (recommended): use conda (README has CPU/GPU conda commands).

- Quick commands the agent may suggest or run:
  - Build an index: `python -m flashrag.retriever.index_builder --corpus_path <corpus.jsonl> --save_dir <dir> --retrieval_method <e.g. e5>`
  - Run examples: check `examples/quick_start/*.py` and `examples/run_mm/*` for usage patterns.

- Pipeline / code patterns to follow:
  - Config pattern: use `flashrag.config.Config` or `Config(config_file, config_dict={...})`. Dict overrides file values.
  - Factory functions: use `get_retriever(config)` and `get_generator(config)` to wire components.
  - Generator API: `generator.generate(input_prompts, ...)` returns response dicts; extract text and hidden states if needed.
  - Retriever API: `retriever.search(...)` and `retriever.batch_search(...)`.
  - Store predictions with: `dataset.update_output("pred", pred_answer_list)` and write outputs to `config['save_dir']` where examples do so.
  - OmniSearch agent: recognizes control tokens like "Text Retrieval" (request retrieval) and "Final Answer" (stop). Follow existing regex parsing behavior in `flashrag/pipeline/mm_pipeline.py`.

- Engineering rules for edits:
  - Preserve public APIs and existing file structure.
  - Run lint/type checks after changes (repo has ruff settings in `pyproject.toml`). If unsure which command to run, ask the maintainer.
  - Do not add secrets or private API keys. Do not change install instructions without updating `setup.py`/`requirements.txt`.

- Files to consult when making changes:
  - `README.md` (quick-start and faiss instructions)
  - `setup.py` (extras and requirements)
  - `flashrag/pipeline/mm_pipeline.py` (pipeline patterns and agent loop)
  - `flashrag/config/basic_config.yaml` (default config values)
  - `flashrag/version.py` (package version)

If you'd like, I can open a PR with this file or adjust tone/length. Ask before making large refactors.
