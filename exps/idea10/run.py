import os
import json
from pathlib import Path

for k in [
    "http_proxy", "https_proxy",
    "HTTP_PROXY", "HTTPS_PROXY",
    "all_proxy", "ALL_PROXY"
]:
    os.environ.pop(k, None)

from flashrag.config import Config
from flashrag.utils import get_dataset, get_generator, get_retriever
from flashrag.pipeline import OmniSearchPipeline

def main():
    config = Config("/home/you/FlashRAG/exps/idea10/configs/config.yaml")
    all_split = get_dataset(config)
    test_data = all_split["validation"]
    print(f"Begin loading generator...")
    generator = get_generator(config)
    print("Finished!")
    if "text_retriever_config" in config or "image_retriever_config" in config:
        print("Skip eager retriever loading; OmniSearchPipeline will lazy-load configured retrievers.")
        retriever = None
    else:
        print(f"Begin loading retriever...")
        retriever = get_retriever(config)
        print("Finished!")
    pipeline = OmniSearchPipeline(config=config, retriever=retriever, generator=generator)
    output_dataset = pipeline.run(test_data, do_eval=True)
    
if __name__ == "__main__":    
    main()
