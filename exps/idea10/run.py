import os

for k in [
    "http_proxy", "https_proxy",
    "HTTP_PROXY", "HTTPS_PROXY",
    "all_proxy", "ALL_PROXY"
]:
    os.environ.pop(k, None)

from flashrag.config import Config
from flashrag.utils import get_dataset, get_generator, get_retriever
from flashrag.pipeline import OmniSearchPipeline

import base64

def turn_to_url(image_path):
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_image}"

def main():
    config = Config("/home/you/FlashRAG/exps/idea10/configs/config.yaml")
    all_split = get_dataset(config)
    test_data = all_split["validation"]
    print(f"Begin loading generator...")
    generator = get_generator(config)
    print("Finished!")
    print(f"Begin loading retriever...")
    retriever = get_retriever(config)
    print("Finished!")
    pipeline = OmniSearchPipeline(config=config, retriever=retriever, generator=generator)
    output_dataset = pipeline.run(test_data, do_eval=True)

def test():
    config = Config("/home/you/FlashRAG/exps/idea10/configs/config.yaml")
    generator = get_generator(config)
    image_path = "/home/you/FlashRAG/exps/idea10/data/datasets/crag_mm/images/0a1f3aaa-7a4f-489a-954b-312393fcae79.jpg"
    base64_image = turn_to_url(image_path)
    message=[
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What's in this image?"},
          {
            "type": "image_url",
            "image_url": {
                "url": base64_image,
              }
          },
        ],
      }
    ]
    response = generator.generate(message)
    print(response)

    
if __name__ == "__main__":    
    main()

