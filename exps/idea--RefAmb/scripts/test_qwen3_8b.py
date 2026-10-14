#!/usr/bin/env python3
import os

from vllm import LLM, SamplingParams


MODEL_NAME = os.getenv("QWEN3_VL_MODEL_PATH", "/mnt/data/you/modelscope/Qwen-3-VL-8B")
GPU_MEMORY_UTILIZATION = 0.95
MAX_MODEL_LEN = int(os.getenv("QWEN3_VL_MAX_MODEL_LEN", "12000"))


def main() -> None:
    llm = LLM(
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                    },
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    sampling_params = SamplingParams(
        temperature=0.2,
        max_tokens=128,
    )

    outputs = llm.chat(messages=messages, sampling_params=sampling_params)
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
