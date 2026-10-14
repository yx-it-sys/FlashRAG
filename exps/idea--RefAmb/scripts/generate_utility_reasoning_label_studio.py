#!/usr/bin/env python3
import html
import json
import random
from hashlib import sha1
from pathlib import Path


SOURCE_CONFIGS = [
    {
        "source": "mcsearch",
        "cache_path": Path(
            "/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B/RefAmb_2026_05_01_11_26_refamb_mcsearch_stage/first_round_disturb_rewrite_static_prefix/trajectory_quality_eval_whole_delta_F/utility_cache.json"
        ),
    },
    {
        "source": "oven",
        "cache_path": Path(
            "/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B/RefAmb_2026_05_01_13_28_refamb_oven_stage/first_round_disturb_rewrite_static_prefix/trajectory_quality_eval_whole_delta_F/utility_cache.json"
        ),
    },
    {
        "source": "infoseek",
        "cache_path": Path(
            "/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B/RefAmb_2026_05_02_13_50_refamb_infoseek_stage/first_round_disturb_rewrite_static_prefix/trajectory_quality_eval_whole_delta_F/utility_cache.json"
        ),
    },
]

OUTPUT_DIR = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B/label_studio/utility_reasoning_sample_120"
)
OUTPUT_JSON = OUTPUT_DIR / "utility_reasoning_sample_120.json"
OUTPUT_XML = OUTPUT_DIR / "utility_reasoning_sample_120_config.xml"
OUTPUT_MANIFEST = OUTPUT_DIR / "utility_reasoning_sample_120_manifest.json"

SAMPLE_SIZE = 120
SEED = 20260506

LABEL_CHOICES = [
    ("no_contribution", "No Contribution"),
    ("partial_contribution", "Partial Contribution"),
    ("full_contribution", "Full Contribution"),
]


def load_items() -> list[dict]:
    items = []
    for cfg in SOURCE_CONFIGS:
        data = json.loads(cfg["cache_path"].read_text(encoding="utf-8"))
        for cache_key, cache_value in data.items():
            cache_input = json.loads(cache_key)
            uid = sha1(f"{cfg['source']}::{cache_key}".encode("utf-8")).hexdigest()[:16]
            items.append(
                {
                    "task_id": uid,
                    "source": cfg["source"],
                    "cache_path": str(cfg["cache_path"]),
                    "cache_key": cache_key,
                    "query": cache_input.get("query", ""),
                    "reasoning": cache_input.get("reasoning", ""),
                    "facts": cache_input.get("facts", []),
                    "llm_label": cache_value.get("label"),
                    "llm_reason": cache_value.get("reason", ""),
                    "llm_filtered_facts": cache_value.get("filtered_facts", []),
                }
            )
    return items


def build_reasoning_html(item: dict) -> str:
    reasoning = (item.get("reasoning") or "").strip()
    if not reasoning:
        reasoning = "[empty]"
    reasoning_html = html.escape(reasoning).replace("\n", "<br/>")
    return (
        "<div style='font-size:14px; line-height:1.5;'>"
        "<h3>Model Reasoning</h3>"
        f"<div>{reasoning_html}</div>"
        "</div>"
    )


def build_tasks(sampled_items: list[dict]) -> list[dict]:
    tasks = []
    for item in sampled_items:
        tasks.append(
            {
                "data": {
                    "task_id": item["task_id"],
                    "source": item["source"],
                    "reasoning_html": build_reasoning_html(item),
                    "llm_label": item["llm_label"],
                    "llm_reason": item["llm_reason"],
                    "query": item["query"],
                    "cache_path": item["cache_path"],
                    "cache_key": item["cache_key"],
                }
            }
        )
    return tasks


def build_xml() -> str:
    lines = [
        "<View>",
        '  <Header value="Utility Annotation" />',
        '  <HyperText name="reasoning_html" value="$reasoning_html" />',
        '  <Text name="utility_prompt" value="Based only on the model reasoning above, how useful was this iteration?" />',
        '  <Choices name="human_utility_label" toName="reasoning_html" choice="single" showInline="true">',
    ]
    for value, alias in LABEL_CHOICES:
        lines.append(f'    <Choice value="{value}" alias="{alias}" />')
    lines.extend(
        [
            "  </Choices>",
            '  <TextArea name="annotator_comment" toName="reasoning_html" '
            'placeholder="Optional comment" editable="true" />',
            "</View>",
        ]
    )
    return "\n".join(lines) + "\n"


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    items = load_items()
    if len(items) < SAMPLE_SIZE:
        raise ValueError(f"Only {len(items)} items available, fewer than sample size {SAMPLE_SIZE}.")

    rng = random.Random(SEED)
    sampled_items = rng.sample(items, SAMPLE_SIZE)
    sampled_items.sort(key=lambda item: (item["source"], item["task_id"]))

    write_json(OUTPUT_JSON, build_tasks(sampled_items))
    OUTPUT_XML.write_text(build_xml(), encoding="utf-8")
    write_json(
        OUTPUT_MANIFEST,
        {
            "sample_size": SAMPLE_SIZE,
            "seed": SEED,
            "sources": {cfg["source"]: str(cfg["cache_path"]) for cfg in SOURCE_CONFIGS},
            "label_choices": [value for value, _ in LABEL_CHOICES],
            "sampled_items": sampled_items,
        },
    )
    print(f"Saved {OUTPUT_JSON}")
    print(f"Saved {OUTPUT_XML}")
    print(f"Saved {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()
