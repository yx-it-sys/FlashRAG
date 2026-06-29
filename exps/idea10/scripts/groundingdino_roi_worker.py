import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image
import torch
from torchvision.ops import box_convert

for k in [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
]:
    os.environ.pop(k, None)


def parse_args():
    parser = argparse.ArgumentParser(description="GroundingDINO ROI crop worker")
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--phrase", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--groundingdino-root", default="/home/you/GroundingDINO")
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--expand-ratio", type=float, default=0.08)
    parser.add_argument("--serve", action="store_true")
    return parser.parse_args()


def clamp(value, lower, upper):
    return max(lower, min(value, upper))


def load_runtime(args):
    groundingdino_root = Path(args.groundingdino_root).resolve()
    if str(groundingdino_root) not in sys.path:
        sys.path.insert(0, str(groundingdino_root))

    from groundingdino.util.inference import load_model, load_image, predict

    config_file = args.config_file or str(groundingdino_root / "groundingdino/config/GroundingDINO_SwinT_OGC.py")
    checkpoint_path = args.checkpoint_path or str(groundingdino_root / "weights/groundingdino_swint_ogc.pth")

    model = load_model(config_file, checkpoint_path, device=args.device)
    return {
        "load_image": load_image,
        "predict": predict,
        "model": model,
        "device": args.device,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "expand_ratio": args.expand_ratio,
    }


def run_inference(runtime, image_path, phrase, output_path, json_output):
    load_image = runtime["load_image"]
    predict = runtime["predict"]
    model = runtime["model"]
    device = runtime["device"]
    box_threshold = runtime["box_threshold"]
    text_threshold = runtime["text_threshold"]
    expand_ratio = runtime["expand_ratio"]

    image_source, image_tensor = load_image(image_path)
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=phrase,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size

    result = {
        "input_image": image_path,
        "phrase": phrase,
        "boxes_found": int(len(boxes)),
        "used_fallback": False,
    }

    if len(boxes) == 0:
        pil_image.save(output_path)
        result["used_fallback"] = True
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    scaled_boxes = boxes * torch.tensor([width, height, width, height], dtype=boxes.dtype)
    xyxy_boxes = box_convert(boxes=scaled_boxes, in_fmt="cxcywh", out_fmt="xyxy")

    best_idx = int(torch.argmax(logits).item())
    best_box = xyxy_boxes[best_idx].tolist()
    best_phrase = phrases[best_idx]
    best_score = float(logits[best_idx].item())

    x1, y1, x2, y2 = best_box
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    pad_x = box_w * expand_ratio
    pad_y = box_h * expand_ratio

    crop_box = [
        int(clamp(round(x1 - pad_x), 0, width)),
        int(clamp(round(y1 - pad_y), 0, height)),
        int(clamp(round(x2 + pad_x), 0, width)),
        int(clamp(round(y2 + pad_y), 0, height)),
    ]

    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        pil_image.save(output_path)
        result["used_fallback"] = True
        result["invalid_box"] = crop_box
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    cropped = pil_image.crop(tuple(crop_box))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cropped.save(output_path)

    result.update({
        "selected_index": best_idx,
        "selected_phrase": best_phrase,
        "selected_score": best_score,
        "selected_box_xyxy": [int(round(v)) for v in best_box],
        "crop_box_xyxy": crop_box,
        "crop_size": list(cropped.size),
    })

    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def validate_single_request(args):
    required_fields = {
        "image_path": args.image_path,
        "phrase": args.phrase,
        "output_path": args.output_path,
        "json_output": args.json_output,
    }
    missing = [name for name, value in required_fields.items() if not value]
    if missing:
        raise ValueError(f"Missing required arguments for single-shot mode: {', '.join(missing)}")


def serve_loop(runtime):
    print(json.dumps({"status": "ready"}, ensure_ascii=False), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                print(json.dumps({"status": "shutdown"}, ensure_ascii=False), flush=True)
                break

            result = run_inference(
                runtime,
                image_path=request["image_path"],
                phrase=request["phrase"],
                output_path=request["output_path"],
                json_output=request["json_output"],
            )
            print(json.dumps({"status": "ok", "result": result}, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(
                json.dumps(
                    {"status": "error", "error": f"{exc.__class__.__name__}: {exc}"},
                    ensure_ascii=False,
                ),
                flush=True,
            )


def main():
    args = parse_args()
    runtime = load_runtime(args)

    if args.serve:
        serve_loop(runtime)
        return

    validate_single_request(args)
    result = run_inference(
        runtime,
        image_path=args.image_path,
        phrase=args.phrase,
        output_path=args.output_path,
        json_output=args.json_output,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
