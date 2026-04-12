import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image
import torch
from torchvision.ops import box_convert


def parse_args():
    parser = argparse.ArgumentParser(description="GroundingDINO ROI crop worker")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--phrase", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--groundingdino-root", default="/home/you/GroundingDINO")
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--expand-ratio", type=float, default=0.08)
    return parser.parse_args()


def clamp(value, lower, upper):
    return max(lower, min(value, upper))


def main():
    args = parse_args()

    groundingdino_root = Path(args.groundingdino_root).resolve()
    if str(groundingdino_root) not in sys.path:
        sys.path.insert(0, str(groundingdino_root))

    from groundingdino.util.inference import load_model, load_image, predict

    config_file = args.config_file or str(groundingdino_root / "groundingdino/config/GroundingDINO_SwinT_OGC.py")
    checkpoint_path = args.checkpoint_path or str(groundingdino_root / "weights/groundingdino_swint_ogc.pth")

    image_source, image_tensor = load_image(args.image_path)
    model = load_model(config_file, checkpoint_path, device=args.device)
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=args.phrase,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
    )

    pil_image = Image.open(args.image_path).convert("RGB")
    width, height = pil_image.size

    result = {
        "input_image": args.image_path,
        "phrase": args.phrase,
        "boxes_found": int(len(boxes)),
        "used_fallback": False,
    }

    if len(boxes) == 0:
        pil_image.save(args.output_path)
        result["used_fallback"] = True
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(json.dumps(result, ensure_ascii=False))
        return

    scaled_boxes = boxes * torch.tensor([width, height, width, height], dtype=boxes.dtype)
    xyxy_boxes = box_convert(boxes=scaled_boxes, in_fmt="cxcywh", out_fmt="xyxy")

    best_idx = int(torch.argmax(logits).item())
    best_box = xyxy_boxes[best_idx].tolist()
    best_phrase = phrases[best_idx]
    best_score = float(logits[best_idx].item())

    x1, y1, x2, y2 = best_box
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    pad_x = box_w * args.expand_ratio
    pad_y = box_h * args.expand_ratio

    crop_box = [
        int(clamp(round(x1 - pad_x), 0, width)),
        int(clamp(round(y1 - pad_y), 0, height)),
        int(clamp(round(x2 + pad_x), 0, width)),
        int(clamp(round(y2 + pad_y), 0, height)),
    ]

    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        pil_image.save(args.output_path)
        result["used_fallback"] = True
        result["invalid_box"] = crop_box
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(json.dumps(result, ensure_ascii=False))
        return

    cropped = pil_image.crop(tuple(crop_box))
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    cropped.save(args.output_path)

    result.update({
        "selected_index": best_idx,
        "selected_phrase": best_phrase,
        "selected_score": best_score,
        "selected_box_xyxy": [int(round(v)) for v in best_box],
        "crop_box_xyxy": crop_box,
        "crop_size": list(cropped.size),
    })

    with open(args.json_output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
