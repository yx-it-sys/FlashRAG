import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from transformers import CLIPModel, CLIPProcessor, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def get_cross_attention(clip_model, clip_processor, device, target_text, image):
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    inputs = clip_processor(text=[target_text], images=image, return_tensor="pt").to(device)
    with torch.no_grad():
        outputs = clip_model.vision_model(**inputs, output_attention=True)
    last_attn = outputs.attentions[-1]

    attn_map = last_attn.mean(dim=1)[0,0,1:]

    grid_size = int(np.sqrt(attn_map.shape[0]))
    heatmap = attn_map.reshape(grid_size, grid_size).cpu().numpy()

    heatmap_resized = cv2.resize(heatmap, (image.width, image.height), interpolation=cv2.INTER_CUBIC)

    norm_heatmap = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
    norm_heatmap = (norm_heatmap * 255).astype(np.uint8)

    blur = cv2.GaussianBlur(norm_heatmap, (15, 15), 0)
    _, thresh = cv2.threshold(blur, int(255 * 0.7), 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        marked_cv_img = cv_img.copy()
        cv2.rectangle(marked_cv_img, (x, y), (x+w, y+h), (0,0,255), 3)

        marked_image = Image.fromarray(cv2.cvtColor(marked_cv_img, cv2.COLOR_BGR2RGB))

        marked_image.save("marked_for_qwen.jpg")

    else:
        print("未检测到显著的高亮区域，使用原图。")
        marked_pil_image = image
