# Load model directly
from transformers import AutoProcessor, AutoModelForVision2Seq

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")