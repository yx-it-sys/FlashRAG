import torch
from torch.nn.functional import log_softmax
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForVision2Seq
from accelerate import Accelerator

def load_vlm_model_and_processor(model_name):
    """
    Load a vision-language model and processor with accelerator support.
    """
    accelerator = Accelerator()
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    model.eval()
    model, processor = accelerator.prepare(model, processor)
    return model, processor
