from torch.nn.functional import log_softmax
from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
import torch
from accelerate import Accelerator

def apply_integrated_gradients(model, inputs_embeds, attention_mask, target_token_id, steps=50):
    baseline = torch.zeros_like(inputs_embeds)
    total_grads = torch.zeros_like(inputs_embeds)

    for alpha in torch.linspace(0, 1, steps):
        interpolated = baseline + alpha * (inputs_embeds - baseline)
        with torch.enable_grad():
            interpolated.requires_grad_(True)

            logits = model(inputs_embeds=interpolated, attention_mask=attention_mask).logits[0, -1]
            log_prob = log_softmax(logits, dim=-1)[target_token_id]

            model.zero_grad()
            log_prob.backward()

        if interpolated.grad is not None:
            total_grads += interpolated.grad

    avg_grads = total_grads / steps
    return (inputs_embeds - baseline) * avg_grads

def integrated_gradient_process(model, processor, tokenizer, image, input_prompt, answer):    
    text_prompt = processor.tokenizer.apply_chat_template(
        input_prompt,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=text_prompt,
        images=image,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # 将 answer 编码为 token id，取第一个 token 作为目标 token id
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if len(answer_ids) == 0:
        raise ValueError("Answer could not be tokenized into any token.")
    answer_id = answer_ids[0]

    model.eval()
    model.zero_grad()

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    embeddings = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)

    attributions = apply_integrated_gradients(model, embeddings, attention_mask, answer_id)[0]

    print(f"Attributions: {attributions}")
    return attributions



