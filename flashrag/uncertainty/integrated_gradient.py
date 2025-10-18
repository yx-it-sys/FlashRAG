from torch.nn.functional import log_softmax
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attributions(tokens, scores):
    data = {"Token": tokens, "Importance": scores}
    df = pd.DataFrame(data).T
    df.columns = df.iloc[0]
    df = df.drop("Token")
    df = df.astype(float)

    plt.figure(figsize=(20,2))
    sns.heatmap(df, annot=True, fmt=".4f", cmap="viridis", cbar=False)
    plt.title("Token Importances (Attributions)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    # plt.show()
    plt.savefig("token_attributions.png") # 保存为图片
    print("Attribution visualization saved to token_attributions.png")

def apply_integrated_gradients(model, inputs_embeds, attention_mask, target_token_id, steps=10):
    baseline = torch.zeros_like(inputs_embeds)
    total_grads = torch.zeros_like(inputs_embeds)

    for alpha in torch.linspace(0, 1, steps):
        interpolated = baseline + alpha * (inputs_embeds - baseline)
        with torch.enable_grad():
            interpolated.requires_grad_(True)
            interpolated.retain_grad()
            logits = model(inputs_embeds=interpolated, attention_mask=attention_mask).logits[0, -1]
            log_prob = log_softmax(logits, dim=-1)[target_token_id]

            model.zero_grad()
            log_prob.backward()

        if interpolated.grad is not None:
            total_grads += interpolated.grad
        else:
            import warnings
            warnings.warn("interpolated.grad is still None after retain_grad(). Check the computation graph.")
    
    del interpolated, logits, log_prob
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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

    # 计算每个Prompt Token嵌入的2范数
    token_importance_scores = torch.norm(attributions, p=2, dim=1)

    # 计算IG熵
    total_ig_score = torch.sum(token_importance_scores)
    if total_ig_score == 0:
        ig_entropy = 0.0
    else:
        probabilities = token_importance_scores / total_ig_score
        log_probs = torch.xlogy(probabilities, probabilities)

        ig_entropy_tensor = torch.sum(log_probs)
        ig_entropy = ig_entropy_tensor.item()
    

    return ig_entropy


