from torch.nn.functional import log_softmax
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy

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
    plt.savefig("token_attributions.png")
    print("Attribution visualization saved to token_attributions.png")

def apply_integrated_gradients(model, inputs_embeds, attention_mask, labels, steps=10):
    baseline = torch.zeros_like(inputs_embeds)
    total_grads = torch.zeros_like(inputs_embeds)

    for alpha in torch.linspace(0, 1, steps):
        interpolated = baseline + alpha * (inputs_embeds - baseline)
        with torch.enable_grad():
            interpolated.requires_grad_(True)
            interpolated.retain_grad()
            outputs = model(inputs_embeds=interpolated, attention_mask=attention_mask, labels=labels)
            # logits = outputs.logits[0, -1]
            # log_prob = log_softmax(logits, dim=-1)[target_token_id]
            # 负对数似然
            loss = outputs.loss
            target_scalar = -loss
            model.zero_grad()
            target_scalar.backward()


        if interpolated.grad is not None:
            total_grads += interpolated.grad
        else:
            import warnings
            warnings.warn("interpolated.grad is still None after retain_grad(). Check the computation graph.")
    
    del interpolated, outputs, loss, target_scalar

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    avg_grads = total_grads / steps
    return (inputs_embeds - baseline) * avg_grads

def find_subsequence(main_seq, sub_seq):
    """在主序列中查找子序列的位置。"""
    len_sub = len(sub_seq)
    if len_sub == 0: return -1
    for i in range(len(main_seq) - len_sub + 1):
        if main_seq[i:i+len_sub] == sub_seq:
            return i
    return -1

_markers_added = False
def setup_tokenizer_for_markers(tokenizer):
    global _markers_added
    if not _markers_added:
        new_tokens = ["<question_start>", "<question_end>"]
        tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
        print(f"Added special tokens: {new_tokens}")
        _markers_added = True


def integrated_gradient_process(model, processor, tokenizer, input_prompt, answer):    
    setup_tokenizer_for_markers(tokenizer)
    START_MARKER = "<question_start>"
    END_MARKER = "<question_end>"

    question = input_prompt['question']
    
    marked_input_prompt = copy.deepcopy(input_prompt['input_prompt'])
    found_and_marked = False
    
    for message in marked_input_prompt:
        if 'content' in message:
            # 处理 {"role": "system", "content": "..."} 格式
            if isinstance(message['content'], str) and question in message['content']:
                message['content'] = message['content'].replace(question, f"{START_MARKER} {question} {END_MARKER}", 1)
                found_and_marked = True
            elif isinstance(message['content'], list):
                for item in message['content']:
                    if item.get('type') == 'text' and question in item.get('text', ''):
                        item['text'] = item['text'].replace(question, f"{START_MARKER} {question} {END_MARKER}", 1)
                        found_and_marked = True
                        break

    if found_and_marked:
        print("Successfully marked the question in the prompt dict.")
    else:
        print("Warning: Could not find and mark the question in the prompt dict.")

    text_prompt = tokenizer.apply_chat_template(
        marked_input_prompt if found_and_marked else input_prompt['input_prompt'],
        tokenize=False,
        add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(text_prompt)

    start_marker_id = tokenizer.convert_tokens_to_ids(START_MARKER)
    end_marker_id = tokenizer.convert_tokens_to_ids(END_MARKER)

    try:
        start_pos = prompt_ids.index(start_marker_id)
        end_pos = prompt_ids.index(end_marker_id)
        
        question_start_index = start_pos + 1
        question_end_index = end_pos
        print(f"Successfully located question tokens using markers from index {question_start_index} to {question_end_index}.")
    except ValueError:
        print("Warning: Markers not found after tokenization. Calculating entropy over the entire prompt.")
        question_start_index = 0
        question_end_index = len(prompt_ids)
    
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not answer_ids: return 0.0
    
    final_prompt_ids = prompt_ids
    input_ids = torch.tensor([final_prompt_ids + answer_ids], device=model.device)
    prompt_len = len(final_prompt_ids)
    labels = torch.tensor([[-100]*prompt_len + answer_ids], device=model.device)
    attention_mask = torch.ones_like(input_ids)
    
    model.eval()
    model.zero_grad()
    embeddings = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
    attributions = apply_integrated_gradients(model, embeddings, attention_mask, labels)[0]
        
    question_attributions = attributions[question_start_index:question_end_index]
    token_importance_scores = torch.norm(question_attributions, p=2, dim=1)

    total_ig_score = torch.sum(token_importance_scores)
    if total_ig_score <= 1e-9:
        ig_entropy = 0.0
    else:
        probabilities = token_importance_scores / total_ig_score
        log_probs = torch.xlogy(probabilities, probabilities)
        ig_entropy = -torch.sum(log_probs).item()
    
    return ig_entropy