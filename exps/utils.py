import nltk
from typing import List
from transformers import pipeline

def general_generate(messages, model, tokenizer):

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    generated_ids = [
        output_ids[len(input_ids)] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

def agentic_search(intent: str, entity: List[str], constraints: List[str], retriever, extractor, ret_thresh, top_k: int = 5) -> str:
    retrieved_intent_content, intent_content_scores = retriever.search(query=intent, num=top_k, return_score=True)
    intent_docs = []

    for doc, score in zip(retrieved_intent_content, intent_content_scores):
        if score >= ret_thresh:
            intent_docs.append(doc['contents'])
    
    background_knowledge = []
    for entity in entity:
        retrieved_docs = []
        retrieved_entity_content, entiity_content_scores = retriever.search(query=entity, num=top_k, return_score=True)
        for doc, score in zip(retrieved_entity_content, entiity_content_scores):
            if score >= ret_thresh:
                retrieved_docs.append(doc['contents'])
        background_knowledge.append({entity: retrieved_docs})
    
    # # Rescaling
    # # Trim intent_docs
    # sentences = nltk.sent_tokenize("\n".join(intent_docs))
    # relevant_sentences = []
    # for sent in sentences:
    #     result = extractor(question=intent, context=sent)
    #     if result['score'] > 0.5:
    #         relevant_sentences.append(sent)
    # intent_docs = relevant_sentences    
    # # Remove duplicates from background_knowledge and trim each entity's docs
    intent_docs = "\n".join(intent_docs)
    sections = ["=== Background Knowledge for Entities ==="]
        
    for item in background_knowledge:
        for entity, docs in item.items():
            if not docs:
                continue
            sections.append(f"\n## Introduction to: {entity}")
            
            for i, doc in enumerate(docs, 1):
                clean_doc = doc.strip().replace('\n', ' ')
                sections.append(f"{i}. {clean_doc}")
        
    background_knowledge = "\n".join(sections)

    final_context = f"=== Intent Related Information ===\n{intent_docs}\n\n{background_knowledge}"

    # LLM 登场
    # 首先根据constraints对final_context进行过滤和调整
    # 最终生成当前Plan的结果
    # 内部抛出异常时，外部捕获