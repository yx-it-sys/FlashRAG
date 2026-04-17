```pseudo
repair_count = 0
action = None
prev_query = None

while action != "final_answer":
    action = LLM_ref(messages)

    if action == "text_retrieval":
        text_query = extract_text_query(messages)
        image_query = extract_image_query(messages)

        # Step 1: pre-retrieval redundancy check
        if prev_query is not None and cos_sim(text_query, prev_query) > t:
            if repair_count < 2:
                thought = rebuild_messages(messages, mode="missing_info")
                messages = form_messages(messages, thought)
                repair_count += 1
                continue
            else:
                thought = rebuild_messages(messages, mode="reflection")
                messages = form_messages(messages, thought)
                action = "reasoning"
                repair_count = 0
                continue

        # Step 2: ambiguity handling
        if AmbiguityCheck(text_query):
            extra_info = ImageRetrieval(ROI(image_query))
            grounded_info = flatten(extra_info)
            retrieval_query = AlignmentLayer(text_query, grounded_info)
        else:
            retrieval_query = text_query

        # Step 3: retrieval
        retrieval_results = TextRetrieval(retrieval_query)
        messages = form_messages(messages, retrieval_results)

        # Step 4: post-retrieval information gain check
        if delta_f(retrieval_histories) < m:
            if repair_count < 2:
                thought = rebuild_messages(messages, mode="missing_info")
                messages = form_messages(messages, thought)
                repair_count += 1
                prev_query = retrieval_query
                continue
            else:
                thought = rebuild_messages(messages, mode="reflection")
                messages = form_messages(messages, thought)
                action = "reasoning"
                repair_count = 0
                continue

        prev_query = retrieval_query
        repair_count = 0
```