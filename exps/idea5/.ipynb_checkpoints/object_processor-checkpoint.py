import tomllib

class ObjectProcessor:
    def __init__(self, model, processor, retriever):
        self.model = model
        self.processor = processor
        with open('prompts/object_process.toml', 'rb') as f:
            self.prompt = tomllib.load(f)
        self.retriever = retriever

    def object_expansion(self, objects):
        object_retrieval_expansion = {}
        object_feature_expansion = {}
        query_list = []
        for object in objects:
            query_list.append(object['object'])
        print(f"Query List: {query_list}")
        results = self.retriever.batch_search(query_list)[0]
        retrieved_contents = [text['contents'] for text in results]
        retrieved_contents = "\n".join(retrieved_contents)
        
        for object in objects:
            object_retrieval_expansion[object['object']] = retrieved_contents
            object_feature_expansion[object['object']] = object['features']

        return object_feature_expansion, object_retrieval_expansion