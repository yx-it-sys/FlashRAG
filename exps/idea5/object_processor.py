import tomllib
from retriever import retrieve

class ObjectProcessor:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        with open('prompts/object_process.toml', 'rb') as f:
            self.prompt = tomllib.load(f)
        

    def object_expansion(self, objects):
        object_retrieval_expansion = {}
        object_feature_expansion = {}
        for object in objects:
            retrieved_content = retrieve(object['object'], top_k=3)
            object_retrieval_expansion[object['object']] = retrieved_content
            object_feature_expansion[object['object']] = object['features']

        return object_feature_expansion, object_retrieval_expansion