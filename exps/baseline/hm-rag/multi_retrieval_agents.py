from retrieval.vector_retrieval import VectorRetrieval
from retrieval.graph_retrieval import GraphRetrieval
from retrieval.web_retrieval import WebRetrieval
from summary_agent import SummaryAgent
from decompose_agent import DecomposeAgent

from typing import List
import json

class MRetrivalAgent():
    def __init__(self, config):
        self.config = config
        self.vector_retrieval = VectorRetrieval(config)
        self.graph_retrieval = GraphRetrieval(config)
        self.web_retrieval = WebRetrieval(config)
        self.sum_agent = SummaryAgent(config)
        self.dec_agent = DecomposeAgent(config)
        
