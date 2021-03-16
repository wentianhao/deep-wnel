import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nel.utils as utils
from nel.abstract_word_entity import AbstractWordEntity

class NTEE(AbstractWordEntity):
    """
    NTEE model,proposed in Yamada et al. "Learning Distributed Representations of Texts and Entities from Knowledge Base"
    """

    def __init__(self,config):
        config['word_embeddings_class'] = nn.EmbeddingBag
        config['entity_embeddings_class'] = nn.Embedding
        super(NTEE,self).__init__(config)
        self.linear = nn.Linear(self.emb_dims,self.emb_dims)

