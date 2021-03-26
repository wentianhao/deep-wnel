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

    def compute_sent_vecs(self,token_ids,token_offsets,use_sum=False):
        sum_vecs = self.word_embeddings(token_ids,token_offsets)
        if use_sum:
            return sum_vecs

        sum_vecs = F.normalize(sum_vecs)
        sent_vecs = self.linear(sum_vecs)
        return sent_vecs

    def forward(self,token_ids,token_offsets,entity_ids,use_sum=False,return_sent_vecs=False):
        sent_vecs = self.compute_sent_vecs(token_ids,token_offsets,use_sum)
        entity_vecs = self.entity_embeddings(entity_ids)

        # computer scores
        batchsize,dims = sent_vecs.size()
        n_entities = entity_vecs.size(1)
        scores = torch.bmm(entity_vecs,sent_vecs.view(batchsize,dims,1)).view(batchsize,n_entities)

        log_probs = F.log_softmax(scores,dim=1)
        if not return_sent_vecs:
            return log_probs
        else:
            return log_probs,sent_vecs