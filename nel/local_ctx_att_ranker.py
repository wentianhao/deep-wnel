import torch
import torch.nn as nn
import torch.nn.functional as E
import nel.utils as utils
from nel.abstract_word_entity import AbstractWordEntity

class LocalCtxAttRanker(AbstractWordEntity):
    """"
    局部模型 local model with context token attention (from G&H's EMNLP paper)
    """

    def __init__(self,config):
        # 一个保存了固定字典和大小的简单查找表
        config['word_embeddings_class'] = nn.Embedding
        config['entity_embeddings_class'] = nn.Embedding
        super(LocalCtxAttRanker,self).__init__(config)

        self.hid_dims = config['hid_dims']
        self.tok_top_n = config['tok_top_n']
        self.margin = config['margin']

        self.att_mat_diag = nn.Parameter(torch.ones(self.emb_dims))
        self.tok_score_mat_diag = nn.Parameter(torch.ones(self.emb_dims))
        # Dropout用于抑制过拟合 p 表示的是不保留节点数的比例
        self.local_ctx_dr = nn.Dropout(p=0)

        # nn.Linear（in_features，out_features，bias = True）是用于设置网络中的全连接层,全连接层的输入与输出都是二维张量，一般形状为[batch_size, size]
        self.score_combine_linear_1 = nn.Linear(2,self.hid_dims)
        # ReLu 线性整流函数，又称为修正线性单元,输入小于0的值，幅值为0，输入大于0的值则不变
        self.score_combine_act_1 = nn.ReLU()
        self.score_combine_linear_2 = nn.Linear(self.hid_dims,1)