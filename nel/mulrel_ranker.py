import torch
import torch.nn.functional as E
from torch.autograd import Variable
import nel.utils as utils

from nel.local_ctx_att_ranker import LocalCtxAttRanker
import numpy as np


class MulRelRanker(LocalCtxAttRanker):
    """
    多关系全局模型
    multi-relational global model with context token attention, using loopy belief propagation (LBP) or star
    """

    def __init__(self,config):
        super(MulRelRanker, self).__init__(config)
        self.inference = config.get('inference','LBP') # LBP or star
        self.df = config.get('df',0.5) # damping factor  用来调节收敛设置的参数
        self.n_loops = config['n_loops',10]
        self.ent_top_n = config['ent_top_n',6]

        self.dr = config['dr']
        self.ew_hid_dims = self.emb_dims

        self.max_dist = 1000

        self.oracle = config.get('oracle',False)
        self.ent_ent_comp = config.get('ent_ent_comp','bilinear') # bilinear,trans_e,fbilinear
        self.ctx_comp = config.get('ctx_comp','bow')  # bow or rnn

        self.mode = config.get('mulrel_type','ment_norm') # ment_norm , rel_norm
        self.n_rels = config['n_rels']
        self.uniform_att = config['uniform_att']

        # options for ment_norm
        self.first_head_uniform = config.get('first_head_uniform',False)
        self.use_pad_ent = config.get('use_pad_ent',False)

        #options for rel_norm
        self.use_stargmax = config.get('use_stargmax',False)

        self.use_local = config.get('use_local',False)
        self.use_local_only = config.get('use_local_only',False)
        self.freeze_local = config.get('freeze_local',False)

        # if using multi instance learning
        self.n_negs = config.get('n_negs',0)

        if self.freeze_local:
            self.att_mat_diag.requires_grad = False
            self.tok_score_mat_diag.requires_grad = False

        if self.use_local:
            self.ent_localctx_comp = torch.nn.Parameter(torch.ones(self.emb_dims))

        if self.use_pad_ent:
            self.pad_ent_emb = torch.nn.Parameter(torch.randn(1,self.emb_dims) * 0.1)
            self.pad_ctx_vec = torch.nn.Parameter(torch.randn(1,self.emb_dims) * 0.1)
        # 顺序容器，模块按照在构造函数种传递的顺序添加
        self.ctx_layer = torch.nn.Sequential(
            # nn.Linear（in_features，out_features，bias = True）是用于设置网络中的全连接层,全连接层的输入与输出都是二维张量，一般形状为[batch_size, size]
            torch.nn.Linear(self.emb_dims * 3,self.ew_hid_dims),
            torch.nn.Tanh(),
            # 以概率p随机将输入张量的一些元素归零
            torch.nn.Dropout(p=self.dr))

        self.rel_embs = torch.randn(self.n_rels,self.emb_dims) * 0.01
        if self.ent_ent_comp == 'bilinear':
            self.rel_embs[0] = 1 + torch.randn(self.emb_dims) * 0.01
            if self.mode == 'ment-norm' and self.n_rels > 1 and self.first_head_uniform:
                self.rel_embs[1] = 1
            if self.mode == 'rel_norm':
                self.rel_embs.fill_(0).add_(torch.randn(self.n_rels,self.emb_dims) * 0.1)

        self.rel_embs = torch.nn.Parameter(self.rel_embs)
        self.ew_embs = torch.nn.Parameter(torch.randn(self.n_rels,self.ew_hid_dims)*
                                          (0.01 if self.mode == 'ment-norm' else 0.1))

        self._coh_ctx_vecs = None

        self.score_combine = torch.nn.Sequential(
            torch.nn.Linear(2,self.hid_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hid_dims,1))

        print('--------------- mode config ------------------')
        for k,v in self.__dict__.items():
            """
            hasattr(object, name)
            object -- 对象。
            name -- 字符串，属性名。
            return
            如果对象有该属性返回 True，否则返回 False
            """
            if not hasattr(v,'__dict__'):
                print(k,v)
        print('-----------------------------------------------')