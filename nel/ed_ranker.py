import nel.ntee as ntee
from nel.vocabulary import Vocabulary
from nel.abstract_word_entity import load as load_model
from nel.mulrel_ranker import MulRelRanker
import nel.utils as utils

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from itertools import product
from random import shuffle

from pprint import pprint
import Levenshtein

ModelClass = MulRelRanker
wiki_prefix = 'en.wikipedia.org/wiki/'
n_best = 4
alpha = 0.2  # 0.1
beta = 0.2  # 1
gamma = 0.05  # 0.95


class EDRanker:
    """
    ranking candidates
    """

    def __init__(self, config):
        print('---- create ed rank model ----')
        # maximum 取最大值
        # np.linalg.norm 求矩阵或者向量的范数，axis=1表示按行向量处理，求多个行向量的范数，keepding：是否保持矩阵的二维特性 True表示保持矩阵的二维特性
        config['entity_embeddings'] = config['entity_embeddings'] / \
                                      np.maximum(np.linalg.norm(config['entity_embeddings'],
                                                                axis=1, keepdims=True), 1e-12)
        config['entity_embeddings'][config['entity_voca'].unkid] = 1e-10
        config['word_embeddings'] = config['word_embeddings'] / \
                                    np.maximum(np.linalg.norm(config['word_embeddings'],
                                                              axis=1, keepdims=True), 1e-12)
        config['word_embeddings'][config['word_voca'].unkid] = 1e-10
        self.args = config['args']

        if self.args.mode == 'prerank':
            self.ent_net = config['ent_net']
            print('prerank model')
            self.prerank_model = ntee.NTEE(config)
            self.prerank_model.cuda()

        print('main model')
        if self.args.mode in {'eval','ed'}:
            print('try loading model from',self.args.mode_path)
            self.model = load_model(self.args.mode_path,ModelClass)
        else:
            try:
                print('try loading model from',self.args.mode_path)
                self.model = load_model(self.args.mode_path,ModelClass)
            except:
                print('create new model')
                if config['mulrel_type'] == 'rel-norm':
                    config['use_stargmax'] = False
                if config['mulrel_type'] == 'ment-norm':
                    config['first_head_uniform'] = False
                    config['use_pad_ent'] = True

                config['use_local'] = True
                config['use_local_only'] = False
                config['oracle'] = False
                self.model = ModelClass(config)

            self.model.cuda()