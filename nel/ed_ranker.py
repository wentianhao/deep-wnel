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
        config['entity_embeddings'][config['entity_voca'].unk_id] = 1e-10
        config['word_embeddings'] = config['word_embeddings'] / \
                                    np.maximum(np.linalg.norm(config['word_embeddings'],
                                                              axis=1, keepdims=True), 1e-12)
        config['word_embeddings'][config['word_voca'].unk_id] = 1e-10
        self.args = config['args']

        if self.args.mode == 'prerank':
            # self.ent_net = config['ent_net']
            print('prerank model')
            self.prerank_model = ntee.NTEE(config)
            # self.prerank_model.cuda()

        print('main model')
        if self.args.mode in {'eval', 'ed'}:
            print('try loading model from', self.args.mode_path)
            self.model = load_model(self.args.mode_path, ModelClass)
        else:
            try:
                print('try loading model from', self.args.mode_path)
                self.model = load_model(self.args.mode_path, ModelClass)
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

            # self.model.cuda()

    def prerank(self, dataset, predict=False):
        new_dataset = []
        has_gold = 0
        total = 0
        correct = 0
        larger_than_x = 0
        larger_than_x_correct = 0
        total_cands = 0

        print('preranking...')

        for count, content in enumerate(dataset):
            if count % 1000 == 0:
                print(count, end='\r')

            items = []

            if self.args.keep_ctx_ent > 0:
                # rank the candidates by ntee scores
                lctx_ids = [m['context'][0][max(len(m['contenxt'][0]) - self.args.prerank_ctx_window // 2, 0):]
                            for m in content]

    def get_data_items(self, dataset, predict=False):
        data = []
        cand_source = 'candidates'
        count = 0

        for doc_name, content in dataset.items():
            count += 1
            if count % 1000 == 0:
                print(count, end='\r')

            items = []
            conll_doc = content[0].get('conll_doc', None)

            for m in content:
                try:
                    # 筛选 候选实体 与 Wikilink 挂钩的
                    named_cands = [c[0] for c in m[cand_source] if
                                   (wiki_prefix + c[0]) in self.model.entity_voca.word2id]
                    p_e_m = [min(1., max(1e-3, c[1])) for c in m[cand_source]]
                except:
                    named_cands = [c[0] for c in m['candidates'] if
                                   (wiki_prefix + c[0]) in self.model.entity_voca.word2id]
                    p_e_m = [min(1., max(1e-3, c[1])) for c in m['candidates']]

                try:
                    true_pos = named_cands.index(m['gold'][0])
                    p = p_e_m[true_pos]
                except:
                    true_pos = -1
                # 选择前 30个候选实体
                named_cands = named_cands[:min(self.args.n_cands_before_rank, len(named_cands))]
                p_e_m = p_e_m[:min(self.args.n_cands_before_rank,len(p_e_m))]

                if true_pos >= len(named_cands):
                    if not predict:
                        true_pos = len(named_cands) - 1
                        p_e_m[-1] = p
                        named_cands[-1] = m['gold'][0]
                    else:
                        true_pos = -1
                # 候选实体 id
                cands = [self.model.entity_voca.get_id(wiki_prefix + c) for c in named_cands]
                mask = [1.] * len(cands)
                if len(cands) == 0 and not predict:
                    continue
                elif len(cands) < self.args.n_cands_before_rank:
                    cands += [self.model.entity_voca.unk_id] * (self.args.n_cands_before_rank - len(cands))
                    named_cands += [Vocabulary.unk_token] * (self.args.n_cands_before_rank - len(named_cands))
                    p_e_m += [1e-8]*(self.args.n_cands_before_rank - len(p_e_m))
                    mask += [0.] * (self.args.n_cands_before_rank - len(mask))

                lctx = m['context'][0].strip().split()
                lctx_ids = [self.prerank_model.word_voca.get_id(t) for t in lctx if utils.is_important_word(t)]
                lctx_ids = [tid for tid in lctx_ids if tid !=self.prerank_model.word_voca.unk_id]
                lctx_ids = lctx_ids[max(0,len(lctx_ids)-self.args.ctx_window//2):]

                rctx = m['context'][1].strip().split()
                rctx_ids = [self.prerank_model.word_voca.get_id(t) for t in rctx if utils.is_important_word(t)]
                rctx_ids = [tid for tid in rctx_ids if tid != self.prerank_model.word_voca.unk_id]
                rctx_ids = rctx_ids[:min(len(rctx_ids), self.args.ctx_window // 2)]

                ment = m['mention'].strip().split()
                ment_ids = [self.prerank_model.word_voca.get_id(t) for t in ment if utils.is_important_word(t)]
                ment_ids = [tid for tid in ment_ids if tid != self.prerank_model.word_voca.unk_id]

                m['sent'] = ' '.join(lctx + rctx)