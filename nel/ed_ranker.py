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
            self.ent_net = config['ent_net']
            print('prerank model')
            self.prerank_model = ntee.NTEE(config)
            self.prerank_model.cuda()

        print('main model')
        if self.args.mode in {'eval', 'ed'}:
            print('try loading model from', self.args.model_path)
            self.model = load_model(self.args.model_path, ModelClass)
        else:
            try:
                print('try loading model from', self.args.model_path)
                self.model = load_model(self.args.model_path, ModelClass)
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
                print("ed_ranker    count:", count, end='\r')

            items = []

            if self.args.keep_ctx_ent > 0:
                # rank the candidates by ntee scores
                lctx_ids = [m['context'][0][max(len(m['context'][0]) - self.args.prerank_ctx_window // 2, 0):]
                            for m in content]
                rctx_ids = [m['context'][1][:min(len(m['context'][1]), self.args.prerank_ctx_window // 2)]
                            for m in content]
                ment_ids = [[] for m in content]
                token_ids = [l + m + r if len(l) + len(m) + len(r) > 0 else [self.prerank_model.word_voca.unk_id]
                             for l, m, r in zip(lctx_ids, ment_ids, rctx_ids)]
                token_ids_len = [len(a) for a in token_ids]

                entity_ids = [m['cands'] for m in content]
                entity_ids = Variable(torch.LongTensor(entity_ids).cuda())

                entity_mask = [m['mask'] for m in content]
                entity_mask = Variable(torch.FloatTensor(entity_mask).cuda())

                token_ids, token_offsets = utils.flatten_list_of_lists(token_ids)
                token_offsets = Variable(torch.LongTensor(token_offsets).cuda())
                token_ids = Variable(torch.LongTensor(token_ids).cuda())

                scores, sent_vecs = self.prerank_model.forward(token_ids, token_offsets, entity_ids, use_sum=True,
                                                               return_sent_vecs=True)
                scores = (scores * entity_mask).add_((entity_mask - 1).mul_(1e10))

                if self.args.keep_ctx_ent > 0:
                    top_scores, top_pos = torch.topk(scores, dim=1, k=self.args.keep_ctx_ent)
                    top_scores = top_scores.data.cpu().numpy() / np.array(token_ids_len).reshape(-1, 1)
                    top_pos = top_pos.data.cpu().numpy()
                else:
                    top_scores = None
                    top_pos = [[]] * len(content)

                # compute distribution for sampling negatives
                probs = F.softmax(torch.matmul(sent_vecs, self.prerank_model.entity_embeddings.weight.t()), dim=1)
                _, neg_cands = torch.topk(probs, dim=1, k=1000)
                neg_cands = neg_cands.data.cpu().numpy()

            else:
                top_scores = None
                top_pos = [[]] * len(content)

            # select candidates : mix between keep_ctx_ent best candidates (ntee scores) with
            # keep_p_e_m best candidates (p_e_m scores)
            for i, m in enumerate(content):
                sm = {'cands': [],
                      'named_cands': [],
                      'p_e_m': [],
                      'mask': [],
                      'true_pos': -1}
                m['selected_cands'] = sm
                m['neg_cands'] = neg_cands[i, :]

                selected = set(top_pos[i])
                idx = 0
                while len(selected) < self.args.keep_ctx_ent + self.args.keep_p_e_m:
                    if idx not in selected:
                        selected.add(idx)
                    idx += 1

                selected = sorted(list(selected))
                for idx in selected:
                    sm['cands'].append(m['cands'][idx])
                    sm['named_cands'].append(m['named_cands'][idx])
                    sm['p_e_m'].append(m['p_e_m'][idx])
                    sm['mask'].append(m['mask'][idx])
                    if idx == m['true_pos']:
                        sm['true_pos'] = len(sm['cands']) - 1

                if not predict and not (self.args.multi_instance or self.args.semisup):
                    if sm['true_pos'] == -1:
                        continue
                        # this insertion only makes the performance worse (why???)
                        # sm['true_pos'] = 0
                        # sm['cands'][0] = m['cands'][m['true_pos']]
                        # sm['named_cands'][0] = m['named_cands'][m['true_pos']]
                        # sm['p_e_m'][0] = m['p_e_m'][m['true_pos']]
                        # sm['mask'][0] = m['mask'][m['true_pos']]
                items.append(m)
                if sm['true_pos'] >= 0:
                    has_gold += 1
                total += 1

                # if predict:
                # only for oracle model, not used for eval
                # if sm['true_pos'] == -1:
                #     sm['true_pos'] = 0  # a fake gold, happens only 2%, but avoid the non-gold
            if len(items) > 0:
                if len(items) > 1:
                    c, l, lc, tc = self.get_p_e_ent_net(items)
                    correct += c
                    larger_than_x += l
                    larger_than_x_correct += lc
                    total_cands += tc

                if (not predict) and (not self.args.multi_instance) and (not self.args.semisup):
                    filtered_items = []
                    for m in items:
                        if m['selected_cands']['true_pos'] >= 0:
                            filtered_items.append(m)
                else:
                    filtered_items = items
                new_dataset.append(filtered_items)
        try:
            print('recall', has_gold / total)
        except:
            pass

        if True:  # not predict
            try:
                print('correct', correct, correct / total)
                print('larger_than_x    ', larger_than_x, 'larger_than_x_correct  ', larger_than_x_correct,
                      'larger_than_x_correct/larger_than_x  ', larger_than_x_correct / larger_than_x)
            except:
                pass

        print('----------------------------------')
        return new_dataset

    def get_p_e_ent_net(self, doc):
        eps = -1e3

        entity_ids = [m['selected_cands']['cands'] for m in doc]
        n_ments = len(entity_ids)
        n_cands = len(entity_ids[0])

        def dist(net, u, v):
            w = 0
            if u in net:
                w += net[u].get(v, 0)
            if v in net:
                w += net[v].get(u, 0)
            return w if w > 0 else eps

        p_e_ent_net = np.ones([n_ments, n_cands, n_ments, n_cands]) * (-1e10)
        for mi, mj in product(range(n_ments), range(n_ments)):
            if mi == mj:
                continue

            for i, j in product(range(n_cands), range(n_cands)):
                ei = entity_ids[mi][i]
                ej = entity_ids[mj][j]
                if ei != self.model.entity_voca.unk_id and ej != self.model.entity_voca.unk_id:
                    p_e_ent_net[mi, i, mj, j] = dist(self.ent_net, ei, ej)

        # find scores using LBP
        prev_msgs = torch.zeros(n_ments, n_cands, n_ments).cuda()
        ent_ent_scores = torch.Tensor(p_e_ent_net).cuda()
        local_ent_scores = torch.Tensor([m['selected_cands']['p_e_m'] for m in doc]).cuda()
        df = 0.3
        mask = 1 - torch.eye(n_ments).cuda()
        for _ in range(15):
            ent_ent_votes = ent_ent_scores + local_ent_scores * 0 + \
                            torch.sum(prev_msgs.view(1, n_ments, n_cands, n_ments) * mask.view(n_ments, 1, 1, n_ments),
                                      dim=3).view(n_ments, 1, n_ments, n_cands)
            msgs, _ = torch.max(ent_ent_votes, dim=3)
            msgs = (F.softmax(Variable(msgs), dim=1).data.mul_(df) + prev_msgs.exp_().mul_(1 - df)).log_()
            prev_msgs = msgs

        # compute marginal belief
        ent_scores = local_ent_scores * 0 + torch.sum(msgs * mask.view(n_ments, 1, n_ments), dim=2)
        ent_scores = F.softmax(Variable(ent_scores), dim=1).cpu().data

        correct = 0
        larger_than_x = 0
        larger_than_x_correct = 0
        total_cands = 0

        _, predict = torch.topk(ent_scores, k=min(100, len(doc[0]['selected_cands']['cands'])), dim=1)

        for i in range(n_ments):
            m = doc[i]
            true_pos = m['selected_cands']['true_pos']
            th = ent_scores[i, predict[i, -1]]
            mark = '*'

            if ent_scores[i, predict[i, 0]] > 0.5:
                larger_than_x += 1
                if true_pos == predict[i, 0]:
                    larger_than_x_correct += 1

            sm = {'true_pos': -1}
            selected_ids = []
            for k in list(predict[i]):
                selected_ids.append(k)
                if true_pos == k:
                    sm['true_pos'] = len(selected_ids) - 1
                    correct += 1
                    mark = '+'

            # print(mask, ent_scores[i][true_pos], list(ent_scores[i]))
            total_cands += len(selected_ids)

            n_pads = len(m['selected_cands']['cands']) - len(selected_ids)
            sm['cands'] = [m['selected_cands']['cands'][k] for k in selected_ids] + [
                self.model.entity_voca.unk_id] * n_pads
            sm['named_cands'] = [m['selected_cands']['named_cands'][k] for k in selected_ids] + [
                self.model.entity_voca.unk_token] * n_pads
            sm['p_e_m'] = [m['selected_cands']['p_e_m'][k] for k in selected_ids] + [0.] * n_pads
            sm['p_e_ent_net'] = [ent_scores[i, k] for k in selected_ids] + [0.] * n_pads
            sm['mask'] = [m['selected_cands']['mask'][k] for k in selected_ids] + [0.] * n_pads
            m['selected_cands'] = sm

        return correct, larger_than_x, larger_than_x_correct, total_cands

    def get_data_items(self, dataset, predict=False):
        data = []
        cand_source = 'candidates'
        count = 0

        for doc_name, content in dataset.items():
            count += 1
            if count % 1000 == 0:
                print("count:", count, end='\r')

            items = []
            conll_doc = content[0].get('conll_doc', None)

            for m in content:
                try:
                    # 筛选 候选实体 与 Wikilink 挂钩的,相当于筛选出在Wiki库里的
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
                p_e_m = p_e_m[:min(self.args.n_cands_before_rank, len(p_e_m))]

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
                    p_e_m += [1e-8] * (self.args.n_cands_before_rank - len(p_e_m))
                    mask += [0.] * (self.args.n_cands_before_rank - len(mask))

                lctx = m['context'][0].strip().split()
                lctx_ids = [self.prerank_model.word_voca.get_id(t) for t in lctx if utils.is_important_word(t)]
                lctx_ids = [tid for tid in lctx_ids if tid != self.prerank_model.word_voca.unk_id]
                lctx_ids = lctx_ids[max(0, len(lctx_ids) - self.args.ctx_window // 2):]

                rctx = m['context'][1].strip().split()
                rctx_ids = [self.prerank_model.word_voca.get_id(t) for t in rctx if utils.is_important_word(t)]
                rctx_ids = [tid for tid in rctx_ids if tid != self.prerank_model.word_voca.unk_id]
                rctx_ids = rctx_ids[:min(len(rctx_ids), self.args.ctx_window // 2)]

                ment = m['mention'].strip().split()
                ment_ids = [self.prerank_model.word_voca.get_id(t) for t in ment if utils.is_important_word(t)]
                ment_ids = [tid for tid in ment_ids if tid != self.prerank_model.word_voca.unk_id]

                m['sent'] = ' '.join(lctx + rctx)

                # secondary local context (for computing relation scores) (计算相关性分数)
                if conll_doc is not None:
                    conll_m = m['conll_m']
                    sent = conll_doc['sentences'][conll_m['sent_id']]
                    start = conll_m['start']
                    end = conll_m['end']

                    snd_lctx = [self.model.snd_word_voca.get_id(t)
                                for t in sent[max(0, start - self.args.snd_local_ctx_window // 2):start]]
                    snd_rctx = [self.model.snd_word_voca.get_id(t)
                                for t in sent[end:min(len(sent), end + self.args.snd_local_ctx_window // 2)]]
                    snd_ment = [self.model.snd_word_voca.get_id(t)
                                for t in sent[start:end]]

                    if len(snd_lctx) == 0:
                        snd_lctx = [self.model.snd_word_voca.unk_id]
                    if len(snd_rctx) == 0:
                        snd_rctx = [self.model.snd_word_voca.unk_id]
                    if len(snd_ment) == 0:
                        snd_ment = [self.model.snd_word_voca.unk_id]
                else:
                    snd_lctx = [self.model.snd_word_voca.unk_id]
                    snd_rctx = [self.model.snd_word_voca.unk_id]
                    snd_ment = [self.model.snd_word_voca.unk_id]

                items.append({'context': (lctx_ids, rctx_ids),
                              'snd_ctx': (snd_lctx, snd_rctx),
                              'ment_ids': ment_ids,
                              'snd_ment': snd_ment,
                              'cands': cands,
                              'named_cands': named_cands,
                              'p_e_m': p_e_m,
                              'mask': mask,
                              'true_pos': true_pos,
                              'doc_name': doc_name,
                              'raw': m})

            if len(items) > 0:
                # note: this shouldn't affect the order of prediction because we use doc_name to add predicted entities,
                # and we don't shuffle the data for prediction
                max_len = 50
                if len(items) > max_len:
                    print("#items:", len(items))
                    for k in range(0, len(items), max_len):
                        data.append(items[k:min(len(items), k + max_len)])
                else:
                    data.append(items)
        return self.prerank(data, predict)

    def minibatch2input(self, batch, predict=False, topk=None):
        if topk == None:
            topk = 10000
        topk = min(topk, len(batch[0]['selected_cands']['cands']))

        n_ments = len(batch)
        # only uisng negative samples when the document doesn't have any supervision (i.e. not CoNLL)
        tps = [m['selected_cands']['true_pos'] >= 0 for m in batch]
        if not predict and (self.args.multi_instance or self.args.semisup) and not np.any(tps):
            n_negs = self.args.n_negs
        else:
            n_negs = 0

        # convert data items to pytorch inputs
        token_ids = [m['context'][0] + m['context'][1]
                     if len(m['context'][0]) + len(m['context'][1]) > 0
                     else [self.model.word_voca.unk_id]
                     for m in batch]
        s_ltoken_ids = [m['snd_ctx'][0] for m in batch]
        s_rtoken_ids = [m['snd_ctx'][1] for m in batch]
        s_mtoken_ids = [m['snd_ment'] for m in batch]

        entity_ids = torch.LongTensor([m['selected_cands']['cands'][:topk] for m in batch])
        p_e_m = torch.FloatTensor([m['selected_cands']['p_e_m'][:topk] for m in batch])
        entity_mask = torch.FloatTensor([m['selected_cands']['mask'][:topk] for m in batch])
        true_pos = torch.LongTensor(
            [m['selected_cands']['true_pos'] if m['selected_cands']['true_pos'] < topk else -1 for m in batch])
        p_e_ent_net = torch.FloatTensor([m['selected_cands']['p_e_ent_net'][:topk] for m in batch]) if len(
            batch) > 1 else torch.zeros(1, entity_ids.shape[1])

        if n_negs > 0:
            # add n_negs negative samples at the beginning of lists
            def ent_neg_sample(neg_cands_p_e_m, exclusive):
                # 随机抽取 [0,len(neg_cands_p_e_m)) ,生成 一维数组 长度为 n_negs*10
                sample_ids = np.random.choice(len(neg_cands_p_e_m), n_negs * 10)
                all_samples = list(zip(np.array([s[0] for s in neg_cands_p_e_m])[sample_ids].astype(int),
                                       np.array(s[1] for s in neg_cands_p_e_m)[sample_ids]))
                exclusive = set(exclusive)
                samples = []
                for s in all_samples:
                    if s[0] not in exclusive:
                        samples.append(s)

                if len(samples) < n_negs:
                    samples = samples + [(self.model.entity_voca.unk_id, 1e-3)] * (n_negs - len(samples))
                else:
                    shuffle(samples)
                    samples = samples[:n_negs]
                return np.array([s[0] for s in samples]), np.array([s[1] for s in samples])

            neg_cands_p_e_m = [list(zip(list(m['cands']), list(m['p_e_m']))) + \
                               (list(zip(list(m['neg_cands'], [1e-3] * len(m['neg_cands'])))) if len(
                                   m['cands']) <= topk else [])
                               for m in batch]
            neg_cands_p_e_m = [ent_neg_sample(si, entity_ids_i) for si, entity_ids_i in
                               zip(neg_cands_p_e_m, entity_ids.numpy())]
            neg_entity_ids = torch.Tensor([si[0].astype(float) for si in neg_cands_p_e_m]).long()
            neg_p_e_m = torch.Tensor([si[1].astype(float) for si in neg_cands_p_e_m])

            neg_entity_mask = torch.ones(n_ments, n_negs)
            entity_ids = torch.cat([neg_entity_ids, entity_ids], dim=1)
            entity_mask = torch.cat([neg_entity_mask, entity_mask], dim=1)
            p_e_m = torch.cat([neg_p_e_m, p_e_m], dim=1)
            true_pos = true_pos.add_(n_negs)

        entity_ids = Variable(entity_ids.cuda())
        true_pos = Variable(true_pos.cuda())
        p_e_m = Variable(p_e_m.cuda())
        p_e_ent_net = Variable(p_e_ent_net.cuda())
        entity_mask = Variable(entity_mask.cuda())

        token_ids, token_mask = utils.make_equal_len(token_ids, self.model.word_voca.unk_id)
        s_ltoken_ids, s_ltoken_mask = utils.make_equal_len(s_ltoken_ids, self.model.snd_word_voca.unk_id,
                                                           to_right=False)
        s_rtoken_ids, s_rtoken_mask = utils.make_equal_len(s_rtoken_ids, self.model.snd_word_voca.unk_id)
        s_rtoken_ids = [l[::-1] for l in s_rtoken_ids]
        s_rtoken_mask = [l[::-1] for l in s_rtoken_mask]
        s_mtoken_ids, s_mtoken_mask = utils.make_equal_len(s_mtoken_ids, self.model.snd_word_voca.unk_id)

        token_ids = Variable(torch.LongTensor(token_ids).cuda())
        token_mask = Variable(torch.FloatTensor(token_mask).cuda())

        s_ltoken_ids = Variable(torch.LongTensor(s_ltoken_ids).cuda())
        s_ltoken_mask = Variable(torch.FloatTensor(s_ltoken_mask).cuda())
        s_rtoken_ids = Variable(torch.LongTensor(s_rtoken_ids).cuda())
        s_rtoken_mask = Variable(torch.FloatTensor(s_rtoken_mask).cuda())
        s_mtoken_ids = Variable(torch.LongTensor(s_mtoken_ids).cuda())
        s_mtoken_mask = Variable(torch.FloatTensor(s_mtoken_mask).cuda())

        ret = {'token_ids': token_ids,
               'token_mask': token_mask,
               'entity_ids': entity_ids,
               'entity_mask': entity_mask,
               'p_e_m': p_e_m,
               'p_e_ent_net': p_e_ent_net,
               'true_pos': true_pos,
               's_ltoken_ids': s_ltoken_ids,
               's_ltoken_mask': s_ltoken_mask,
               's_rtoken_ids': s_rtoken_ids,
               's_rtoken_mask': s_rtoken_mask,
               's_mtoken_ids': s_mtoken_ids,
               's_mtoken_mask': s_mtoken_mask,
               'n_negs': n_negs}
        return ret

    def train(self, org_train_dataset, org_dev_datasets, config, preranked_train=None, preranked_dev=None):
        print('extracting training data')
        if preranked_train is None:
            train_dataset = self.get_data_items(org_train_dataset, predict=False)
        else:
            train_dataset = preranked_train
        print('#train docs', len(train_dataset))

        if preranked_dev is None:
            dev_datasets = []
            for dname, data in org_dev_datasets:
                dev_datasets.append(dname, self.get_data_items(data, predict=True))
                print(dname, '#dev docs', len(dev_datasets[-1][1]))
        else:
            dev_datasets = preranked_dev

        print('creating optimizer')
        # 构造一个优化器对象optimizer
        optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=config['lr'])
        best_f1 = -1
        not_better_count = 0
        is_counting = False
        stop = False
        eval_after_n_epochs = self.args.eval_after_n_epochs
        final_result_str = ''

        print('total training items', len(train_dataset))
        n_updates = 0
        if config['multi_instance']:
            n_updates_to_eval = 1000
            n_updates_to_stop = 60000
            f1_threshold = 0.875
            f1_start_couting = 0.87
        elif config['semisup']:
            n_updates_to_eval = 5000
            n_update_to_stop = 1e10
            f1_threshold = 0.86
            f1_start_couting = 0.86
        else:  # for supervised learning
            n_updates_to_eval = 1000
            n_updates_to_stop = 1000 * self.args.n_epochs
            f1_threshold = 0.95
            f1_start_couting = 0.95

        for e in range(config['n_epochs']):
            shuffle(train_dataset)

            total_loss = 0
            total = 0

            for dc, batch in enumerate(train_dataset):  # each document is a minibatch
                self.model.train()
                optimizer.zero_grad()
                tps = [m['selected_cands']['true_pos'] >= 0 for m in batch]
                any_true = np.any(tps)

                if any_true:
                    inputs = self.minibatch2input(batch)
                else:
                    inputs = self.minibatch2input(batch, topk=2)

                if config['semisup']:
                    if any_true:  # from supervision (i.e CoNLL)
                        scores = self.model.forward(inputs, gold=inputs['true_pos'].view(-1, 1), inference='LBP')
                    else:
                        scores = self.model.forward(inputs, gold=inputs['true_pos'].view(-1, 1), inference='star')
                else:
                    scores = self.model.forward(inputs, gold=inputs['true_pos'].view(-1, 1))

                if any_true:
                    loss = self.model.loss(scores, inputs['true_pos'])
                else:
                    loss = self.model.multi_instance_loss(scores, inputs)

                loss.backward()
                optimizer.step()

                loss = loss.cpu().data.item()
                total_loss += loss

                if dc % 100 == 0:
                    print('epoch', e, "%0.2f%%" % (dc / len(train_dataset) * 100), loss, end='\r')
