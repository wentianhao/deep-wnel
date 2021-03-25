import torch
import torch.nn as nn

import io
from nel.vocabulary import Vocabulary
import json


def load(path, model_class, suffix=''):
    with io.open(path, '.config', 'r', encoding='utf8') as f:
        config = json.load(f)

    word_voca = Vocabulary()
    word_voca.__dict__ = config['word_voca']
    config['word_voca'] = word_voca
    entity_voca = Vocabulary()
    entity_voca.__dict__ = config['entity_voca']
    config['entity_voca'] = entity_voca

    if 'snd_word_voca' in config:
        snd_word_voca = Vocabulary()
        snd_word_voca.__dict__ = config['snd_word_voca']
        config['snd_word_voca'] = snd_word_voca

    model = model_class(config)
    model.load_state_dict(torch.load(path + '.state_dict' + suffix))
    return model


class AbstractWordEntity(nn.Module):
    """
    包含 word 、 entity embeddings and vocabulary
    """

    def __init__(self, config=None):
        super(AbstractWordEntity, self).__init__()
        if config is None:
            return

        self.emb_dims = config['emb_dims']
        self.word_voca = config['word_voca']
        self.entity_voca = config['entity_voca']
        self.freeze_embs = config['freeze_embs']

        self.word_embeddings = config['word_embeddings_class'](self.word_voca.size(), self.emb_dims)
        self.entity_embeddings = config['entity_embeddings_class'](self.entity_voca.size(), self.emb_dims)

        if 'word_embeddings' in config:
            # 类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter
            self.word_embeddings.weight = nn.Parameter(torch.Tensor(config['word_embeddings']))
        if 'entity_embeddings' in config:
            self.entity_embeddings.weight = nn.Parameter(torch.Tensor(config['entity_embeddings']))

        if 'snd_word_voca' in config:
            self.snd_word_voca = config['snd_word_voca']
            self.snd_word_embeddings = config['word_embeddings_class'](self.snd_word_voca.size(),self.emb_dims)
        if 'snd_word_embeddings' in config:
            self.snd_word_embeddings.weight = nn.Parameter(torch.Tensor(config['snd_word_embeddings']))

        if self.freeze_embs:
            # requires_grad : Tensor的一个属性，用于说明当前量是否需要在计算中保留对应的梯度信息,手动指定requires_grad保证梯度的回传
            self.word_embeddings.weight.requires_grad = False
            self.entity_embeddings.weight.requires_grad = False
            if 'snd_word_embeddings' in config:
                self.snd_word_embeddings.weight.requires_grad = False