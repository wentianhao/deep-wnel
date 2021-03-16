from nel.mulrel_ranker import MulRelRanker
from nel.ed_ranker import EDRanker
import pickle

import nel.utils as utils

from random import shuffle
from pprint import pprint

import argparse

parser = argparse.ArgumentParser()
use_large = True

datadir = 'D:/IDE/Python/EntityLinking/wnel-data/generated/test_train_data/'
conll_path = 'D:/IDE/Python/EntityLinking/wnel-data/basic_data/test_datasets/'
person_path = 'D:/IDE/Python/EntityLinking/wnel-data/basic_data/p_e_m_data/persons.txt'
voca_emb_dir = 'D:/IDE/Python/EntityLinking/wnel-data/generated/embeddings/word_ent_embs/'
large_voca_emb_dir = 'D:/IDE/Python/EntityLinking/wnel-data/generated/embeddings/large_word_ent_embs/'

if use_large:
    voca_emb_dir = large_voca_emb_dir

ModelClass = MulRelRanker

# general args
parser.add_argument("--mode", type=str,
                    help="train,eval,prerank,or ed",
                    default="prerank")
parser.add_argument("--model_path", type=str,
                    help="model path to save/load",
                    default="model")
parser.add_argument("--filelist", type=str,
                    help="filelist for ed (candidate filenames should be ended by .csv)",
                    default=None)
parser.add_argument("--preranked_data", type=str,
                    help="filelist for ed (candidate filenames should be ended by .csv)",
                    default=None)

# args for preranking (i.e. 2-step candidate selection)
parser.add_argument("--n_cands_before_rank", type=int,
                    help="number of candidates",
                    default=30)
parser.add_argument("--prerank_ctx_window", type=int,
                    help="size of context window for the preranking model",
                    default=50)
parser.add_argument("--keep_p_e_m", type=int,
                    help="number of top candidates to keep w.r.t using context",
                    default=4)
parser.add_argument("--keep_ctx_ent", type=int,
                    help="number of top candidates to keep w.r.t using context",
                    default=3)

# args for local model
parser.add_argument("--ctx_window", type=int,
                    help="size of context window for the local model",
                    default=100)
parser.add_argument("--tok_top_n", type=int,
                    help="number of top contextual words for the local model",
                    default=25)

# args for global model
parser.add_argument("--mulrel_type", type=str,
                    help="type for multi relation (rel-norm or ment-norm)",
                    default="ment-norm")
parser.add_argument("--n_rels", type=int,
                    help="number of relations",
                    default=5)
parser.add_argument("--hid_dims", type=int,
                    help="number of hidden neurons",
                    default=100)
parser.add_argument("--snd_local_ctx_window", type=int,
                    help="local ctx window size for relation scores",
                    default=6)
parser.add_argument("--dropout_rate", type=float,
                    help="dropout rate for relation scores",
                    default=0.3)
parser.add_argument("--uniform_att", action='store_true',
                    help='with uniform attention(equ. to G&H)')

# args for training
parser.add_argument("--n_epochs", type=int,
                    help="max number of epochs",
                    default=200)
parser.add_argument("--dev_f1_change_lr", type=float,
                    help="dev f1 to change learning rate",
                    default=0.915)
parser.add_argument("--n_not_inc", type=int,
                    help="number of evals after dev f1 not increase",
                    default=10)
parser.add_argument("--eval_after_n_epochs", type=int,
                    help="number of epochs to eval",
                    default=5)
parser.add_argument("--learning_rate", type=float,
                    help="learning rate",
                    default=1e-4)
parser.add_argument("--margin", type=float,
                    help="margin",
                    default=0.01)

parser.add_argument("--n_negs", type=int,
                    help="number of negatives",
                    default=0)
parser.add_argument('--semisup', action='store_true',
                    help='using multi-instance learning and supervised learning (for semi supervised learning)')
parser.add_argument('--multi_instance', action='store_true',
                    help='using multi-instance learning')

# args for inference (LBP or star)
parser.add_argument("--inference", type=str,
                    help="inference method (LBP or star)",
                    default='LBP')
parser.add_argument("--df", type=float,  # for LBP
                    help="dampling factor (for LBP)",
                    default=0.5)
parser.add_argument("--ent_loops", type=int,  # for LBP
                    help="number of LBP loops",
                    default=10)
parser.add_argument("--ent_top_n", type=int,  # for star
                    help="number of kept neighbours",
                    default=30)

parser.add_argument("--n_docs", type=int,
                    help="number og documents",
                    default=1000000)
parser.add_argument("--dev_enr", type=str,
                    help="dev net path",
                    default=None)

# args for debugging
parser.add_argument("--print_rel", action='store_true')
parser.add_argument("--print_incorrect", action='store_true')

args = parser.parse_args()
if (args.semisup or args.multi_instance) and args.n_negs < 1:
    raise Exception("multi instance requires at least 1 negative sample")

if __name__ == "__main__":
    print('create model')
    word_voca, word_embeddings = utils.load_voca_embs(voca_emb_dir + 'dict.word',
                                                      voca_emb_dir + 'word_embeddings.npy')
    print('word voca size', word_voca.size())
    snd_word_voca, snd_word_embeddings = utils.load_voca_embs(voca_emb_dir + '/glove/dict.word',
                                                              voca_emb_dir + '/glove/word_embeddings.npy')
    print('snd word voca size', snd_word_voca.size())

    entity_voca, entity_embeddings = utils.load_voca_embs(voca_emb_dir + 'dict.entity',
                                                          voca_emb_dir + 'entity_embeddings.npy')
    print('entity voca size', entity_voca.size())

    config = {'hid_dims': args.hid_dims,
              'emb_dims': entity_embeddings.shape[1],
              'freeze_embs': True,
              'tok_top_n': args.tok_top_n,
              'margin': args.margin,
              'word_voca': word_voca,
              'entity_voca': entity_voca,
              'word_embeddings': word_embeddings,
              'entity_embeddings': entity_embeddings,
              'snd_word_voca': snd_word_voca,
              'snd_word_embeddings': snd_word_embeddings,
              'dr': args.dropout_rate,
              'multi_instance': args.multi_instance,
              'semisup': args.semisup,
              'preranked_data': args.preranked_data,
              'uniform_att': args.uniform_att,
              'args': args}

    if args.mode == 'prerank':
        print('load entity net from', datadir + '/../entity_net.dat')
        entity_net = pickle.load(open(datadir + '/../entity_net.dat', 'rb'))
        config['ent_net'] = entity_net

    if args.multi_instance or args.semisup:
        config['n_negs'] = args.n_negs

    if ModelClass == MulRelRanker:
        config['inference'] = args.inference
        config['df'] = args.df
        config['n_loops'] = args.n_loops
        config['ent_top_n'] = args.ent_top_n

        config['n_rels'] = args.n_rels
        config['mulrel_type'] = args.mulrel_type
    else:
        raise Exception('unknown model class')

    pprint(config)
    ranker = EDRanker(config=config)

    if args.mode == 'prerank':
