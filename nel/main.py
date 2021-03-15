from nel.mulrel_ranker import MulRelRanker

import nel.utils as utils

import argparse

parse = argparse.ArgumentParser()
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
parse.add_argument("--mode", type=str,
                   help="train,eval,prerank,or ed",
                   default="train")
parse.add_argument("--model_path", type=str,
                   help="model path to save/load",
                   default="model")
parse.add_argument("--filelist", type=str,
                   help="filelist for ed (candidate filenames should be ended by .csv)",
                   default=None)
parse.add_argument("--preranked_date", type=str,
                   help="filelist for ed (candidate filenames should be ended by .csv)",
                   default=None)

# args for preranking (i.e. 2-step candidate selection)
parse.add_argument("--n_cands_before_rank", type=int,
                   help="number of candidates",
                   default=30)
parse.add_argument("--prerank_ctx_window", type=int,
                   help="size of context window for the preranking model",
                   default=50)
parse.add_argument("--keep_p_e_m", type=int,
                   help="number of top candidates to keep w.r.t using context",
                   default=4)
parse.add_argument("--keep_ctx_ent", type=int,
                   help="number of top candidates to keep w.r.t using context",
                   default=3)

# args for debugging
parse.add_argument("--print_rel", action='store_true')
parse.add_argument("--print_incorrect", action='store_true')

if __name__ == "__main__":
    print('create model')
    word_voca, word_embeddings = utils.load_voca_embs(voca_emb_dir + 'dict.word',
                                                      voca_emb_dir + 'word_embeddings.npy')
    print('word voca size', word_voca.size())
    snd_word_voca, snd_word_embeddings = utils.load_voca_embs(voca_emb_dir + '/glove/dict.word',
                                                              voca_emb_dir + '/glove/word_embeddings.npy')
    print('snd word voca size', snd_word_voca.size())

    entity_voca,entity_embeddings = utils.load_voca_embs(voca_emb_dir+'dict.entity',
                                                         voca_emb_dir + 'entity_embeddings.npy')
    print('entity voca size',entity_voca.size())

