
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
