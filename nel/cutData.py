import nel.utils as utils
import pickle
from random import shuffle

datadir = '/home/wenh/data/generated/test_train_data/'
conll_path = '/home/wenh/data/basic_data/test_datasets/'
person_path = '/home/wenh/data/basic_data/p_e_m_data/persons.txt'
voca_emb_dir = '/home/wenh/data/generated/embeddings/large_word_ent_embs/'
preranked_data = '/home/wenh/data/generated/test_train_data_bk/preranked_all_datasets_50kRCV1_large'
# print('create model')
# word_voca, word_embeddings = utils.load_voca_embs(voca_emb_dir + 'dict.word',
#                                                   voca_emb_dir + 'word_embeddings.npy')
# print('word voca size', word_voca.size())
# snd_word_voca, snd_word_embeddings = utils.load_voca_embs(voca_emb_dir + '/glove/dict.word',
#                                                           voca_emb_dir + '/glove/word_embeddings.npy')
# print('snd word voca size', snd_word_voca.size())
#
# entity_voca, entity_embeddings = utils.load_voca_embs(voca_emb_dir + 'dict.entity',
#                                                       voca_emb_dir + 'entity_embeddings.npy')
# print('entity voca size', entity_voca.size())
#
# print('load entity net from', datadir + '/../entity_net.dat')
# entity_net = pickle.load(open(datadir + '/../entity_net.dat', 'rb'))
path = preranked_data
print('load all datasets from', path)
with open(path, 'rb') as f:
    all_datasets = pickle.load(f)
    conll_train, preranked_train = all_datasets[0][1], all_datasets[0][2]
    print(len(conll_train), len(preranked_train))
    print(len(all_datasets[1][1]))
    preranked_train = preranked_train[:min(50000, len(preranked_train))]
    shuffle(preranked_train)

    dev_datasets = [(all_datasets[i][0], all_datasets[i][1]) for i in range(1, len(all_datasets))]
    preranked_dev = [(all_datasets[i][0], all_datasets[i][2]) for i in range(1, len(all_datasets))]
