import pickle
datadir = '/home/wenh/data/generated/test_train_data/'
print('load entity net from', datadir + '/../entity_net.dat')
entity_net = pickle.load(open(datadir + '/../entity_net.dat', 'rb'))
print(entity_net)