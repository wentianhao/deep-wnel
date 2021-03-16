import re
from pprint import pprint

wiki_link_prefix = 'http://en.wikipedia.org/wiki/'


def read_csv_file(path):
    data = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            comps = line.strip().split('\t')
            doc_name = comps[0] + ' ' + comps[1]
            mention = comps[2]
            lctx = comps[3]
            rctx = comps[4]

            if comps[6] != 'EMPTYCAND':
                # id , p , candidate
                cands = [c.split(',') for c in comps[6:-2]]
                # candidate , p
                cands = [(','.join(c[2:]).replace('"', '%22').replace(' ', '_'), float(c[1])) for c in cands]
            else:
                cands = []

            gold = comps[-1].split(',')
            if gold[0] == '-1':
                gold = (','.join(gold[2:]).replace('"', '%22').replace(' ', '_'), 1e-5, -1)
            else:
                gold = (','.join(gold[3:]).replace('"', '%22').replace(' ', '_'), 1e-5, -1)

            if doc_name not in data:
                data[doc_name] = []
            data[doc_name].append({'mention': mention,
                                   'context': (lctx, rctx),
                                   'candidates': cands,
                                   'gold': gold})
        return data


def read_conll_file(data, path, ner_path=None):
    conll = {}
    with open(path, 'r', encoding='utf8') as f:
        if ner_path is not None:
            fner = open(ner_path, 'r')
        else:
            fner = None

        cur_sent = None
        cur_doc = None

        for line in f:
            line = line.strip()
            if fner is not None:
                l1 = fner.readline()
                if l1 == '':
                    l1 = None
                else:
                    l1 = l1.strip()
            else:
                l1 = None

            if line.startswith('-DOCSTART-'):
                doc_name = line.split()[1][1:]
                conll[doc_name] = {'sentences': [], 'mentions': []}
                cur_doc = conll[doc_name]
                cur_sent = []
                if l1 is not None:
                    if not l1.startswith('-DOCSTART-'):
                        raise Exception('wrong')
                    l1 = fner.readline().strip()  # an empty line right after this
                else:
                    if line == '':
                        cur_doc['sentences'].append(cur_sent)
                        cur_sent = []
                        if l1 is not None:
                            if l1 != '':
                                raise Exception('wrong')
                    else:
                        comps = line.split('\t')
                        tok = comps[0]
                        cur_sent.append(tok)

                        if len(comps) >= 6:
                            bi = comps[1]
                            wiki_link = comps[4]
                            if bi == 'I':
                                cur_doc['mention'][-1]['end'] += 1
                                if l1 is not None:
                                    tok, _, _, cat = l1.split()
                                    if not cat.startswith('I'):
                                        raise Exception('wrong')
                            else:
                                new_ment = {'sent_id': len(cur_doc['sentences']),
                                            'start': len(cur_sent) - 1,
                                            'end': len(cur_sent),
                                            'wiki_link': wiki_link}
                                if l1 is not None:
                                    tok, _, _, cat = l1.split()
                                    _, cat = cat.split('-')
                                    new_ment['cat'] = cat
                                cur_doc['mentions'].append(new_ment)
        # the last sentence
        if len(cur_sent) > 0:
            cur_doc['sentences'].append(cur_sent)
            cur_sent = []

    # merge with data


def find_coref(ment, ment_list, person_names):
    cur_m = ment['mention'].lower()
    coref = []
    for m in ment_list:
        if len(m['candidates']) == 0 or m['candidates'][0][0] not in person_names:
            continue

        mention = m['mention'].lower()
        start_pos = mention.find(cur_m)
        if start_pos == -1 or mention == cur_m:
            continue

        end_pos = start_pos + len(cur_m) - 1
        if (start_pos == 0 or mention[start_pos - 1] == ' ') and \
                (end_pos == len(mention) - 1 or mention[end_pos + 1] == ' '):
            coref.append(m)

    return coref


def with_coref(dataset, person_names):
    for data_name, content in dataset.items():
        for cur_m in content:
            coref = find_coref(cur_m, content, person_names)
            if coref is not None and len(coref) > 0:
                cur_cands = {}
                for m in coref:
                    for c, p in m['candidates']:
                        cur_cands[c] = cur_cands.get(c, 0) + p
                for c in cur_cands.keys():
                    cur_cands[c] /= len(coref)
                cur_m['candidates'] = sorted(list(cur_cands.items()), key=lambda x: x[1])[::-1]


class CoNLLDataset:
    """
    reading dataset from CoNLL dataset,extracted by https://github.com/dalab/deep-ed/
    """

    def __init__(self, path, person_path, conll_path):
        print('load csv')
        self.train = read_csv_file(path + '/aida_train.csv')
        self.testA = read_csv_file(path + '/aida_testA.csv')
        self.testB = read_csv_file(path + '/aida_testB.csv')
        self.ace2004 = read_csv_file(path + '/wned-ace2004.csv')
        self.aquaint = read_csv_file(path + '/wned-aquaint.csv')
        self.clueweb = read_csv_file(path + '/wned-clueweb.csv')
        self.msnbc = read_csv_file(path + '/wned-msnbc.csv')
        self.wikipedia = read_csv_file(path + '/wned-wikipedia.csv')
        self.wikipedia.pop('Jiří_Třanovský Jiří_Třanovský', None)  # unknown problem with this


if __name__ == "__main__":
    path = 'D:/download/wnel-data/generated/test_train_data/'
    conll_path = 'D:/download/wnel-data/basic_data/test_datasets/'
    person_path = 'D:/download/wnel-data/basic_data/p_e_m_data/persons.txt'

    dataset = CoNLLDataset(path, person_path, conll_path)
    pprint(dataset.testA, width=200)
