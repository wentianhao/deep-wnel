import re
import io

LOWER = False
DIGIT_0 = False
UNK_TOKEN = "#UNK#"

# 括号
BRACKETS = {"-LCB-": "{", "-LRB-": "(", "-LSB-": "[", "-RCB-": "}", "-RRB-": ")", "-RSB-": "]"}


class Vocabulary:
    unk_token = UNK_TOKEN

    def __init__(self):
        self.word2id = {}
        self.id2word = []
        self.counts = []
        self.unk_id = 0
