from bert_sequence_tagger import SequenceTaggerBert, BertForTokenClassificationCustom

import os
import torch
from keras.preprocessing.text import text_to_word_sequence

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda')
n_gpu = torch.cuda.device_count()

for i in range(n_gpu):
    print(torch.cuda.get_device_name(i))

seq_tagger = SequenceTaggerBert.load_serialized('./bert_model', BertForTokenClassificationCustom)

class BERT:
  def return_tags(self,para):
    para_split = para.split('.')
    para_split = [text_to_word_sequence(x) for x in para_split]
    tags,prob = seq_tagger.predict(para_split)
    print(tags)
  def __init__(self):
    run = 1
    while run == 1:
      print('k')
      print("Enter text: ")
      print('k2')
      para = input()
      self.return_tags(para)
      run = int(input("do you want to continue: 1)yes 2)no : "))

run = BERT()