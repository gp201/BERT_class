#for returning tags
from bert_sequence_tagger import SequenceTaggerBert, BertForTokenClassificationCustom

import os
import torch
from keras.preprocessing.text import text_to_word_sequence

#for returning metrics
from bert_sequence_tagger.bert_utils import get_model_parameters, prepare_flair_corpus
from bert_sequence_tagger.model_trainer_bert import ModelTrainerBert
from bert_sequence_tagger.metrics import f1_entity_level, f1_token_level, f1_per_token
import logging
import sys
from flair.datasets import ColumnCorpus

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('sequence_tagger_bert')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda')
n_gpu = torch.cuda.device_count()

for i in range(n_gpu):
    print(torch.cuda.get_device_name(i))

seq_tagger = SequenceTaggerBert.load_serialized('./bert_model_full', BertForTokenClassificationCustom)

class BERT:
  def return_tags(self,para):
    para_split = para.split('.')
    para_split = [text_to_word_sequence(x) for x in para_split]
    tags,prob = seq_tagger.predict(para_split)
    print(tags)

  def give_metrics(self):
    data_folder = 'data'
    corpus = ColumnCorpus(data_folder, {0 : 'idx', 2 : 'text', 3 : 'ner' },
                        train_file='test.txt',
                        test_file='test.txt',
                        dev_file='test.txt')
    test_dataset = prepare_flair_corpus(corpus.test)

    _, __, test_metrics = seq_tagger.predict(test_dataset, evaluate=True, 
                                            metrics=[f1_entity_level, f1_token_level, f1_per_token])
    logger.info(f'Entity-level f1: {test_metrics[1]}')
    logger.info(f'Token-level f1: {test_metrics[2]}') 
    logger.info(f'per-Token-level f1: \n {test_metrics[3]}')
    
  def __init__(self):
    run = 1
    print('Please put the test.txt file generated for the BERT model in the data folder')
    while run != 3:
      run = int(input("do you want to continue: \n1)Tags \n2)Metrics \n3)Exit \ninput option num : "))
      if run == 1: 
        print("Enter text: ")
        para = input()
        self.return_tags(para)
      elif run == 2:  
        self.give_metrics()

run = BERT()

