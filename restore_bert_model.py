import os
import csv
import sys
import torch
import logging
from flair.datasets import ColumnCorpus
from keras.preprocessing.text import text_to_word_sequence

from bert_sequence_tagger.model_trainer_bert import ModelTrainerBert
from bert_sequence_tagger import SequenceTaggerBert, BertForTokenClassificationCustom
from bert_sequence_tagger.bert_utils import get_model_parameters, prepare_flair_corpus
from bert_sequence_tagger.metrics import f1_entity_level, f1_token_level, f1_per_token, f1_per_token_plot
from bert_sequence_tagger.plot_classification_report import plot_classification_report
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('sequence_tagger_bert')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda')
n_gpu = torch.cuda.device_count()

for i in range(n_gpu):
    print(torch.cuda.get_device_name(i))

seq_tagger = SequenceTaggerBert.load_serialized('/content/drive/My Drive/GITHUB/BERT/Bert_models/bert_model_T047005007016', BertForTokenClassificationCustom)

locs = ['/content/drive/My Drive/GITHUB/BERT/Bert_models/bert_model_T005','/content/drive/My Drive/GITHUB/BERT/Bert_models/bert_model_T007','/content/drive/My Drive/GITHUB/BERT/Bert_models/bert_model_T016','/content/drive/My Drive/GITHUB/BERT/Bert_models/bert_model_T047']
models = [SequenceTaggerBert.load_serialized(x , BertForTokenClassificationCustom) for x in locs]

class BERT:
  def return_tags_MMT(self,para):
    para_split = para.split('.')
    para_split = [text_to_word_sequence(x) for x in para_split]
    tags,prob = seq_tagger.predict(para_split)
    op_file_1 = open("data/para_tags_MMT.csv",'w')
    writer_1 = csv.writer(op_file_1)
    _ = ['Words','Tags']
    writer_1.writerow(_)
    for i,t in zip(para_split,tags):
      for x,y in zip(i,t):
        row = [x,y]
        writer_1.writerow(row)
      writer_1.writerow([''])
    '''
    for each row to have a full sentence(word per cell in a row) and tags(wrt word) use the code below
    for i,t in zip(para_split,final_tags):
      writer.writerow(i)
      writer.writerow(t)
      writer.writerow([''])
    '''

  def return_tags_MPT(self,para):
    final_tags = []
    for mod in models:
        para_split = para.split('. ')
        para_split = [text_to_word_sequence(x) for x in para_split]
        tags,prob = mod.predict(para_split)
        if not final_tags:
          final_tags = tags
        for _ in range(len(final_tags)):
          for x in range(len(final_tags[_])):
            if final_tags[_][x] == 'O':
              final_tags[_][x] = tags[_][x] 
            elif tags[_][x] != 'O' and final_tags[_][x] != tags[_][x]:
              final_tags[_][x] += ', ' + tags[_][x]
    op_file = open("data/para_tags_MPT.csv",'w')
    writer = csv.writer(op_file)
    _ = ['Words','Tags']
    writer.writerow(_)
    for i,t in zip(para_split,final_tags):
      for x,y in zip(i,t):
        row = [x,y]
        writer.writerow(row)
      writer.writerow([''])
    '''
    for each row to have a full sentence(word per cell in a row) and tags(wrt word) use the code below
    for i,t in zip(para_split,final_tags):
      writer.writerow(i)
      writer.writerow(t)
      writer.writerow([''])
    '''

  def give_metrics(self):
    test_file = input("Test-File: ")
    data_folder = 'data'
    corpus = ColumnCorpus(data_folder, {0 : 'idx', 2 : 'text', 3 : 'ner' },
                        train_file=test_file,
                        test_file=test_file,
                        dev_file=test_file)
    test_dataset = prepare_flair_corpus(corpus.test)

    _, __, test_metrics = seq_tagger.predict(test_dataset, evaluate=True, 
                                            metrics=[f1_entity_level, f1_token_level, f1_per_token, f1_per_token_plot])
    logger.info(f'Entity-level f1: {test_metrics[1]}')
    logger.info(f'Token-level f1: {test_metrics[2]}') 
    logger.info(f'per-Token-level f1: \n {test_metrics[3]}')

    ##Confusion matrix style
    plot_classification_report(test_metrics[3])

    ##Classification report BAR graph##
    results = test_metrics[4][0]
    key = ['precision','recall','f1-score','support']
    indices = np.arange(len(results))
    labels = [y for y in results.keys()]
    results_t = [[x[i] for x in results.values()] for i in key]
    precision,recall,f1_score,support = results_t
    plt.title("Metrics")
    plt.barh(indices, precision, .2, label="precision", color='navy',)
    plt.barh(indices + .3, f1_score, .2, label="f1_score",
             color='c')
    plt.barh(indices + .6, recall, .2, label="recall", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, labels):
        plt.text(-.1, i, c,)

    for i, v in enumerate(precision):
        plt.text(v , i - 0.1, str(v)[0:6], color='navy')

    for i, v in enumerate(f1_score):
        plt.text(v , i + 0.2, str(v)[0:6], color='c')

    for i, v in enumerate(recall):
        plt.text(v , i + 0.5, str(v)[0:6], color='darkorange')

    #plt.show()
    fig = plt.gcf()
    #20,100
    fig.set_size_inches(20,20, forward=True)
    fig.savefig('class_rep_bar_graph.png',format='png', bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close()

    ##Accuracy graph##
    objects = test_metrics[4][2]
    y_pos = np.arange(len(objects))
    performance = test_metrics[4][1]

    plt.barh(y_pos, performance, align='center', alpha=0.5)
    plt.yticks(y_pos, objects)
    plt.xlabel('Accuracy')
    plt.title('Accuracy per token/class')

    for i, v in enumerate(performance):
      plt.text(v , i - 0.1, str(v)[0:4], color='navy')

    fig = plt.gcf()
    fig.set_size_inches(7,5, forward=True)
    fig.savefig('acc_bar_graph.png',format='png', bbox_inches='tight', dpi=300)
    
  def __init__(self):
    run = 1
    print('Please put the test.txt file generated for the BERT model in the data folder')
    while run != 4:
      run = int(input("\n--------------------\nChoose: \n1)Tags_MMT \n2)Tags_MPT \n3)Metrics \n4)Exit \nInput option num : "))
      print("--------------------\n")
      if run == 1:
        print("Enter text: ")
        para = input()
        self.return_tags_MMT(para)
      elif run == 2:
        print("Enter text: ")
        para = input()
        self.return_tags_MPT(para)
      elif run == 3:
        self.give_metrics()

run = BERT()
