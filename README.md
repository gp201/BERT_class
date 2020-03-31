# BERT_executable

bert_model file [creation](https://github.com/gp201/BERT/tree/master/BERT_Custom)

Unzip the bert_model.zip file in the same folder the repository content is placed in.     
Run the restore_bert_model.py file

## Input:
Any sentence to para    
ex: Chikungunya virus ( CHIKV ) is a mosquito. DCTN4 as a modifier of chronic Pseudomonas aeruginosa infection in cystic fibrosis.

## Output:
A list within a list, where each list contains tags for each sentence    
ex: ```[['T005', 'T005', 'T005', 'O', 'O', 'T204'], ['T116,T123', 'O', 'O', 'O', 'O', 'T047', 'T047', 'T047', 'T047', 'O', 'T047', 'T047'], []]```
