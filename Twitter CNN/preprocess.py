# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 14:00:33 2018

@author: Reza Marzban
"""

from string import punctuation
from collections import Counter
from nltk.corpus import stopwords
import csv

# turn a doc into clean tokens
def clean_doc(filename):
    file = open(filename, 'r', encoding="utf8")
    doc=csv.reader(file,delimiter=',')
    words=[]
    row_count = sum(1 for row in doc)
    train_size=int(row_count*0.95)
    print("Total size: "+ str(row_count))
    print("Train size: "+ str(train_size))
    print("Test size: "+ str(row_count-train_size))
    file = open(filename, 'r', encoding="utf8")
    doc=csv.reader(file,delimiter=',')
    for i,row in enumerate(doc):
        if(i>=train_size):
            break
        tweet=row[2]
        tokens = tweet.split(" ")    
        # remove punctuation from each token
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]
        words+=tokens
    file.close()
    return words

# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
    tokens = clean_doc(filename)
    vocab.update(tokens)
        
# save list to file
def save_list(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
    
file="sentiment_train.csv"    
vocab = Counter()
add_doc_to_vocab(file, vocab)
print("The number of Total words:")
print(len(vocab))
min_occurane = 1
tokens = [k for k,c in vocab.items() if c >= min_occurane]
print("The number of words after deleting words that has not repeated even one time:")
print(len(tokens))
# save tokens to a vocabulary file
save_list(tokens, file+'_vocab.txt')

