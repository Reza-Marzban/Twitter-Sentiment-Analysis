# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 18:46:00 2018

@author: Reza Marzban
"""

from string import punctuation
from gensim.models import Word2Vec
import csv

# load doc into memory
def load_doc(filename):
	file = open(filename, 'r', encoding="utf8")
	text = file.read()
	file.close()
	return text

# turn a doc into clean tokens
def doc_to_clean_lines(line, vocab):
    clean_lines = list()
    tokens = line.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [w for w in tokens if w in vocab]
    clean_lines.append(tokens)
    return clean_lines

# load all docs in a directory
def load_sentences(filename, vocab, is_train):
    lines = list()
    file = open(filename, 'r', encoding="utf8")
    doc=csv.reader(file,delimiter=',')
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
        line = doc_to_clean_lines(tweet, vocab)
        lines+=line
    return lines


# load the vocabulary
file="sentiment_train.csv" 
vocab_filename = file+'_vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load training data
sentences = load_sentences(file,vocab,True)
print('Total training sentences: %d' % len(sentences))

# train word2vec model (workers=cpu cores, window= number of neighbor words considered)
model = Word2Vec(sentences, size=100, window=5, workers=8, min_count=1)
# summarize vocabulary size in model
words = list(model.wv.vocab)
print('Vocabulary size: %d' % len(words))

# save model in ASCII (word2vec) format
filename = file+'_embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)