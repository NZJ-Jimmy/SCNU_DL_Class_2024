import os

import sentencepiece as spm
import numpy as np
from collections import defaultdict

import nltk
from nltk.util import ngrams
from collections import Counter

nltk.download('punkt')

debug = False
###############################################
# generate Bi-Gram counter for training corpus
###############################################
corpus_text = '''I play tennis. I like Chinese friends. I talk with Chinese student. I play with tennis friends. I have friends who like tennis.'''
token = nltk.word_tokenize(corpus_text)

###########################
# Generate Bi-Gram counter
###########################
unigrams = Counter([w[0] for w in list(ngrams(token, 1))])
bigrams = Counter(list(ngrams(token, 2)))


###########################
# generate query Bi-Gram
###########################
query_text_1 = "I play with Chinese friends"
query_text_2 = "Chinese friends who like tennis"
query_token = nltk.word_tokenize(query_text_1)
query_bigram = list(ngrams(query_token, 2))

# DO NOT MODIFY ABOVE


if debug:
    print(token)
    print(unigrams)
    print(bigrams)
    print(query_bigram)

###########################
# TODO: lookup each query bigram in each query_text
# compute Uni-Counter[bg[0]] /  Bi-Counter[(bg[0],bg[1])]
# convert to PPL and output

# P(w2|w1) = P(w1,w2) / P(w1) = Bi-Counter[(w1,w2)] / Uni-Counter[w1]

# PPL   = (1 / (P(w2|w1) * P(w3|w2) * ... * P(wn|wn-1))) ** (1/n)
#       = ((Uni-Counter[w1] / Bi-Counter[(w1,w2)]) * (Uni-Counter[w2] / Bi-Counter[(w2,w3)]) * ... * (Uni-Counter[wn-1] / Bi-Counter[(wn-1,wn)])) ** (1/n)

prod = 1    # product of (Uni-Counter / Bi-Counter)
for bigram in query_bigram:
    prod *= unigrams[bigram[0]] / bigrams[bigram]
    
ppl = prod ** (1/len(query_bigram))
print(ppl)