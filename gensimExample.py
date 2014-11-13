'''
Created on Oct 31, 2014

@author: croninrm
'''
import sys
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities
from gensim.models import LdaModel

import os
import csv
filetexts = []
filetexts2 =""
filesuper=""
os.chdir('C:\\Users\\croninrm\\Documents\\BMI-Fellow-PGY2\\Masters\\Data\\Messages\\ProcessedBetter\\CSV')

'''
First I will need to loop through all the files to make a dictionary
Then I can go through each file, and take each document and see its score...

'''

#107 labeled documents
# COLUMNS ARE: ID, DATE, MESSAGE, CATEGORY (one or more)

with open('LABELED_2008_175.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    filetexts2=[word[3] for word in spamreader]
     
with open('LABELED_2008_175.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    filetextslabeled=[word for word in spamreader]


documents=filetexts2
# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]


#Create the dictionary from the words in texts, and save it
dictionary = corpora.Dictionary(texts)
dictionary.save('robCSVDictionary.dict') # store the dictionary, for future reference

#Create the corpus and save it
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('robCSVcorpus.mm', corpus) # store to disk, for later use
#import numpy as np
#corpusnp=np.array(corpus)
#print len(corpusnp), len(np.delete(corpusnp, 1, axis=0))
#Initialize the transformation
#term freq inverse doc freq
#trying to ind the frequency on that page versus overall frequency

lda = LdaModel(corpus, num_topics=5)
print corpus[1]
doc_lda = lda[corpus[50]]
print(doc_lda)




sys.exit()
