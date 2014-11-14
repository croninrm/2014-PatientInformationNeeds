'''
Created on Oct 31, 2014

@author: croninrm
'''

import sys

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

'''
These are the input and output directories, check depending on your directory structure
'''
inputdirectory = 'C:\\Users\\croninrm\\Documents\\BMI-Fellow-PGY2\\Masters\\Data\\Messages\\ProcessedBetter\\CSV'
outputdirectory = 'C:\\Users\\croninrm\\Documents\\BMI-Fellow-PGY2\\Masters\\Data\\Messages\\ProcessedBetter\\Output'

#Number of topics
numTopics=5

from gensim import corpora, models, similarities
from gensim.models import LdaModel

import os
import csv
filetexts = []
filetexts2 =""
filesuper=""
os.chdir(inputdirectory)

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


os.chdir(outputdirectory)


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

lda = LdaModel(corpus, num_topics=numTopics)

doc_lda = []
for i in range(len(corpus)):
    doc_lda.append(lda[corpus[i]])
#print(doc_lda)

'''
This will simply put the tuples in a csv file, poor format
'''
with open('CorpusTopicsOld.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile)
    for i in range(len(corpus)):
        spamwriter.writerow(doc_lda[i])

'''
Here we split up the tuples into a proper csv file
With columns being the different topics
and the rows are the probabilities that this document
is in each of the topics 
'''
with open('CorpusTopics.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile)
    topicstr='topic1'
    for j in range(numTopics-1):
        topicstr+=',topic%d' % (j+2)
#    print topicstr
    spamwriter.writerow(topicstr.split(','))
    for i in range(len(corpus)):
        topicarray=[]
        topicids=[]
        for j in range(len(doc_lda[i])):
            topicids.append(doc_lda[i][j][0])   
 #           print topicids
        count=0
        for k in range(numTopics):           
            if k in topicids:
                topicarray.append(doc_lda[i][count][1])
                count+=1
            else:
                topicarray.append(0)
        #topics='%d,%d' % (topicid,doc_lda[i][1])
        spamwriter.writerow(topicarray)


print "Completed topic modeling"
sys.exit()
