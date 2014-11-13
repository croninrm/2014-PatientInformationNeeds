'''
Created on Nov 11, 2014

@author: croninrm
'''



import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities

import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold

import textmining 

import os
import csv
import sys
documents = []
outcomes=[]
os.chdir('C:\\Users\\croninrm\\Documents\\BMI-Fellow-PGY2\\Masters\\Data\\Messages\\ProcessedBetter\\CSV')
tdm = textmining.TermDocumentMatrix()

with open('LABELED_2008_175.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        documents.append(row[3])
        outcomes.append(row[4:10])
        # Add the documents
        tdm.add_doc(row[3])

# Write out the matrix to a csv file. Note that setting cutoff=1 means
# that words which appear in 1 or more documents will be included in
# the output (i.e. every word will appear in the output). The default
# for cutoff is 2, since we usually aren't interested in words which
# appear in a single document. For this example we want to see all
# words however, hence cutoff=1.
tdm.write_csv('matrix.csv', cutoff=1)
# Instead of writing out the matrix you can also access its rows directly.
# Let's print them to the screen.
textsall=[]

for row in tdm.rows(cutoff=1):
#    print row
    textsall.append(row)

stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]


#Create the dictionary from the words in texts, and save it
dictionary = corpora.Dictionary(texts)
dictionary.save('robCSVDictionary.dict') # store the dictionary, for future reference


#Create the corpus and save it
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('robCSVcorpus.mm', corpus) # store to disk, for later use

problemOutcomes=[int(row[0]) for row in outcomes[1:]]
managementOutcomes=[int(row[1]) for row in outcomes[1:]]
interventionsOutcomes=[int(row[2]) for row in outcomes[1:]]
testsOutcomes=[int(row[3]) for row in outcomes[1:]]
logisticsOutcomes=[int(row[4]) for row in outcomes[1:]]
ackthxOutcomes=[int(row[5]) for row in outcomes[1:]]

#print problemOutcomes
firsttexts=[row for row in textsall[1:]]

all_auc = []
print 'Positive Class:', sum(problemOutcomes)
cv = StratifiedKFold(problemOutcomes, n_folds=5, shuffle=True)




#def cross_val(self):


print '[+] Got CV...'
for i, (train, test) in enumerate(cv):
#    print "train = "
    trainsettexts=[firsttexts[row] for row in train]
    trainsetoutcomes=[problemOutcomes[row] for row in train]
    testsettexts=[firsttexts[row] for row in test]
    testsetoutcomes=[problemOutcomes[row] for row in test]
    
#    print len(trainsettexts) , train 
#    print len(trainsetoutcomes), test
    

    clf = MultinomialNB(alpha=0.1)
    probs = clf.fit(trainsettexts, trainsetoutcomes)
    probas_= probs.predict_proba(testsettexts)
    fpr, tpr, thresholds = metrics.roc_curve(testsetoutcomes, probas_[:, 1])



plt.plot(fpr, tpr)
#        title = 'plots/%s_%s_%d' % (self.__class__.__name__, self.c.__class__.__name__, self.vumc.args.num_patients)
title = "Naive Bayes"
plt.title(title)
plt.savefig('%s.png' % title)
roc_auc = metrics.auc(fpr, tpr)
all_auc.append(roc_auc)

plt.clf()


print 'Mean AUC:', np.mean(all_auc), all_auc


sys.exit()

