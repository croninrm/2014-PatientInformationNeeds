2014-PatientInformationNeeds
============================

Currently there are two  files, CINClassifier.py, and gensimExample.py:

CINClassifier.py: designed to take in a csv file which has 9 columns:
Columns:
1 - ID
2 - Date of message
3 - Message contents
4:9 - Outcomes including problems, test, management, logistics, and acknowledgement/thanks

Then it will split up the outcomes and the texts and form one matrix and multiple vectors
The matrix is the number of occurrences of each word for each document
The vectors are the different outcomes

It will then run a 10 fold cross validation with Multinomial Naive Bayes



gensimExample.py: designed to take in a file with the following columns
Columns:
1 - ID
2 - Date of message
3 - Message contents

The user can change the input and output directories and number of topics:
inputdirectory, outputdirectory, numTopics

Then it will take the message contents and run LDA for the number of topics on the documents
Then it goes back and looks at every document and creates a probability that it is in each topic.
