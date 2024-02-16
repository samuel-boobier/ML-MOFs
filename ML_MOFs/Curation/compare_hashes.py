# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 23:34:47 2023

@author: isabe

Script to read in lists of cif names and corresponding Weisfeiler-Lehman structure
graph hashes for the training set (from CSD) and the test set (from Northwestern Hypotheticals)
Then to compare hashes and names and write to files lists of unique MOFs from the training set and
unique MOFs from the test set which also are not duplicated in the training set.

"""
import re

with open('hashes_train.txt','r') as f:
    train = f.readlines()
    
with open('hashes_test.txt','r') as f:
    test = f.readlines()

trainhash = []
trainhash_nohashrep = []
trainname_nohashrep = []
trainuniquehash = []
trainuniquename = []
for i in range(len(train)):
    train[i] = train[i].split()
    trainhash.append(train[i][1])
    if train[i][1] not in trainhash_nohashrep:
        trainhash_nohashrep.append(train[i][1])
        trainname_nohashrep.append(train[i][0])

trainname_nonum = []
trainuniquename_nonum = []
for i in range(len(trainname_nohashrep)):
    trainname_nonum.append(re.sub(r'[0-9]','',trainname_nohashrep[i]))
    if trainname_nonum[i] not in trainuniquename_nonum:
        trainuniquename.append(trainname_nohashrep[i])
        trainuniquehash.append(trainhash_nohashrep[i])
        trainuniquename_nonum.append(trainname_nonum[i])

testhash = []
testuniquehash = []
testuniquename = []
testnotintrainhash = []
testnotintrainname = []
for i in range(len(test)):
    test[i] = test[i].split()
    testhash.append(test[i][1])
    if test[i][1] not in testuniquehash:
        testuniquehash.append(test[i][1])
        testuniquename.append(test[i][0])
        if test[i][1] not in trainuniquehash:
            testnotintrainhash.append(test[i][1])
            testnotintrainname.append(test[i][0])
            
with open('uniqueMOFstrain.txt','w') as f:
    for i in range(len(trainuniquename)):
        f.write('{0} {1}\n'.format(trainuniquename[i],trainuniquehash[i]))
        
with open('uniqueMOFstest.txt','w') as f:
    for i in range(len(testnotintrainname)):
        f.write('{0} {1}\n'.format(testnotintrainname[i],testnotintrainhash[i]))
    
