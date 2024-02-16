#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 10:59:15 2023

@author: pcyic2

Script to read in outputs ffrom the MOSAEC code and write to files 
a list of cifs with no detected issues and a list of cifs with detected issues.

"""

import pandas as pd

# Read in data
MOSres = pd.read_csv('OxStatesOutput.csv')
MOSres = MOSres.drop('Unnamed: 0',axis=1)

# Make variable of only rows where all are good
MOSresGOOD = MOSres.loc[(MOSres['Impossible'] == 'GOOD') &\
                        (MOSres['Unknown'] =='GOOD') &\
                        (MOSres['Zero_Valent'] == 'GOOD') &\
                        (MOSres['noint_flag'] == 'GOOD') &\
                        (MOSres['low_prob_1'] == 'GOOD') &\
                        (MOSres['low_prob_2'] == 'GOOD') &\
                        (MOSres['low_prob_3'] == 'GOOD') &\
                        (MOSres['low_prob_multi'] == 'GOOD') &\
                        (MOSres['high_count'] == 'GOOD') &\
                        (MOSres['low_count'] == 'GOOD')]

# Make variable of only rows where not all are good
MOSresBAD = MOSres.loc[~((MOSres['Impossible'] == 'GOOD') &\
                        (MOSres['Unknown'] =='GOOD') &\
                        (MOSres['Zero_Valent'] == 'GOOD') &\
                        (MOSres['noint_flag'] == 'GOOD') &\
                        (MOSres['low_prob_1'] == 'GOOD') &\
                        (MOSres['low_prob_2'] == 'GOOD') &\
                        (MOSres['low_prob_3'] == 'GOOD') &\
                        (MOSres['low_prob_multi'] == 'GOOD') &\
                        (MOSres['high_count'] == 'GOOD') &\
                        (MOSres['low_count'] == 'GOOD'))]
                         
# Make list of all the bad cifs
badlist = []
for cif in MOSresBAD['CIF']:
    if cif not in badlist:          # Avoid repeats
        badlist.append(cif)

# Make list of all the good cifs
goodlist = []
for cif in MOSresGOOD['CIF']:
    if cif not in goodlist and cif not in badlist:      # Avoid repeats and don't include cifs that are in the bad list
        goodlist.append(cif)                        

# Write good and bad metals to excel
MOSresGOOD.to_excel('good_metals.xlsx')
MOSresBAD.to_excel('bad_metals.xlsx')
        
# Write good and bad cifs to file
with open('good_cifs.txt','w') as f:
    for cif in goodlist:
        f.write('{0}\n'.format(cif))
        
with open('bad_cifs.txt','w') as f:
    for cif in badlist:
        f.write('{0}\n'.format(cif))
