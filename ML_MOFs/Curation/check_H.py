#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:49:43 2022

@author: pcyic2

Script to read in cifs and check for any which do not contain H atoms.
Writes to files a list of cifs with no H and a list of cifs containing H (passed cifs)
Takes as an argument the directory containing the cifs to be checked.

"""

import os
import re
import argparse
#import matplotlib.pyplot as plt

# List all metals and elements --------------------------

s= ''
metals = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr', \
          'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra', \
          'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', \
          'Y', 'Zr', 'Nb', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', \
          'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', \
          'Rf', 'Db', 'Sg', 'Bh', 'Hs', \
          'Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'Po', 'At', \
          'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', \
          'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', \
          'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
elements = ['H','He','Li','Be','B','C','N','O','F','Ne', \
            'Na','Mg','Al','Si','P','S','Cl','Ar', \
            'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn', \
            'Ga','Ge','As','Se','Br','Kr', \
            'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd', \
            'In','Sn','Sb','Te','I','Xe', \
            'Cs','Ba', \
            'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu', \
            'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg', \
            'Tl','Pb','Bi','Po','At','Rn', \
            'Fr','Ra', \
            'Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr', \
            'Rf','Db','Sg','Bh','Hs', \
            'Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og']

# Get argumnets    
parser = argparse.ArgumentParser()                                                                                           
# Input directory which contains the cifs as an argument
parser.add_argument('directory',type=str)
args = parser.parse_args()

path = os.getcwd()

# Read in names of cifs
entries = sorted(os.listdir(path + '/' + args.directory))

MOFatomslist = []
MOFsnoH = []
MOFnamesnoH = []

for entryno in range(len(entries)):
#    print(entries[entryno])
# Make variable of the cif
    with open(path + '/' + args.directory + '/' + entries[entryno], 'r') as f:
        cif = f.readlines()
# Make list of atoms in MOF
    mofatoms = []
    for i in range(len(cif)):
        if len(cif[i].split()) > 1:
            if re.split('(\d+)',cif[i].split()[0])[0] in elements:
                mofatoms.append(re.split('(\d+)',cif[i].split()[0])[0])
# Make a list of unique atoms in MOF
    uniqueatoms = list(dict.fromkeys(mofatoms))
    MOFat = [entries[entryno]]
    for element in uniqueatoms:
        MOFat.append(element)
    MOFatomslist.append(MOFat)      # Make list of unique atoms in all MOFs
    if 'H' not in uniqueatoms:
        MOFsnoH.append(MOFat)
        MOFnamesnoH.append(MOFat[0])
        

# Count number of cifs with no H
print('There are {0} cif(s) with no hydrogen'.format(len(MOFsnoH)))


# Print list of no H cifs to file
f = open('MOFs_noH.txt','w')
for mof in MOFsnoH:
    for item in mof:
        f.write('{0}  '.format(item))
    f.write('\n')
f.close()


# Make list of all MOFs that do contain H
entries_withH = []
for entry in entries:
    if entry not in MOFnamesnoH:
        entries_withH.append(entry)
#Print list of MOFs not in the categories to a file
f = open('cifswithH.txt','w')
for mof in entries_withH:
    f.write('{0}'.format(mof))
    f.write('\n')
f.close()
