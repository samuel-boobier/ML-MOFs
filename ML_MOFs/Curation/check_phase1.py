#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:49:43 2022

@author: pcyic2

Script to read in cif files from a specified directory and determine whether they have
No metal atom, no carbon atom, one element, two elements or none of these problematic features
Writes to files lists of all cifs with each problematic feature
And a list of all cifs with none of them (passed cifs)
Takes the directory containing the cifs as an argument

"""

import os
import re
import argparse

# List all metals and elements --------------------------

s= ''
metals = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr', \
          'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra', \
          'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', \
          'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', \
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
# Input directory containing the cifs as an argument
parser.add_argument('directory',type=str)
args = parser.parse_args()

path = os.getcwd()

# Read in names of cifs
entries = sorted(os.listdir(path + '/' + args.directory))

MOFatomslist = []
MOFsoneatom = []
MOFstwoatom = []
MOFsnoC = []
MOFsnomet = []

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
    if len(uniqueatoms) == 1:
        MOFsoneatom.append(MOFat)
    if len(uniqueatoms) == 2:
        MOFstwoatom.append(MOFat)
    if 'C' not in uniqueatoms:
        MOFsnoC.append(MOFat)
    metcount = 0
    for atom in uniqueatoms:
        if atom in metals:
            metcount += 1
    if metcount == 0:
        MOFsnomet.append(MOFat)

# Count number of cifs with one and two elements, with no C, and with no metals
print('There are {0} cif(s) with only one element'.format(len(MOFsoneatom)))
print('There are {0} cif(s) with only two elements'.format(len(MOFstwoatom)))
print('There are {0} cif(s) with no carbon'.format(len(MOFsnoC)))
print('There are {0} cif(s) with no metal'.format(len(MOFsnomet)))

# Make list of all cifs in all four categories
MOFsallchecks = MOFsoneatom + MOFstwoatom + MOFsnoC + MOFsnomet
MOFsallcheckstup = [tuple(cif) for cif in MOFsallchecks]
uniquecifsalltup = list(dict.fromkeys(MOFsallcheckstup))
uniquecifsalllist = [list(cif) for cif in uniquecifsalltup]
uniquecifsalllistnames = [cif[0] for cif in uniquecifsalllist]

# Count number of cifs in all categories
print('There are {0} cif(s) with one or two elements, and/or no Carbon or no metal'\
      .format(len(uniquecifsalllist)))


# Print list of one elememt cifs to file
f = open('MOFs_oneatom.txt','w')
for mof in MOFsoneatom:
    for item in mof:
        f.write('{0}  '.format(item))
    f.write('\n')
f.close()
# Print list of two element cifs to file
f = open('MOFs_twoatom.txt','w')
for mof in MOFstwoatom:
    for item in mof:
        f.write('{0}  '.format(item))
    f.write('\n')
f.close()
# Print list of cifs with no carbon to file
f = open('MOFs_noC.txt','w')
for mof in MOFsnoC:
    for item in mof:
        f.write('{0}  '.format(item))
    f.write('\n')
f.close()
# Print list of cifs with no metal to file
f = open('MOFs_nomet.txt','w')
for mof in MOFsnomet:
    for item in mof:
        f.write('{0}  '.format(item))
    f.write('\n')
f.close()

# Print list of all identified cifs to files
# Name and atoms
f = open('MOFsatoms_fourcategories.txt','w')
for mof in uniquecifsalllist:
    for item in mof:
        f.write('{0}  '.format(item))
    f.write('\n')
f.close()
#Name only
f = open('MOFs_fourcategories.txt','w')
for mof in uniquecifsalllist:
    f.write('{0}'.format(mof[0]))
    f.write('\n')
f.close()

# Make list of all MOFs other than those in the four categories
entries_fourcatsremoved = []
for entry in entries:
    if entry not in uniquecifsalllistnames:
        entries_fourcatsremoved.append(entry)
#Print list of MOFs not in the categories to a file
f = open('passed_cifs.txt','w')
for mof in entries_fourcatsremoved:
    f.write('{0}'.format(mof))
    f.write('\n')
f.close()
