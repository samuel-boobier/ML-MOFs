#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:38:48 2022

@author: pcyic2
"""

import os
import re
import argparse
#import matplotlib.pyplot as plt
import math
import sys

elements = ['H','D','He','Li','Be','B','C','N','O','F','Ne', \
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
# Input directory as an argument
parser.add_argument('directory',type=str)
args = parser.parse_args()

path = os.getcwd()

# Read in names of cifs
entries = sorted(os.listdir(path + '/' + args.directory))

low_dists = []

for entryno in range(len(entries)):
#    print(entries[entryno])
# Make variable of the cif
    print(entries[entryno])
    with open(path + '/' + args.directory + '/' + entries[entryno], 'r') as f:
        cif = f.readlines()
# Find place to stop reading in lines of the cif
    stopread = len(cif)
    for i in range(len(cif)):
        if 'loop' in cif[i] and len(cif[i-1]) > 1:
            if re.split('(\d+)',cif[i-1].split()[0])[0] in elements:
                stopread = i
# Get cell parameters
    cell_a = 0
    cell_b = 0
    cell_c = 0
    cell_alpha = 0
    cell_beta = 0
    cell_gamma = 0
    for line in cif:
        if len(line.split()) > 0:
            if line.split()[0] == '_cell_length_a':
                cell_a = float(re.sub('\(.*?\)','',line.split()[1]))
            elif line.split()[0] == '_cell_length_b':
                cell_b = float(re.sub('\(.*?\)','',line.split()[1]))
            elif line.split()[0] == '_cell_length_c':
                cell_c = float(re.sub('\(.*?\)','',line.split()[1]))
            elif line.split()[0] == '_cell_angle_alpha':
                cell_alpha = float(re.sub('\(.*?\)','',line.split()[1]))
            elif line.split()[0] == '_cell_angle_beta':
                cell_beta = float(re.sub('\(.*?\)','',line.split()[1]))
            elif line.split()[0] == '_cell_angle_gamma':
                cell_gamma = float(re.sub('\(.*?\)','',line.split()[1]))                
    if cell_a == 0 or cell_b == 0 or cell_c == 0 or \
    cell_alpha == 0 or cell_beta == 0 or cell_gamma == 0:
        print('Couldn\'t get cell parameters')
        sys.exit()
# Make list of atoms in MOF and their coordinates
    mofatoms = []
    for i in range(stopread):
        if len(cif[i].split()) > 3:
            if re.split('(\d+)',cif[i].split()[0])[0] in elements:
                print(cif[i].split())
                mofatoms.append([cif[i].split()[j] for j in range(1,5)])
# Remove numbers in brackets
    for i in range(len(mofatoms)):
        for j in range(len(mofatoms[i])):
            mofatoms[i][j]=re.sub('\(.*?\)','',mofatoms[i][j])
# Convert coordinates to float
        for k in range(1,4):
            mofatoms[i][k] = float(mofatoms[i][k])
# Calculate distances between each atom pair
    dists = []
    for i in range(len(mofatoms)):
        for j in range (len(mofatoms)):
            if j != i:
                deltax = mofatoms[i][1]-mofatoms[j][1]
                deltay = mofatoms[i][2]-mofatoms[j][2]
                deltaz = mofatoms[i][3]-mofatoms[j][3]
                cosalph = math.cos(math.radians(cell_alpha))
                cosbet = math.cos(math.radians(cell_beta))
                cosgam = math.cos(math.radians(cell_gamma))
                distijsq = (cell_a*deltax)**2 + \
                        (cell_b*deltay)**2 + \
                        (cell_c*deltaz)**2 + \
                        (2*cell_b*cell_c*cosalph*deltay*deltaz) + \
                        (2*cell_c*cell_a*cosbet*deltaz*deltax) + \
                        (2*cell_a*cell_b*cosgam*deltax*deltay)

                dists.append(math.sqrt(distijsq))
    
    if min(dists) < 0.5:
        low_dists.append([entries[entryno],min(dists)])
        
with open('Lessthan0.5distances.txt','w') as f:
    for cif in low_dists:
        f.write('{0}  {1}\n'.format(cif[0],cif[1]))
with open('Lessthan0.5.txt','w') as f:
    for cif in low_dists:
        f.write('{0}\n'.format(cif[0]))
        
# Make a list of cifs which are not excluded        
lowdist_names = []
for cif in low_dists:
    lowdist_names.append(cif[0])
nottooclose = []
for entry in entries:
    if entry not in lowdist_names:
        nottooclose.append(entry)
with open('passed_cifs.txt','w') as f:
    for cif in nottooclose:
        f.write('{0}\n'.format(cif))
