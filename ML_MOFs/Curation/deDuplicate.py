#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 2023

@author: pcyic2

Script to read in cifs and write to a file Weisfeihler-Lehman graph hashes of each based on their structure graphs
With the purpose of deduplicaiton.
Reads from a list of cifs in a file called mofs_todedup.txt and takes cif files from a directory - 
needs to be specified by replacing path\to\cifs with the path to the cifs' location.

"""

import pymatgen
from pymatgen.core import Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CutOffDictNN
import networkx as nx
from pymatgen.util.graph_hashing import weisfeiler_lehman_graph_hash
from pymatgen.analysis.local_env import CrystalNN

with open('../mofs_todedup.txt','r') as f:
    todedup  = f.readlines()

for i in range(len(todedup)):
    todedup[i] = todedup[i].replace('\n','')

hashes = []
for struct in todedup:
    structure = Structure.from_file('path\to\cifs/{0}'.format(struct))
#    print(structure)
    prim = structure.get_primitive_structure() 
#    print(prim)
    sgraph=(StructureGraph.with_local_env_strategy(prim,CutOffDictNN.from_preset('vesta_2019')))
#    print('sgraph is {0}'.format(sgraph))
#    print('sgraph.graph is {0}'.format(sgraph.graph))
    wlhash = weisfeiler_lehman_graph_hash(sgraph.graph)
    print('{0} hash: {1}'.format(struct,wlhash))
#    print(sgraph.graph.nodes)
    hashes.append(wlhash)

with open('hashes.txt','w') as f:
    for i in range(len(todedup)):
        f.write('{0} {1}\n'.format(todedup[i],hashes[i]))

uniquenames = []
uniquehashes = []
for i in range(len(todedup)):
    if hashes[i] not in uniquehashes:
        uniquehashes.append(hashes[i])
        uniquenames.append(todedup[i])

with open('uniquehashes.txt','w') as f:
    for i in range(len(uniquehashes)):
        f.write('{0} {1}\n'.format(uniquenames[i],uniquehashes[i]))


