#!/bin/bash

cifs='/home/pcyic2/to_download/ML_biogas/for_github/phase2check/6638_cifs'

grep 'D ' $cifs/* > DMOFs.txt	
grep 'D[0-9]' $cifs/* > DMOFs.txt
