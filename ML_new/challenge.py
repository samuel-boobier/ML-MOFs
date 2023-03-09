import os, sys, re

reglar_expression = re.compile("\d\d\d") # for 3 numbers

import numpy as np

list = np.arange(2,100,2)
print(list)

# too many lines



ls = []

for i in range(100):
    if i < 2:
        continue
    if i % 2 == 0:
        ls.append(i)

print(ls)

x = None

if x == None:
    print("yes")

# functions? objects? libraries?
workbook_id = ", Workbook[13]"
workbook_id = re.findall("\d+", workbook_id)[0]
print(workbook_id)
