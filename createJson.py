# coding: utf-8
import sys
import random
import operator
from collections import Counter
from collections import OrderedDict

outfile1 = open('trainedNOCAT.json',"w")

# lista_appoggio = list()
# count=0

outfile1.write("[")

with open("Training_nocat.tsv") as queries:
    for line in queries:
	outfile1.write("\n\t{\n\t\t\"input\" : \""+line.split("\t")[0]+"\", \"annotation\" : \""+line.split("\t")[1].strip()+"\"\n\t},")

outfile1.write("]")
       # print line
