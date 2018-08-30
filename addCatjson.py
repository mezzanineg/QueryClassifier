# coding: utf-8
import sys
import random
import operator
from collections import Counter
from collections import OrderedDict
import re
import string
import json, operator

outfile1 = open('TAGGEDTouse.json',"w")

# lista_appoggio = list()
# count=0

# Empty dict
d = {}
with open("training.tsv") as dict:
	for di in dict:
		#print di
		#print di.split("\",\"")[0].strip()
		d[di.split("\t")[0].strip()]=di.split("\t")[1].strip()
# Fill in the entries one by one
my_dict=[]
with open("Training_nocat.json", 'r') as queries:
	data = json.load(queries)
	print data
	for element in data:
		#print element["query"]
		if d.has_key(element["query"]):
			print "MATCH TROVATO!"
			my_dict.append(element)
			my_dict.append (", u\'topic\':"+d[element["query"]])


for el in my_dict:
	print (el)
	json.dump(el, outfile1, indent=4, separators=(',', ': '))
outfile1.close()

# 			q= line.split("\":\"")[1]
# 			q1= string.replace(line.split("\":\"")[1], '\",', '')
# 			q1=q1.strip()
# 			#print "CELANED: "+q1+"\n"
# 			if (q1 in d):
# 				linenew=line+"\"topic1\": \""+d[q1].split('.')[0]+"\",\n"+"\"topic2\": \""+d[q1].split('.')[1]+"\",\n"
# 				outfile.write(linenew)
# 		else:
# 			outfile.write(line)