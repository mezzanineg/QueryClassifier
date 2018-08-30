# coding: utf-8
import json, operator, sys
from collections import Counter

faq=[]
prodotto=[]
ordiniaccountpersonali=[]

output=open('mfwMacrocatRootsList.txt',"w")


with open('nlpOutput.json') as datafile:
	data=json.load(datafile)

for element in data:
	if (element["roots_list"]!=""):
		print element["roots_list"]
		#l_w=element["object_lemma"].split(" ")
		for l_w in element["roots_list"].split(" "):
			print l_w
			if element["topic1"] == "FAQ":
				faq.append(l_w)
			elif element["topic1"]=="Prodotto":
				prodotto.append(l_w)
			else:
				ordiniaccountpersonali.append(l_w)

FAQ_clean=[]
Prodotto_clean=[]
OrdiniAccountPersonali_clean=[]

for w_e in faq:
	if w_e not in prodotto and w_e not in ordiniaccountpersonali:
		FAQ_clean.append(w_e)

for w_e in prodotto:
	if w_e not in faq and w_e not in ordiniaccountpersonali:
		Prodotto_clean.append(w_e)

for w_e in ordiniaccountpersonali:
	if w_e not in prodotto and w_e not in faq:
		OrdiniAccountPersonali_clean.append(w_e)


frequenza_termine_FAQ = Counter(faq)
frequenza_termine_FAQ_clean = Counter(FAQ_clean)

frequenza_termine_Prodotto = Counter(prodotto)
frequenza_termine_Prodotto_clean = Counter(Prodotto_clean)
#
frequenza_termine_OrdiniAccountPersonali = Counter(ordiniaccountpersonali)
frequenza_termine_OrdiniAccountPersonali_clean = Counter(OrdiniAccountPersonali_clean)
#
#
#
for el in frequenza_termine_FAQ_clean.most_common(10):
	output.write(el[0]+"\n")
for el in frequenza_termine_Prodotto_clean.most_common(10):
	output.write(el[0]+"\n")
for el in frequenza_termine_OrdiniAccountPersonali_clean.most_common(10):
	output.write(el[0]+"\n")