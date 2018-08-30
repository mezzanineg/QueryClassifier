# coding: utf-8
import json, operator, sys
from collections import Counter

faq=[]
prodotto=[]
ordiniaccountpersonali=[]

output=open('mfwMacrocat1.txt',"w")


with open('multitopic_labels.json') as datafile:
	data=json.load(datafile)

for element in data:
	if (element["clean_lemma"]!=""):
		print element["clean_lemma"]
		for l_w in element ["pos_and_lemma"]:
			print l_w
			if element["topic1"]=="FAQ":
				print element["topic1"]
				if l_w[1]!="nocat":
					faq.append(l_w[2])
			elif element["topic1"]=="Prodotto":
				print element ["topic1"]
				if l_w[1]!="nocat":
					prodotto.append(l_w[2])
			else:
				print element["topic1"]
				if l_w[1]!="nocat":
					ordiniaccountpersonali.append(l_w[2])


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

frequenza_termine_OrdiniAccountPersonali = Counter(ordiniaccountpersonali)
frequenza_termine_OrdiniAccountPersonali_clean = Counter(OrdiniAccountPersonali_clean)



for el in frequenza_termine_FAQ_clean.most_common(10):
	output.write(el[0]+"\n")
for el in frequenza_termine_Prodotto_clean.most_common(10):
	output.write(el[0]+"\n")
for el in frequenza_termine_OrdiniAccountPersonali_clean.most_common(10):
	output.write(el[0]+"\n")