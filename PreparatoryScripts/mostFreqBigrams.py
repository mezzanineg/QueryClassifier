# coding: utf-8
import nltk
from nltk.util import ngrams
from itertools import groupby
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import json, operator, sys
from collections import Counter

faq=[]
prodotto=[]
ordiniaccountpersonali=[]

output=open('bigramsTOTtoTest.txt',"w")


with open('multitopic_labels.json') as datafile:
	data=json.load(datafile)

bigrams=[]
for element in data:
	if (element["clean_lemma"]!=""):
		text=element["clean_lemma"]
		print text
		token = nltk.word_tokenize(text)
		bigrams = ngrams(token, 2)
		b1=list(bigrams)
		print b1
		if element["topic1"] == "FAQ":
			faq.append(b1)
			print "AGGIUNTO A FAQ"
		if element["topic1"]=="Prodotto":
			prodotto.append(b1)
			print "AGGIUNTO A Prodotto"
		if element["topic1"]=="OrdiniAccountPersonali":
			ordiniaccountpersonali.append(b1)
			print "AGGIUNTO A OrdiniAccountPersona"

print "--------------------------"
#print prodotto
prodotto_lista=[]
for i in prodotto:
	for element in i:
		prodotto_lista.append(element)

faq_lista=[]
for i in faq:
	for element in i:
		faq_lista.append(element)


ordiniaccountpersonali_lista = []
for i in ordiniaccountpersonali:
	for element in i:
		ordiniaccountpersonali_lista.append(element)


prodotto_l_cleaned=[]
for i in prodotto_lista:
	#print i
	if i not in faq_lista:
		#print i
		prodotto_l_cleaned.append(i)
print Counter(prodotto_l_cleaned)


faq_l_cleaned=[]
for i in faq_lista:
	#print i
	if i not in  prodotto_lista:
		#print i
		faq_l_cleaned.append(i)
print Counter(faq_l_cleaned)

ordiniaccountpersonali_l_cleaned=[]
for i in ordiniaccountpersonali_lista:
#print i
	if i not in faq_lista and i not in prodotto_lista:
# 		#print i
		ordiniaccountpersonali_l_cleaned.append(i)
print Counter(ordiniaccountpersonali_l_cleaned)
#

# for el in frequenza_termine_Prodotto_clean.most_common(10):
# 	output.write(el[0]+"\n")
# for el in frequenza_termine_OrdiniAccountPersonali_clean.most_common(10):
# 	output.write(el[0]+"\n")