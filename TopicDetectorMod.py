# -*- coding: utf-8 -*-

import json, operator, sys
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from nltk.corpus import wordnet as wn
from collections import Counter
import nltk
from nltk.util import ngrams
import codecs

whlist = ['quando', 'dove', 'perch', 'quanto', 'quanti', 'quante', 'come', 'com\'', 'cosa']


def checkWHList(line, query, whlist):
	line['quando'] = 0
	line['dove'] = 0
	line['perch'] = 0
	line['quanto'] = 0
	line['come'] = 0
	line['cosa'] = 0

	for whword in whlist:
		if whword in query:
			if 'quant' in whword:
				line["quanto"] = 1
			elif 'com' in whword:
				line["come"] = 1
			else:
				line[whword] = 1


def getPOS(pal, pos):
	Number = 0
	for l in pal:
		Number = Number + l.count(pos)
	return Number


def creaTTR(line, querylemmatized):
	ttr_list = []
	for e in querylemmatized.split(" "):
		ttr_list.append(e)
	line["TTR"] = len(Counter(ttr_list)) * 1.0 / len(querylemmatized.split(" ")) * 1.0


def wordSimilarity(word1, word2):
	listaSyns1 = wn.synsets(word1, lang='ita')
	listaSyns2 = wn.synsets(word2, lang='ita')
	if not listaSyns1 or not listaSyns2:
		return 0
	else:
		maxSim = 0.72
		for element in listaSyns1:
			for element2 in listaSyns2:
				similarity = element.wup_similarity(element2)
				if similarity is None:
					similarity = 0
				if similarity > maxSim:
					maxSim = round(similarity, 3)
	return maxSim


def cercaFrequentWords(line, querylemmatized, lista):
	for mfw in lista:
		if mfw in querylemmatized:
			line[mfw] = 1
		else:
			line[mfw] = 0


def cercaBigrammi(line, querylemmatized, lista):
	# print (querylemmatized)
	for mfb in lista:
		if mfb in querylemmatized:
			line[mfb] = 1
		else:
			line[mfb] = 0

def freqWordsWeight(line,querylemmatized,lista,querylength):
	paroleinquery=0
	for mfw in lista:
		if mfw in querylemmatized:
			paroleinquery=paroleinquery+1
		else:
			paroleinquery=paroleinquery
	peso=round(paroleinquery/querylength,3)
	line["pesoFeat"]=peso


def freqObjectWordsWeight(line,querylemmatized,lista,querylength):
	paroleinquery=0
	for mfw in lista:
		if mfw in querylemmatized:
			paroleinquery=paroleinquery+1
		else:
			paroleinquery=paroleinquery
	peso=round(paroleinquery/querylength,3)
	line["pesoObjectWords"]=peso

def freqBigramsWeight(line,querylemmatized,lista,querylength):
	#print ("Lunghezza: "+str(querylength))
	#print (querylemmatized)
	paroleinquery=0
	for mfw in lista:
		if mfw in querylemmatized:
			paroleinquery=paroleinquery+1
		else:
			paroleinquery=paroleinquery
	if querylength>0:
		peso=round(paroleinquery/querylength,3)
		line["pesoBigrams"]=peso
		#print ("PESO BIGRAMMA: "+str(line["pesoBigrams"]))


def cercaSinonimi(line, querylemmatized, lista_etichette):
	querysplitted = element["clean_lemma"].split(" ")
	for etichetta in lista_etichette:
		etichetta_splitted = etichetta.split(" ")
		querysim = 0
		for es in etichetta_splitted:
			for wq in querysplitted:
				querysim = querysim + wordSimilarity(es, wq)
		line[etichetta] = querysim


# importing the standard json file
with open('NEWTRAIN.json') as data_file:
	data = json.load(data_file)

most_frequent_words = []

with codecs.open('mfwMacrocat1.txt', "r", "utf-8") as mfw:
	for line in mfw:
		most_frequent_words.append(line.strip())

most_frequent_bigrams = []
with codecs.open('mfBigrams.txt', "r", "utf-8") as mfb:
    for line in mfb:
        most_frequent_bigrams.append(line.strip())

lista_etichette=['FAQ','Prodotto','OrdiniAccountPersonali']


#getting corpus and labels
corpus = []
feat_dict=[]
line={}
label = []
for element in data:
	corpus.append(element["clean_lemma"])
	#line["lunghezza"]=len(element["clean_lemma"].split(" "))
	#line["emptywords"]=len(element["query"])-len(element["clean_lemma"])
	line["question_mark"]=element["query"].count('?')
	pal=element["pos_and_lemma"]
	line["verbNumber"]=getPOS(pal,"ver")
	line["noCat"]=getPOS(pal,"nocat")
	line["advNumber"]=getPOS(pal,"adv")
	line["adjNumber"]=getPOS(pal,"adj")
	line["npr"]=getPOS(pal,"npr")*1.0/len(element["query"])
	line["noCatpercent"]=round(getPOS(pal,"nocat")*1.0/len(element["query"])*1.0,3)
	token = nltk.word_tokenize(element["clean_lemma"])
	bigrams = ngrams(token, 2)
	b1 = list(bigrams)
	creaTTR(line,element["clean_lemma"])
	cercaFrequentWords(line,element["clean_lemma"],most_frequent_words)
	freqWordsWeight(line, element["clean_lemma"], most_frequent_words, len(element["clean_lemma"].split(" ")))
	freqBigramsWeight(line, element["clean_lemma"], most_frequent_bigrams, len(b1))
	cercaBigrammi(line, element["clean_lemma"], most_frequent_bigrams)
	#cercaSinonimi(line,element["clean_lemma"],lista_etichette)
	checkWHList(line,element["query"],whlist)
	feat_dict.append(line)
	line={}
	label.append(element["topic"])

#tf-idf sui bigrammi-
vec = DictVectorizer()
length_matrix = vec.fit_transform(feat_dict).toarray()
lunghezzaMatrice=np.array(length_matrix)
feature_names_vettore = vec.get_feature_names()
#print ("Features Name Vettore: " + str(feature_names_vettore))

tfidf_vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1,2))
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus).A
feature_names_tfidf = tfidf_vectorizer.get_feature_names()
#print ("Features Name tfidf: "+ str(feature_names_tfidf))
pipelineMatrice=np.array(tfidf_matrix)
feature_names_globale=feature_names_vettore+feature_names_tfidf
X = np.concatenate((lunghezzaMatrice,pipelineMatrice), axis=1)
#X = np.array(lunghezzaMatrice)
X = sparse.csr_matrix(X).toarray()
y = np.array(label)


# e qui abbiamo la svm - e' un classificatore come l'altro che abbiamo - ma questo fa multilabel classification, quindi per ogni classe ci da' un livello di confidence
#feature_names = tfidf_vectorizer.get_feature_names()
classifier = OneVsRestClassifier(svm.SVC(kernel = "linear", C=1, random_state=0, probability = True)).fit(X, y)
classes = classifier.classes_.tolist()

outfile1 = open('ALL_queries_tagged_BALANCED_NEWTRAIN_SecondRUN_090.txt',"w")

def classify(path_query):
	with open(path_query, 'r') as fp:
		data = json.load(fp)
		for element in data:
			new_vector = [0 for x in range(len(feature_names_globale))]
			for word in element["clean_lemma"].split(" "):
				try:
					new_vector[feature_names_globale.index(word)] = 1
				except Exception:
					pass
			new_vector=np.array(new_vector)
			Y_proba = classifier.predict_proba(new_vector.reshape(1, -1))
			dict_proba = {classes[k]: Y_proba[0][k] for k in range(len(classes))}


			for key, value in dict_proba.items():
				if key =="OrdiniAccountPersonali" and value > 0.90:
					outfile1.write(element["query"].strip()+"\t"+key+"\n")
				if key =="FAQ" and value > 0.90:
					outfile1.write(element["query"].strip() + "\t" + key + "\n")
				if key =="Prodotto" and value > 0.90:
					outfile1.write(element["query"].strip() + "\t" + key + "\n")
			#print ("\n")


			#print (sorted(dict_proba.items(), key=operator.itemgetter(1), reverse=True))

if __name__ == "__main__":
	classify(sys.argv[1])
else:
	pass