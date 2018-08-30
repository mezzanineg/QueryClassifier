# coding: utf-8

import json, operator, sys
import numpy as np
import nltk
from nltk.util import ngrams
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn import svm
from sklearn import cross_validation
from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import confusion_matrix
from nltk.corpus import wordnet as wn
import pylab as plt
from collections import Counter
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
		maxSim = 0.90
		for element in listaSyns1:
			for element2 in listaSyns2:
				similarity = element.wup_similarity(element2)
				if similarity is None:
					similarity = 0
				if similarity > maxSim:
					maxSim = round(similarity, 3)
	return maxSim

def cercaFrequentWords(line,querylemmatized,lista):
	for mfw in lista:
		if mfw in querylemmatized:
			line[mfw]=1
		else:
			line[mfw]=0

def freqWordsWeight(line,querylemmatized,lista,querylength):
	paroleinquery=0
	for mfw in lista:
		if mfw in querylemmatized:
			paroleinquery=paroleinquery+1
		else:
			paroleinquery=paroleinquery
	peso=round(paroleinquery/querylength,3)
	line["WordsWeight"]=peso

def freqObjectWordsWeight(line,querylemmatized,lista,querylength):
	paroleinquery=0
	for mfw in lista:
		if mfw in querylemmatized:
			paroleinquery=paroleinquery+1
		else:
			paroleinquery=paroleinquery
	peso=round(paroleinquery/querylength,3)
	line["pesoObjectWords"]=peso


def cercaBigrammi(line,querylemmatized,lista):
    #print (querylemmatized)
    for mfb in lista:
        if mfb in querylemmatized:
            line[mfb] = 1
        else:
            line[mfb] = 0

def freqBigramsWeight(line,querylemmatized,lista,querylength):
	print ("Lunghezza: "+str(querylength))
	print (querylemmatized)
	paroleinquery=0
	for mfw in lista:
		if mfw in querylemmatized:
			paroleinquery=paroleinquery+1
		else:
			paroleinquery=paroleinquery
	if querylength>0:
		peso=round(paroleinquery/querylength,3)
		line["bigramsWeight"]=peso
		print ("PESO BIGRAMMA: "+str(line["bigramsWeight"]))






def cercaSinonimi(line,querylemmatized,lista_etichette):
    querysplitted=element["clean_lemma"].split(" ")
    for etichetta in lista_etichette:
        etichetta_splitted=etichetta.split(" ")
        querysim=0
        for es in etichetta_splitted:
            for wq in querysplitted:
                querysim=querysim+wordSimilarity(es,wq)
        line[etichetta]=querysim

def most_informative_feature_for_class(vectorizer, classifier, classlabel, n=12):
	labelid = list(classifier.classes_).index(classlabel)
	feature_names = vectorizer.get_feature_names()
	topn = sorted(zip(classifier.coef_[labelid], feature_names))[-n:]
	return topn
	#for coef, feat in topn:
		#print (classlabel, feat, coef)


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




# most_frequent_object_lemma=[]
# with open (sys.argv[3],"r") as mfol:
# 	most_frequent_object_lemma.append(line.strip())
#
# most_frequent_roots=[]
# with open (sys.argv[4],"r") as mfr:
# 	most_frequent_roots.append(line.strip())

#getting corpus and labels
lista_etichette=['FAQ','Prodotto', 'OrdiniAccountPersonali']
corpus = []
feat_dict=[]
line={}
label = []

#adding features lines
for element in data:
	if element["clean_lemma"]!="":
		corpus.append(element["clean_lemma"])
		#line["lunghezza"] = len(element["clean_lemma"].split(" "))
		#line["question_mark"] = element["query"].count('?')
		#line["emptywords"] = len(element["query"]) - len(element["clean_lemma"])
		pal = element["pos_and_lemma"]
		line["verbNumber"] = getPOS(pal, "ver")
		#line["noCat"] = getPOS(pal, "nocat")
		#line["noun"] = getPOS(pal, "noun")
		line["advNumber"] = getPOS(pal, "adv")
		#line["adjNumber"] = getPOS(pal, "adj")
		#line["npr"] = getPOS(pal, "npr") * 1.0 / len(element["query"])
		#line["noCatpercent"] = round(getPOS(pal, "nocat") * 1.0 / len(element["query"]) * 1.0, 3)
		#if element["roots_list"]!="":
			#print (element["roots_list"])
			#freqObjectWordsWeight(line, element["object_lemma"], most_frequent_words,len(element["object_lemma"].split(" ")))
			#cercaFrequentWords(line, element["roots_list"], most_frequent_roots)

		token = nltk.word_tokenize(element["clean_lemma"])
		bigrams = ngrams(token, 2)
		b1 = list(bigrams)
		#creaTTR(line, element["clean_lemma"])
		cercaFrequentWords(line, element["clean_lemma"], most_frequent_words)
		freqWordsWeight(line, element["clean_lemma"], most_frequent_words,len(element["clean_lemma"].split(" ")))
		freqBigramsWeight(line, element["clean_lemma"], most_frequent_bigrams,len(b1))

		cercaBigrammi(line, element["clean_lemma"], most_frequent_bigrams)
		feat_dict.append(line)
		line = {}
		label.append(element["topic"])

target = open("/Users/giulia/Desktop/Awhy_TrainMultilevellClassifier/SecondRUNwithTrainedDATA/Plot/MostUsefulFeat.txt", 'w')

pipeline = Pipeline([
	# Use FeatureUnion to combine the features from subject and body
	('union', FeatureUnion(
		transformer_list=[
			('tfidf_vectorizer', TfidfVectorizer(norm='l2')),
			# ('tfidf_vectorizer_bi', TfidfVectorizer(min_df=2, ngram_range=(1,2))), ######TFIDF DEL BIGRAMMA: >0.75
			# ('count_vectorizer', CountVectorizer(min_df=3)),
			# ('count_vectorizer_bi', CountVectorizer(ngram_range=(1,2))),

		],

		# weight components in FeatureUnion
		transformer_weights={
			'tfidf_vectorizer': 1.0,
			# 'tfidf_vectorizer_bi':1.0,
			# 'count_vectorizer': 1.0,
			# 'count_vectorizer_bi': 1.0,
		}
	))
])


vec = DictVectorizer()
length_matrix = vec.fit_transform(feat_dict).toarray()
feature_names = vec.get_feature_names()
lunghezzaMatrice=np.array(length_matrix)

pipeline_matrix = pipeline.fit_transform(corpus).A

pipelineMatrice = np.array(pipeline_matrix)
pipelineRow=len(pipelineMatrice)
print("Il numero di righe di pipeline è: "+str(pipelineRow))
print("Il numero di righe di length è: "+str(len(lunghezzaMatrice)))
X = np.concatenate((lunghezzaMatrice,pipelineMatrice), axis=1)
#X = np.array(pipeline_matrix)
X = sparse.csr_matrix(X).toarray()
y = np.array(label)

featuresFAQ={}
featuresProdotto={}
featuresAccount={}

def accuracyCalc():
	final_true=[]
	final_pred=[]
	kf_total = StratifiedKFold(y, n_folds=10, shuffle=True)

	for (train,test)in kf_total:
		X_train, X_test = X[train], X[test]
		y_train, y_test = y[train], y[test]
		classifier = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)

		classifier.fit(X_train, y_train)
		y_pred = classifier.predict(X_test)

		#print (most_informative_feature_for_class(vec,classifier,"FAQ"))
		#print (most_informative_feature_for_class(vec,classifier,"Prodotto"))

		#calcolo le feature più rilevanti per la classificazione di ogni categoria, a ogni giro di classificazione

		if most_informative_feature_for_class(vec,classifier,"FAQ") is not None:
			for coef,feat in most_informative_feature_for_class(vec,classifier,"FAQ"):
				if feat in featuresFAQ:
					if feat in most_frequent_bigrams:
						featuresFAQ["Bigrams"]=featuresFAQ["Bigrams"]+coef
					elif feat in most_frequent_words:
						featuresFAQ["Unigrams"] = featuresFAQ["Unigrams"] + coef
					else:
						featuresFAQ[feat]=featuresFAQ[feat]+coef
				else:
					if feat in most_frequent_bigrams:
						featuresFAQ["Bigrams"] = coef
					elif feat in most_frequent_words:
						featuresFAQ["Unigrams"] = coef
					else:
						featuresFAQ[feat]=coef
		if most_informative_feature_for_class(vec, classifier, "Prodotto") is not None:
			for coef,feat in most_informative_feature_for_class(vec,classifier,"Prodotto"):
				if feat in featuresProdotto:
					if feat in most_frequent_bigrams:
						featuresProdotto["Bigrams"]=featuresProdotto["Bigrams"]+coef
					elif feat in most_frequent_words:
						featuresProdotto["Unigrams"] = featuresProdotto["Unigrams"] + coef
					else:
						featuresProdotto[feat]=featuresProdotto[feat]+coef
				else:
					if feat in most_frequent_bigrams:
						featuresProdotto["Bigrams"] = coef
					elif feat in most_frequent_words:
						featuresProdotto["Unigrams"] = coef
					else:
						featuresProdotto[feat]=coef


		if most_informative_feature_for_class(vec, classifier, "OrdiniAccountPersonali") is not None:
			for coef,feat in most_informative_feature_for_class(vec,classifier,"OrdiniAccountPersonali"):
				if feat in featuresAccount:
					if feat in most_frequent_bigrams:
						featuresAccount["Bigrams"]=featuresAccount["Bigrams"]+coef
					elif feat in most_frequent_words:
						featuresAccount["Unigrams"] = featuresAccount["Unigrams"] + coef
					else:
						featuresAccount[feat]=featuresAccount[feat]+coef
				else:
					if feat in most_frequent_bigrams:
						featuresAccount["Bigrams"] = coef
					elif feat in most_frequent_words:
						featuresAccount["Unigrams"] = coef
					else:
						featuresAccount[feat]=coef

		print ("accuracy:", accuracy_score(y_test, y_pred))
		print ("P/R/F1 micro:", precision_recall_fscore_support(y_test, y_pred, average="micro"))
		print ("P/R/F1 macro:", precision_recall_fscore_support(y_test, y_pred, average="macro"))

		final_true = final_true + list(y_test)
		final_pred = final_pred + list(y_pred)

	print ("Final accuracy:", accuracy_score(final_true, final_pred))
	print ("P/R/F1 Final micro:", precision_recall_fscore_support(final_true, final_pred, average="macro"))
	print ("P/R/F1 Final macro:", precision_recall_fscore_support(final_true, final_pred, average="micro"))


	#confusion_MATRIX
	cm = confusion_matrix(final_true, final_pred)
	np.set_printoptions(precision=2)
	print('Confusion matrix, without normalization')
	print(cm)
	plt.figure(figsize=(10, 10), dpi=100)

	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title("Confusion Matrix")
	plt.colorbar()
	tick_marks = np.arange(len(classifier.classes_))
	plt.xticks(tick_marks, classifier.classes_, rotation=90)
	plt.yticks(tick_marks, classifier.classes_)
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

	plt.savefig("/Users/giulia/Desktop/Awhy_TrainMultilevellClassifier/SecondRUNwithTrainedDATA/Plot/RandomForestClassifier10plot1_ALL.png")

	#normalized_confusionMATRIX
	cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	print('Normalized confusion matrix')
	print(cm_normalized)
	plt.figure(figsize=(10, 10), dpi=100)

	plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title("Normalized Confusion Matrix")
	plt.colorbar()
	tick_marks = np.arange(len(classifier.classes_))
	plt.xticks(tick_marks, classifier.classes_, rotation=90)
	plt.yticks(tick_marks, classifier.classes_)
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

	plt.savefig("/Users/giulia/Desktop/Awhy_TrainMultilevellClassifier/SecondRUNwithTrainedDATA/Plot/RandomForestClassifier10plot2_ALL.png")

	# Stampo i valori globali rilevanti di classificazione per ogni etichetta
	target.write("\nFeatures that discriminates the most, for category: ")
	for i in featuresFAQ:
		print ("FAQ " + i, featuresFAQ[i])


	for i in featuresProdotto:
		print ("Prodotto " + i, featuresProdotto[i])


	for i in featuresAccount:
		print ("OrdiniAccountPersonali " + i, featuresAccount[i])


	print ("######################################################")

if __name__ == "__main__":
	accuracyCalc()
else:
	pass