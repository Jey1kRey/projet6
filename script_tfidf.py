# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 13:17:03 2018

@author: Jérôme
"""



"""-----------------------------------------------------------------------"""
"""

    Ce programme va créer les variables Tf-Idf pour la base de question. Le 
    modèle sera sauvegardé pour une utilisation dans la création de l'API
    La seconde partie du programme consiste en test de trois méthodes de 
    classification via la stratégie OneVsRest puisque les labels sont 
    tous les tags.
    Les méthodes de classification sont : Logistic Regression, Naive Bayes
    et Forêt Aléatoire

"""
"""-----------------------------------------------------------------------"""




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import DistanceMetric




df=pd.read_csv('base_totale.csv',sep=',', engine='python')

dftags=pd.read_csv('dico_tags.csv', sep=',')

df_test=pd.read_csv('base_test.csv', sep=',', engine='python')




df=df.fillna('')


indice=2000

df_body=df['Body'].iloc[0:indice]

df_tags=df['Tags'].iloc[0:indice]





'''--------------------- normalisation du texte-------------------- '''

def dico_tag(texte):
    
    dico=[]
    
    for x in texte : 
        
        dico.append(x)
    return dico

dico=dico_tag(dftags.TagName)



def tokeniz(texte):
    
    regex=re.compile("[^a-zA-Z]")
    text_modif=regex.sub(" ", texte)
    texte_lower=text_modif.lower()
    phrase=texte_lower.split()
    
    for i in list(phrase):
        if i in stopwords.words('english'):
            phrase.remove(i)
    
    for i in list(phrase):
        if i not in dico:
            phrase.remove(i)    
    
    mots=" ".join(phrase)
    
    return mots



def nettoyage_dataframe(data):
    
    texte=data.apply(tokeniz)

    return texte


#y=nettoyage_dataframe(df_tags)

df_question=nettoyage_dataframe(df_body)



''' -----------------normalisation des tags utilisés comme label----------- '''


def net_tg(texte):
    regex=re.compile("[^a-zA-Z]")
    text_modif=regex.sub(" ", texte)
    texte_lower=text_modif.lower()
    
    
    return texte_lower

def net_df_tg(data):
    
    texte=data.apply(net_tg)
    
    return texte

y_tg=net_df_tg(df_tags)



''' création des variables Tf-Idf avec utilisation de un-gram, bigram et trigram '''



tfidf=TfidfVectorizer(ngram_range=(1,3),stop_words='english')

vocabulaire=tfidf.fit_transform(df_question)

''' sauvegarde des modèles '''
#nom_fichier='vocabulaire_tfidf.sav'
#pickle.dump(vocabulaire, open(nom_fichier, 'wb'))
modele_tfidf='modele_tfidf_bis.sav'
pickle.dump(tfidf, open(modele_tfidf, 'wb'))



''' ----------------------test des méthodes de classification ------------------'''


x_train, x_test, y_train, y_test = train_test_split(vocabulaire, y_tg, train_size=0.7)

'''
clr=OneVsRestClassifier(LogisticRegression(solver='lbfgs'))
clr.fit(x_train,y_train)
#print(clr.score(x_test, y_test))
#print(clr.predict(x_test[25:50]))
#print(y_test[25:50])


naive=OneVsRestClassifier(MultinomialNB())
naive.fit(x_train, y_train)
print(naive.score(x_test,y_test))
print(naive.predict(x_test[25:50]))
'''

foret=OneVsRestClassifier(RandomForestClassifier())
foret.fit(x_train, y_train)
#print(foret.score(x_test,y_test))
print(foret.predict(x_test[25:50]))
print(y_test[25:50])

dist=DistanceMetric('jaccard')

distance= dist.pairwise(foret.predict(x_test[25:50]), y_test[25:50])
print(distance)


''' sauvegarde du modèle de la forêt aléatoire '''
#foret_sauv='modele_foret_bis.sav'
#pickle.dump(foret, open(foret_sauv, 'wb'))

#logistic_sauv='modele_logistic.sav'
#pickle.dump(clr, open(logistic_sauv, 'wb'))