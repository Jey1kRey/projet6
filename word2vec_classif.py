# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 08:42:39 2018

@author: Jérôme
"""

import pandas as pd
import gensim
import numpy as np
from gensim.models import word2vec
import re
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier





#model=gensim.models.Word2Vec.load('model_wv_150.bin')
model=gensim.models.Word2Vec.load('model_wv_complet.bin')
#w2v = dict(zip(model.wv.index2word, model.wv.syn0))

vocabulaire=model.wv.syn0

#print(len(w2v))
#print(len(model.wv.index2word))
#print(len(model.wv.vocab.keys()))


#df=pd.read_csv('base_texte.csv',sep=',')
df=pd.read_csv('base_totale.csv',sep=',', engine='python')

dftags=pd.read_csv('dico_tags.csv', sep=',')

df_test=pd.read_csv('base_test.csv', sep=',', engine='python')

df=df.fillna('')



questions=df['Body'].iloc[0:20]
indices=questions.index

df_tags=df['Tags'].iloc[0:20]


''' normalisation du texte '''

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

df_question=nettoyage_dataframe(questions)
question_test=df_question.iloc[0]
print(question_test)

def net_tg(texte):
    regex=re.compile("[^a-zA-Z]")
    text_modif=regex.sub(" ", texte)
    texte_lower=text_modif.lower()
    
    
    return texte_lower

def net_df_tg(data):
    
    texte=data.apply(net_tg)
    
    return texte

y_tg=net_df_tg(df_tags)




def recup_vecteurs(corpus, model):
    
    index2word_set = set(model.wv.vocab.keys()) # words known to model
    
    featureVec = np.zeros(model.vector_size, dtype="float32")
    liste_vecteur=[]
    for word in corpus : 
        
        if word in index2word_set:
            featureVec = np.add(featureVec, model[word])
            liste_vecteur.append(featureVec)
    return liste_vecteur


def creation_corpus(questions):

    liste_questions=[]
    for element in questions : 
        mots_wv=recup_vecteurs(element, model)
        liste_questions.append(mots_wv)
    
    return liste_questions

x_train, x_test, y_train, y_test = train_test_split(questions, y_tg, train_size=0.7)           

liste_xtrain=creation_corpus(x_train)
liste_xtest=creation_corpus(x_test)


train=np.vstack(liste_xtrain)
test=np.vstack(liste_xtest)


#☻essai=pd.DataFrame(liste_finale, index=indices)

#print(essai)
#vecteurs_corpus=recup_vecteurs(question_test, model)
#print(len(vecteurs_corpus))
#print(vecteurs_corpus)



#x_train, x_test, y_train, y_test = train_test_split(test, y_tg, train_size=0.7)

foret=OneVsRestClassifier(RandomForestClassifier())
foret.fit(train, y_train)
#print(foret.score(x_test,y_test))
#☺print(foret.predict(x_test[10:15]))
#print(y_test[10:15])

'''
x_train, x_test, y_train, y_test = train_test_split(vocabulaire, y_tg, train_size=0.7)


foret=OneVsRestClassifier(RandomForestClassifier())
foret.fit(x_train, y_train)
print(foret.score(x_test,y_test))
print(foret.predict(x_test[10:15]))
print(y_test[10:15])
'''


