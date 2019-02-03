# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 13:13:23 2018

@author: Jérôme
"""



"""-----------------------------------------------------------------------"""
"""

    Ce programme a pour but de créer des sujets via la méthode LDA.
    Ces modèles sont ensuite affichés soit à l'écran, soit sous forme de
    WordCloud pour une représentation plus graphique.

"""
"""-----------------------------------------------------------------------"""




import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords





df=pd.read_csv('base_totale.csv',sep=',', engine='python')
df_tags=pd.read_csv('dico_tags.csv', sep=',')
df_test=pd.read_csv('base_test.csv', sep=',', engine='python')


df=df.fillna('')

df_text=df['Body']
bout_text=df_text.iloc[0:500]



''' lignes pour le test de l'appartenance de questions aux sujets '''

question=df_test['Body'].iloc[506]
question_tags=df_test['Tags'].iloc[506]
print(question_tags)




'''---------------- normalisation du texte --------------------------'''


def dico_tag(texte):
    
    dico=[]
    
    for x in texte : 
        
        dico.append(x)
    return dico

dico=dico_tag(df_tags.TagName)


# fonction de modif d'une question

def tokeniz_question(texte):
    
    regex=re.compile("[^a-zA-Z0-9]")
    text_modif=regex.sub(" ", texte)
    texte_lower=text_modif.lower()
    phrase=texte_lower.split()
    
    for i in list(phrase):
        if i in stopwords.words('english'):
            phrase.remove(i)
    
    for i in list(phrase):
        if i not in dico:
            phrase.remove(i)
    
        
    return phrase



# fonction de modif du corpus

def tokeniz(texte):
    
    regex=re.compile("[^a-zA-Z0-9]")
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

# fonctions de nettoyage du dataframe et de création du corpus

def nettoyage_dataframe(data):
    
    texte=data.apply(tokeniz)

    return texte


def creation_corpus(data):
    
    corpus=[]
    for x in data.iteritems():
        liste_mots=x[1]
        corpus.append(liste_mots)
    
    return corpus



essai=nettoyage_dataframe(bout_text)

corpus=creation_corpus(essai)

question=tokeniz_question(question)



''' mise en place de la méthode LDA : décompte des occurences des mots, 
un-gram et bigram, les trigrams n'apportant rien, ils n'ont pas 
été pris en compte lors de la mise en place finale '''


vect=CountVectorizer(stop_words='english', ngram_range=(1,2), max_features=500)

#mot=vect.fit_transform(df.Body)

mot=vect.fit_transform(corpus)

vocab=vect.get_feature_names()


mot=mot.toarray()

dist=np.sum(mot, axis=0)

''' graphe de la fréquence d'utilisation des mots '''
#for tag, count in zip(vocab, dist):
#    print (count,tag)

#viz=FreqDistVisualizer(features=feature)
#viz.fit(mot)
#viz.poof()



''' --------------définition des fonctions d'affichage pour la méthode LDA--------------- '''


def affichage(model, features_names,no_top_words):
    
    for sujet_id, sujet in enumerate(model.components_):
        print( 'Sujet :', sujet_id)
        print(" ".join([features_names[i] for i in sujet.argsort()[: -no_top_words -1:-1]]))
        
def recup_sujet(model, features_names,no_top_words):
    
    for sujet_id, sujet in enumerate(model.components_):
        titre_sujet = sujet_id
        mots_sujet=" ".join([features_names[i] for i in sujet.argsort()[: -no_top_words -1:-1]])
        wordcloud=WordCloud(max_font_size=30, background_color='white').generate(mots_sujet)
        plt.imshow(wordcloud,interpolation='bilinear')
        #plt.axis("off")
        plt.title(titre_sujet)
        plt.show()

def stockage(model, features_names, no_top_words):
    
    liste=[]
    for sujet_id, sujet in enumerate(model.components_):
        titre_sujet = sujet_id
        mots_sujet=" ".join([features_names[i] for i in sujet.argsort()[: -no_top_words -1:-1]])
        liste.append(mots_sujet)
        
        
    return liste
        

''' -----------------Mise en place de la méthode LDA------------------------ '''
        
no_top_words=15
vocabulaire=vect.vocabulary_
    

lda=LatentDirichletAllocation(n_components=10, learning_method='batch', max_iter=10, random_state=0)
sujet=lda.fit_transform(mot)
lda_tf=lda.fit(mot)


recup_sujet(lda, vocab, no_top_words)
#affichage(lda, vocab, no_top_words)
sujet=stockage(lda, vocab, no_top_words)



''' Tests effectués pour à l'envoi d'une question, le programme renvoi le sujet associé, et les mots
clés appartenant à ce sujet : comparaison aux tags réels et décompte des mots similaires '''

def correspondance(sujet, question):
    
    nbr_tags_init=len(question)
    liste_element_sujet=[]
    for k in range(0,10):
    
        for element in sujet[k]:
            liste_element_sujet.append(element)
    
        for i in question:
            if i not in liste_element_sujet:
                question.remove(i)
        print('pour le sujet', k,' le nombre de tags correspondant est de ', len(question), 'sur une base de ', nbr_tags_init)        
        print('les tags de la question sont : ', question)      

#correspondance(sujet, question)




               
    
    