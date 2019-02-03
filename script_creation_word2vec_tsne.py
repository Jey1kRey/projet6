# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 18:04:05 2018

@author: Jérôme
"""


"""-----------------------------------------------------------------------"""
"""

    Ce programme va créer et entrainer le modèle Word2Vec pour notre 
    situation précise : le vocabulaire sera informatique et technique.
    
    La première partie consiste au nettoyage de la base des questions,
    en limitant le vocabulaire aux mots techniques et informatiques, puis
    l'entrainement du modèle et le tracé du graphe t-SNE pour représenter
    les mots du modèle et visualiser les relations entre eux

"""
"""-----------------------------------------------------------------------"""



import pandas as pd
import re
import matplotlib.pyplot as plt

from gensim.models import word2vec
from sklearn.manifold import TSNE
from nltk.corpus import stopwords



df=pd.read_csv('base_totale.csv',sep=',', engine='python')

df_tags=pd.read_csv('dico_tags.csv', sep=',')

df=df.fillna('')



df_text=df['Body']

bout_text=df_text.iloc[0:10000]



'''----------- création de la fonction qui récupère dans une liste tous les tags---------- '''

def dico_tag(texte):
    
    dico=[]
    
    for x in texte : 
        
        dico.append(x)
    return dico

dico=dico_tag(df_tags.TagName)




''' création de la fonction de nettoyage du texte : les mots non présents dans les tags
ne sont pas conservés : cela supprime les verbes inutiles et les mots non pertinents pour la suite '''

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
    



'''--------- application de la fonction de nettoyage au dataframe, puis création
du corpus des mots---------------------------------------------------------------- '''


def nettoyage_dataframe(data):
    
    texte=data.apply(tokeniz)

    return texte


def creation_corpus(data):
    
    corpus=[]
    for x in data.iteritems():
        liste_mots=x[1].split(" ")
        corpus.append(liste_mots)
    
    return corpus


#essai=nettoyage_dataframe(bout_text)
essai=nettoyage_dataframe(df_text)
corpus=creation_corpus(essai)




'''------------------- mise en place de word2vec ------------------------------------------'''

model=word2vec.Word2Vec(corpus, size=300, window=20, min_count=2, workers=4, iter=100)
#model=word2vec.Word2Vec(corpus, size=150, window=20, min_count=2, workers=4, iter=100)


vocabulaire=model.wv.vocab
words=list(sorted(model.wv.vocab))

model.save('model_wv_150.bin') # sauvegarde du modèle

''' essai du modèle en affichant la similarité entre des mots, la prédiction
de mots proches d'autres ... '''
#print(model.similarity('binary','binary_data'))
#print(model.predict_output_word(['asmx', 'aspx']))





'''------------------- tracé du graphe t-SNE pour le modèle calculé précédemment -----------'''



x=model[model.wv.vocab]
tsne=TSNE(n_components=2)
x_tsne=tsne.fit_transform(x)

plt.scatter(x_tsne[:,0], x_tsne[:,1])
plt.show()



def graphe_tsne(model):
    
    labels=[]
    mots=[]
    
    for word in model.wv.vocab :
        
        mots.append(model[word])
        labels.append(word)
        
    
    t_sne=TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=0)
    
    t_sne_appli=t_sne.fit_transform(mots)
    
    x=[]
    y=[]
    
    for i in t_sne_appli : 
        x.append(i[0])
        y.append(i[1])
    
    plt.figure( figsize=(16,16))
    
    for k in range(len(x)):
        plt.scatter(x[k], y[k])
        plt.annotate(labels[k], xy=(x[k],y[k]), xytext=(5,2), textcoords='offset points', ha='right', va='bottom')
    
    plt.show()

graphe_tsne(model)




  
    


