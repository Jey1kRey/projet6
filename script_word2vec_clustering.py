# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 17:36:06 2018

@author: Jérôme
"""


"""-----------------------------------------------------------------------"""
"""

    Ce programme permet d'effectuer les tests sur le modèle Word2Vec
    préalablement entrainé. 
    Pour une question donnée, on retournera les mots les plus proches. On
    effectuera en outre un clustering des mots pour obtenir
    des groupes similaires et retournés pour une question donnée les tags 
    potentiels rendus par le modèle Word2Vec.

"""
"""-----------------------------------------------------------------------"""





import pandas as pd
import gensim
import matplotlib.pyplot as plt
import re


from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from gensim.models import word2vec



model=gensim.models.Word2Vec.load('model_wv_150.bin')
#model=gensim.models.Word2Vec.load('model_wv_complet.bin')

#df=pd.read_csv('base_texte.csv',sep=',')
df=pd.read_csv('base_totale.csv',sep=',', engine='python')
df_tags=pd.read_csv('dico_tags.csv', sep=',')
df_test=pd.read_csv('base_test.csv', sep=',', engine='python')

df=df.fillna('')


df_text=df['Body']

question=df_test['Body'].iloc[7]
question_tags=df_test['Tags'].iloc[7]
print(question_tags)



'''------------- normalisation du texte des questions testées--------------- '''

def dico_tag(texte):
    
    dico=[]
    
    for x in texte : 
        
        dico.append(x)
    return dico

dico=dico_tag(df_tags.TagName)


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
    
        
    return phrase
    

def creation_corpus(data):
    
    corpus=[]
    for x in data:
        liste_mots=x.split(" ")
        corpus.append(liste_mots)
    
    return corpus

question_nettoye=tokeniz(question)



''' test direct des premiers mots similaires retournés pour des mots de
questions diverses '''

#words=list(sorted(model.wv.vocab))
#print(words[:10])

#print(model.similarity('firefox','safari'))
#print(model.predict_output_word(question_nettoye))


#tags_possible=model.predict_output_word(question_nettoye)
#print(tags_possible)




'''------------------ mise en place du clustering k-mean-------------- '''



inertie=[]
'''
for i in range(2,25):
    k_moyenne=KMeans(n_clusters=i,init='k-means++',random_state=0)
    k_moyenne.fit(model.wv.syn0)
    inertie.append(k_moyenne.inertia_)
    

plt.plot(range(2,11),inertie)
plt.xlabel('nbr clusters')
plt.ylabel('inertie')
plt.show()
'''
# le nombre k de clusters = 20 permet d'avoir des groupes bien différenciés sans
# répétition des mêmes contenus

kmean=KMeans(n_clusters=20, init='k-means++', max_iter=100)
x=kmean.fit(model.wv.syn0)
idx=kmean.fit_predict(model.wv.syn0)
label=kmean.labels_.tolist()

label_kmeans=pd.Series(label, name='label_cluster')
print(label_kmeans.value_counts())



''' liste mots dans les clusters : permet de récupérer les mots proches dans le
même cluster  '''


liste_mots=list(zip(model.wv.index2word, idx))
liste_mots_trier=sorted(liste_mots, key = lambda element: element[1], reverse=False)
'''
for mot in liste_mots_trier:
    
    print(mot[0], '\t', str(mot[1]))
'''
corpus_question=['wikipedia','safari','firefox','wiki','http']
for element in corpus_question:
    
    if element in liste_mots_trier:
        
        print(element[0], '\t', str(element[1]))