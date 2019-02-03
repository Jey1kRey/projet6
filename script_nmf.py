# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 17:17:05 2018

@author: Home
"""



import pandas as pd

from wordcloud import WordCloud
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


df=pd.read_csv('base_texte.csv',sep=',')

df=df.drop(['Id','PostTypeId','AcceptedAnswerId','ParentId','DeletionDate','Score','ViewCount','OwnerUserId','LastEditorDisplayName'], axis=1)
df=df.drop(['LastEditDate','LastActivityDate','AnswerCount','CommentCount','FavoriteCount','ClosedDate','CommunityOwnedDate'], axis=1)
df=df.drop(['OwnerDisplayName', 'LastEditorUserId'], axis=1)
df=df.drop(['CreationDate'], axis=1)

df=df.fillna('')



df_text=df['Body']

bout_text=df_text.iloc[0:]




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

def structure_sujet(model, features_names, no_top_words):
    
    for sujet_id, sujet in enumerate(model.components_):
        titre_sujet = sujet_id
        mots_sujet=" ".join([features_names[i] for i in sujet.argsort()[: -no_top_words -1:-1]])
        
        


no_top_words=20


tfidf=TfidfVectorizer(max_df=0.95, min_df=2,stop_words='english')

vocab=tfidf.fit_transform(df.Title)

nom_feat=tfidf.get_feature_names()

nb_sujet=15

nmf=NMF(n_components=nb_sujet, alpha=.1, l1_ratio=.05, init='nndsvd', random_state=0)
nmf.fit(vocab)


#affichage(nmf, nom_feat, no_top_words)

recup_sujet( nmf, nom_feat, no_top_words)