# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:13:47 2018

@author: Jérôme
"""


"""-----------------------------------------------------------------------"""
"""

    Ce programme effectue la création de l'API en utilisant le module Flask
    de Python. Il charge le modèle Tf-Idf exécuté dans un autre programme
    ainsi que le modèle de Forêt Aléatoire préalablement entraîné. 
    Ensuite, il demande à l'utilisateur une question, effectue un nettoyage
    de cette question, normalisation du texte et recherche des mots
    pertinents, enfin, retourne les tags possibles.

"""
"""-----------------------------------------------------------------------"""

from flask import Flask, request, render_template, url_for
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer



df_tags=pd.read_csv('dico_tags.csv', sep=',')



vocabulaire_tfidf=pickle.load(open('vocabulaire_tfidf.sav', 'rb'))
tfidf=pickle.load(open('modele_tfidf_bis.sav', 'rb'))

foret=pickle.load(open('modele_foret_bis.sav', 'rb'))
#foret=pickle.load(open('modele_foret.sav', 'rb'))

''' -------------- nettoyage et normalisation de la question -----------'''

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
        if i not in dico:
            phrase.remove(i)
    
     
    return phrase
    



app = Flask(__name__)

''' options '''

#app.config.from_object('config')

''' page d'accueil '''

@app.route('/')
def home():
    return render_template("question.html")




@app.route('/', methods=['POST'])
def formulaire():
    
    text=request.form.get('question')
    
    texte_modifie=tokeniz(text)
    
    ligne=tfidf.transform(texte_modifie)
    
    prediction_tag=foret.predict(ligne)       

    return render_template('question.html',  tag0=prediction_tag[0], tag1=prediction_tag[1], tag2=prediction_tag[2])




if __name__ == '__main__':
    app.run()
    
  