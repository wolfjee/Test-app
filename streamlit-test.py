# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 09:49:53 2020

@author: Gideon
"""

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


st.write("""
# Simple Prediction App        
         
This app predicts Iris flower types with Logit or Random Forest models.        
         """)

st.sidebar.header('User Input')

def input_user():
    sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)
    data = {'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = input_user()

st.subheader('User Input')
st.write(df)

flowers = datasets.load_iris()

flowers_df = pd.DataFrame(data= np.c_[flowers['data'], flowers['target']],
                     columns= flowers['feature_names'] + ['target'])

X = flowers.data
y = flowers.target

classif = st.select_slider(
        'Model Selection',
        options=['Random Forest', 'Logit'])

if classif == 'Random Forest':
    rfc = RandomForestClassifier()
    rfc.fit(X,y)
    pred = rfc.predict(df)
    pred_proba = rfc.predict_proba(df)


if classif == 'Logit':
    lrc = LogisticRegression()
    lrc.fit(X,y)
    pred = lrc.predict(df)
    pred_proba = lrc.predict_proba(df)

point = pd.concat([df, pd.DataFrame(pred, columns=['target'])], axis=1)

st.subheader('Classes of flower')
st.write(flowers.target_names)

st.subheader('Prediction')
st.write(flowers.target_names[pred])

st.subheader('Prediction Probability')
st.write(pred_proba)

sidebar = st.sidebar.selectbox(
        'What plot would you like to view?',
        ('Sepal Length/Width', 'Petal Length/Width', 'Pairplot', 'Contour Plot'))

if sidebar == 'Sepal Length/Width':
    st.subheader('Visualization of Sepal')
    ax = sns.FacetGrid(flowers_df, hue="target", palette="husl", height=5) \
            .map(plt.scatter, "sepal length (cm)", "sepal width (cm)")         
    st.write(ax.fig, plt.scatter(point["sepal length (cm)"],point["sepal width (cm)"],c='k'), \
                     plt.legend(loc='upper left', labels=['setosa', 'versicolor', 'virginica','input']))
if sidebar == 'Petal Length/Width':
    st.subheader('Visualization of Sepal')
    ax = sns.FacetGrid(flowers_df, hue="target", palette="husl", height=5) \
            .map(plt.scatter, "petal length (cm)", "petal width (cm)")         
    st.write(ax.fig, plt.scatter(point["petal length (cm)"],point["petal width (cm)"],c='k'), \
                     plt.legend(loc='upper left', labels=['setosa', 'versicolor', 'virginica','input']))
if sidebar == 'Pairplot':
   st.subheader('Pairplot')
   ax = sns.PairGrid(flowers_df, hue="target", palette="husl", height=5,\
                      x_vars=["sepal length (cm)", "sepal width (cm)"],\
                      y_vars=["petal length (cm)", "petal width (cm)"])
   ax.map(sns.scatterplot)
   ax.add_legend(loc='upper right', labels=['setosa', 'versicolor', 'virginica'])
   st.write(ax.fig)
if sidebar == 'Contour Plot':
    
    st.subheader('Contour Plot')
    flowers1 = datasets.load_iris() 
    def fig_show():
        for pairidx, pair in enumerate([[0, 1], [2, 3]]):
            # We only take the two corresponding features
            X1 = flowers1.data[:, pair]
            y1 = flowers1.target    
            # Train
            if classif == 'Random Forest':
                clf = RandomForestClassifier().fit(X1, y1)        
                # Plot the decision boundary   
                fig = plt.subplot(2, 1, pairidx + 1)        
                x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
                y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                     np.arange(y_min, y_max, 0.02))
                plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)        
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)    
                plt.xlabel(flowers1.feature_names[pair[0]])
                plt.ylabel(flowers1.feature_names[pair[1]])    
                # Plot the training points
                for i, color in zip(range(3), 'ryb'):
                    idx = np.where(y == i)
                    plt.scatter(X1[idx, 0], X1[idx, 1], c=color, label=flowers1.target_names[i],
                                cmap=plt.cm.RdYlBu, edgecolor='black', s=15)
            elif classif == 'Logit':
                clf = LogisticRegression().fit(X1, y1)        
                # Plot the decision boundary   
                fig = plt.subplot(2, 1, pairidx + 1)        
                x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
                y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                     np.arange(y_min, y_max, 0.02))
                plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)        
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)    
                plt.xlabel(flowers1.feature_names[pair[0]])
                plt.ylabel(flowers1.feature_names[pair[1]])    
                # Plot the training points
                for i, color in zip(range(3), 'ryb'):
                    idx = np.where(y == i)
                    plt.scatter(X1[idx, 0], X1[idx, 1], c=color, label=flowers1.target_names[i],
                                cmap=plt.cm.RdYlBu, edgecolor='black', s=15)
        return fig
        
    st.write(fig_show().figure)






