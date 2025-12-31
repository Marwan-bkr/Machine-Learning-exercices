
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np

#source ./venvMac/bin/activate

#A.1
iris = datasets.load_iris()

#A.2
st.write('Iris Data:')
st.write(iris.data)

st.write('Iris Variables:')
st.write(iris.feature_names)

st.write('Iris Classes:')
st.write(iris.target_names)

#A.3
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
c = 0
for id in df['species'] :
    df['species'][c] = iris.target_names[id]
    c = c+1
st.write('Iris DataFrame:')
st.write(df)

#A.4
st.write('Iris Stats :')
st.write(df.describe())

#A.5
st.write('Iris Data Size :')
st.write(df.size)
st.write('Iris Data Shape :')
st.write(df.shape)

#B.1
mnist = fetch_openml('mnist_784')

#B.2
    #Affichez matrice, le nombre de donn´ees et de variables, les num´eros de classes pour chaque donn´ee, ainsi que la moyenne, l’´ecart-type, les valeurs min et max pour chaque variable ; enfin donnez le nombre de classes avec la fonction unique
st.write('MNIST :')
#st.write(mnist)

df_mnist = pd.DataFrame(data=mnist.data)
st.write('MNIST Data :')
st.write('Too big to be display')
#st.write(df_mnist) #to big
st.write('MNIST Size :')
st.write(df_mnist.size)
st.write('MNIST Shape (row,features):')
st.write(df_mnist.shape)
st.write('MNIST Variables:')
st.write(mnist.feature_names)

st.write('MNIST Classes per data (the index):')
st.write(mnist.target)

st.write('MNIST Stats per data :')
st.write(df_mnist.T.describe())

#C.2
st.write('Blobs 1 :')
X, y = datasets.make_blobs(n_samples=1000, n_features=2, centers=4)
fig, ax = plt.subplots(figsize=(10,10))
scatter = ax.scatter(X[:, 0], X[:, 1], c=y)
ax.set_title("Data with blobs")
ax.set_xlabel("Axe x")
ax.set_ylabel("Axe y")
ax.set_xlim(-15,15)
ax.set_ylim(-15,15)
st.pyplot(fig)

X1, y1 = datasets.make_blobs(n_samples=100, n_features=2, centers=2)
X2, y2 = datasets.make_blobs(n_samples=500, n_features=2, centers=3)
X3 = np.vstack((X1,X2))
y3 = np.hstack((y1, y2))

st.write('Blobs 2 :')
fig, ax = plt.subplots(figsize=(10,10))
scatter = ax.scatter(X1[:, 0], X1[:, 1], c=y1)
ax.set_title("Data with blobs")
ax.set_xlabel("Axe x")
ax.set_ylabel("Axe y")
ax.set_xlim(-15,15)
ax.set_ylim(-15,15)
st.pyplot(fig)

st.write('Blobs 3 :')
fig, ax = plt.subplots(figsize=(10,10))
scatter = ax.scatter(X2[:, 0], X2[:, 1], c=y2)
ax.set_title("Data with blobs")
ax.set_xlabel("Axe x")
ax.set_ylabel("Axe y")
ax.set_xlim(-15,15)
ax.set_ylim(-15,15)
st.pyplot(fig)

st.write('Blobs 2 et 3 :')
fig, ax = plt.subplots(figsize=(10,10))
scatter = ax.scatter(X3[:, 0], X3[:, 1], c=y3)
ax.set_title("Data with blobs")
ax.set_xlabel("Axe x")
ax.set_ylabel("Axe y")
ax.set_xlim(-15,15)
ax.set_ylim(-15,15)
st.pyplot(fig)
#fusionner les labels apporte de la confusion pcq dans les deux dataset il existe un groupe avec le label 0 et quand les datastes sont joint deux clusters bien distincts se retrouvent sous le même label.