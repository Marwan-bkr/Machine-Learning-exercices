import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, fcluster
import scipy.cluster.hierarchy as sch
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
import seaborn as sns
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score 


st.header("Machine Learning I")
st.header("TP4 - Classification Non-Supervisée")
st.header("")
st.subheader("Marwan BENLKHIR")
st.header("")
st.header("")

st.header("1 Analyse des Iris de Fisher avec l'algorithme K-Moyennes")

st.info("1")
iris = datasets.load_iris()
st.write('Iris Data:')
st.write(iris.data)
st.write('Iris Variables:')
st.write(iris.feature_names)
st.write('Iris Classes:')
st.write(iris.target_names)
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]

st.write('Iris Describe:')
st.write(df.describe())

st.info("2")
data = pd.DataFrame(iris.data)
label = pd.DataFrame(iris.target)

st.info("3")
pca = PCA(n_components=2)
X = pca.fit_transform(data)
PCA(n_components=2)
st.write('PCA explained variance ratio :')
st.write(pca.explained_variance_ratio_)
st.write('Singular values :')
st.write(pca.singular_values_)
st.write(X)

st.info("4")
kmeans0 = KMeans(n_clusters=3, random_state=150, n_init=15).fit(X)
pred = kmeans0.predict(X)
fig = plt.figure()
ax = fig.add_subplot(111)
centroid0 = kmeans0.cluster_centers_
plt.scatter(centroid0[:,0], centroid0[:,1], c='red', s=50)
ax.scatter(X[:,0], X[:,1], c=pred)
st.pyplot(fig)

st.info("5")
kmeans1 = KMeans(n_clusters=3, random_state=10, n_init=15).fit(X)
pred1 = kmeans1.predict(X)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
centroid1 = kmeans1.cluster_centers_
plt.scatter(centroid1[:,0], centroid1[:,1], c='red', s=50)
ax1.scatter(X[:,0], X[:,1], c=pred1)
st.pyplot(fig1)

st.info("6")
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(X[:,0], X[:,1], c=label)
st.pyplot(fig2)

st.info("7")
contingency_table = pd.crosstab(label[0], pred)
st.write(contingency_table)

st.info("8")
st.write("Silhouette Score:", silhouette_score(X, pred1))
st.write("Le Silhouette score n'est pas fabuleux. C'est compréhensible car les clusters ne sont pas très séparés.")

st.info("9")
kmeans2 = KMeans(n_clusters=3, random_state=10, n_init=15).fit(data)
pred2 = kmeans2.predict(data)
st.write("Silhouette Score:", silhouette_score(data, pred2))
st.write("Dans notre cas le réduction de dimension a permis d'améliorer le Silhouette Score. La réduction de dimension a fait disparaître les données inutiles que considère comme du bruit.")

st.header("2 Clustering hiérarchique sur les données Iris")

st.info("1")
st.info("2")
st.info("3")
fig3 = plt.figure(figsize=(30, 10))
Z = sch.linkage(data, method='complete')
dn = sch.dendrogram(Z, count_sort='ascending')
plt.axhline(y=Z[-3, 2], color='r', linestyle='--', label='3 clusters')  # cut line
st.pyplot(fig3)

st.info("4")
labels_pred = fcluster(Z, t=3, criterion='maxclust')
contingence = pd.crosstab(label[0], labels_pred, rownames=['Vrai label\Cluster'])
st.write(contingence)

st.info("5")
fig4 = plt.figure(figsize=(30, 10))
Z1 = sch.linkage(data, method='average')
dn = sch.dendrogram(Z1, count_sort='ascending')
plt.axhline(y=Z1[-3, 2], color='r', linestyle='--', label='3 clusters')  # cut line
st.pyplot(fig4)
labels_pred1 = fcluster(Z1, t=3, criterion='maxclust')
contingence = pd.crosstab(label[0], labels_pred1, rownames=['Vrai label\Cluster'])
st.write(contingence)
st.write("Le linkage moyen est plus perfomant que le linkage complet car sur deux catégorie il n'y a auncune erreur tandis que le complet ne s'est pas trompé unquement sur une seule catégorie.")

st.info("6")
st.write("Silhouette Score linkage complet:", silhouette_score(X, labels_pred))
st.write("Silhouette Score linkage moyen:", silhouette_score(X, labels_pred1))
st.write("Le linkage moyen est meilleur que le linkage complet sur le Silhouette Score. Néanmoins le complet atteint le même silhouette score que kmeans avec des données reduites en dimension.")

st.header("3 Nombre optimal de clusters sur les données atmosphère d'exoplanète")

st.info("1")
dfplan = pd.read_csv("14, 15/planete.csv", sep=';')
X_plan = dfplan.drop(columns=['Type'])
st.write(X_plan.head())

st.info("2")
st.write("Le DBI exprime à quel point les clusters sont compactes et à quel point ils sont séparés entre eux. Plus le DBI est faible, plus les clusters sont de qualité. Le CHI se base sur les mêmes critères mais avec un méthode différente. Plus le CHI est haut mieux les clusters sont.")

st.info("3")

chi_scores = []
dbi_scores = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=150, n_init=15).fit(X_plan)
    labels = kmeans.labels_
    chi = calinski_harabasz_score(X_plan, labels)
    dbi = davies_bouldin_score(X_plan, labels)
    chi_scores.append(chi)
    dbi_scores.append(dbi)
kmeans0 = KMeans(n_clusters=3, random_state=150, n_init=15).fit(X_plan)
pred = kmeans0.predict(X_plan)
fig5 = plt.figure(figsize=(12, 5))
ax5 = fig5.add_subplot(1, 2, 1)
ax5.plot(K, chi_scores, 'bx-')
ax5.set_xlabel('Nombre de clusters k')
ax5.set_ylabel('Calinski-Harabasz Index (CHI)')
ax5.set_title('CHI en fonction du nombre de clusters k')
ax6 = fig5.add_subplot(1, 2, 2)
ax6.plot(K, dbi_scores, 'rx-')
ax6.set_xlabel('Nombre de clusters k')
ax6.set_ylabel('Davies-Bouldin Index (DBI)')
ax6.set_title('DBI en fonction du nombre de clusters k')    
st.pyplot(fig5)
st.write("Avec la méthode du coude basée sur le CHI et le DBI, on peut déterminer que le nombre de cluster idéal est 4.")

st.header("4 Pipeline d'analyse de données et apprentissage")

st.write("J'ai choisi un dataset sur kaggle portant sur les propriétés physiques et chimiques d'acides aminés. Le lien pour le télécharger est le suivant : https://www.kaggle.com/datasets/alejopaullier/aminoacids-physical-and-chemical-properties")

st.info("Prétraitement et transformation des données")
dfacid = pd.read_csv("16, 17, 18/aminoacids.csv", sep=',')
st.markdown("""Columns description:

    - Name: name of the amino acid.
    - Abbr: abbreviation of the amino acid.
    - Letter: letter of the amino acid.
    - Molecular Weight: molecular weight.
    - Molecular Formula: molecular formula.
    - Residue Formula: residue formula.
    - Residue Weight: residue weight (-H20)
    - pKa1: the negative of the logarithm of the dissociation constant for the -COOH group.
    - pKb2: the negative of the logarithm of the dissociation constant for the -NH3 group.
    - pKx3: the negative of the logarithm of the dissociation constant for any other group in the molecule.
    - pl4: the pH at the isoelectric point.
    - H: hydrophobicity.
    - VSC: volumes of side chains amino acids.
    - P1: polarity.
    - P2: polarizability.
    - SASA: solvent accesible surface area.
    - NCISC: net charge of side chains.
""")

st.markdown("#### 1")
st.write("Shape:", dfacid.shape)
st.write(dfacid.dtypes)
st.write(dfacid.head())

st.markdown("#### 2")
st.write(dfacid.describe())
st.write("Voici deux exemple de visualisation d'une plus ou moins grande corrélation.")
fig, ax = plt.subplots(figsize=(20, 5))
plot = sns.lmplot(y='Molecular Weight', x='VSC', data=dfacid)
st.pyplot(plot.figure)
fig, ax = plt.subplots(figsize=(20, 5))
plot = sns.lmplot(y='P1', x='VSC', data=dfacid)
st.pyplot(plot.figure)

st.markdown("#### 3")
st.write("Valeurs manquantes par colonne :")
st.write(dfacid.isnull().sum())
st.write("Pour la colonne pKx3, le nombre de valeurs manquantes par rapport au nombre de lignes est trop important. La colonne sera supprmiée.")
st.write("Valeurs manquantes par ligne en pourcentage:")
st.write(dfacid.isnull().sum(axis=1)/len(dfacid.columns)*100)
st.write("Au dessus de 30% de valeurs manquantes dans une ligne, il est d'usage de supprimer la ligne. Pour les autres, les valeurs manquantes seront remplacées par le mode pour les catégorielles et par la médiane pour les numériques.")
dfacid = dfacid.drop(columns=['pKx3'])
dfacid = dfacid.drop(index=(dfacid[dfacid.isnull().sum(axis=1)/len(dfacid.columns)*100>30].index))
st.write("Valeurs manquantes par colonne après traitement :")
st.write(dfacid.isnull().sum())
st.write("Valeurs manquantes par ligne en pourcentage après traitement:")
st.write(dfacid.isnull().sum(axis=1))
st.write("Après avoir fait ces deux opérations de suppression, il n'y a pas besoin de remplacer de valeurs manquantes puisque le dataset est complet. Ces deux trie ont servis à obtenir un dataset complet à 100%.")
st.write("Nouvelle shape du dataset : ", dfacid.shape)

st.markdown("#### 4")
dfaciddum = pd.get_dummies(dfacid[["Molecular Formula","Residue Formula"]])
scaler = StandardScaler()
dfacidstd = scaler.fit_transform(dfacid[['Molecular Weight','Residue Weight','pKa1','pKb2','pl4','H','VSC','P1','P2','SASA','NCISC','carbon','hydrogen','nitrogen','oxygen','sulfur']])
dfacidstd = pd.DataFrame(dfacidstd, columns=scaler.get_feature_names_out(input_features=None))
X = pd.concat([dfaciddum.reset_index(drop=True), dfacidstd.reset_index(drop=True)], axis=1)
st.write("Nouvelle shape du dataset standardisé : ", X.shape)
st.write("Exemple de 5 lignes des données standardiées :")
st.write(X.head())

st.info("Analyse des données & Visualisation")
corr_mat = X.corr().stack().reset_index(name="correlation")
g = sns.relplot(
    data=corr_mat,
    x="level_0", y="level_1", hue="correlation", size="correlation",
    palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
    height=10, sizes=(25, 75), size_norm=(-.2, .8),
)
g.set(xlabel="", ylabel="", aspect="equal")
g.despine(left=True, bottom=True)
g.ax.margins(.02)
for label in g.ax.get_xticklabels():
    label.set_rotation(90)
st.pyplot(g)
st.write("Exemple de visualisation des features les plus corrélées.")
fig, ax = plt.subplots(figsize=(20, 5))
plot = sns.lmplot(y="Molecular Weight", x="P2", data=X)
st.pyplot(plot.figure)
fig, ax = plt.subplots(figsize=(20, 5))
plot = sns.lmplot(y="SASA", x="VSC", data=X)
st.pyplot(plot.figure)
fig, ax = plt.subplots(figsize=(20, 5))
plot = sns.lmplot(y="Residue Weight", x="carbon", data=X)
st.pyplot(plot.figure)

st.info("Modélisation et Evaluation")
data = X.drop(columns=['VSC'])
label = X['VSC']

st.markdown("#### 1")
#K-Means
sil = []
clus = []
for i in range(2,11):
    kmeans0 = KMeans(n_clusters=i, random_state=150, n_init=50).fit(data)
    pred = kmeans0.predict(data)
    clus.append(i)
    sil.append(silhouette_score(data, pred))
fig, ax = plt.subplots()
ax.plot(clus, sil)
st.pyplot(fig)
st.write("Je choisis k=6 grâce à la méthode du coude convexe car le silhouette score est à maximiser. Les résultats du modèle sont décevant. Il faut prendre en compte que le sujet est complexe. Déterminer la charge d'un acide aminé.")
st.write("K-Means :")
kmeans0 = KMeans(n_clusters=6, random_state=150, n_init=50).fit(data)
pred = kmeans0.predict(data)
st.write("K-Means Silhouette Score : ", silhouette_score(data, pred))
st.write("K-Means Hierarchical Clustering Contingence table : ")
contingence = pd.crosstab(label[0], pred, rownames=['Vrai label\Cluster'])
st.write(contingence)
#Hierar
sil = []
clus = []
for i in ['average','single','complete','ward']:
    Z1 = sch.linkage(data, method=i)
    labels_pred = fcluster(Z1, t=6, criterion='maxclust')
    clus.append(i)
    sil.append(silhouette_score(data, labels_pred))
fig, ax = plt.subplots()
ax.bar(clus, sil)
st.pyplot(fig)
st.write("Je choisis linkage=ward grâce à la méthode du coude convexe car le silhouette score est à maximiser.")
st.write("Hierarchical Clustering :")
Z1 = sch.linkage(data, method='ward')
labels_pred = fcluster(Z1, t=6, criterion='maxclust')
st.write("Hierarchical Clustering Silhouette Score : ", silhouette_score(data, labels_pred))
st.write("Hierarchical Clustering Contingence table : ")
contingence = pd.crosstab(label[0], labels_pred, rownames=['Vrai label\Cluster'])
st.write(contingence)

st.markdown("#### 2")
poly = PolynomialFeatures(degree=2, include_bias=False)
data_inter = poly.fit_transform(data)
feature_names = poly.get_feature_names_out(data.columns)
X_train, X_test, y_train, y_test = train_test_split(data_inter, label, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
st.write("Score of Linear Regression with polynomial features")
st.write("Linear Regression Accuracy : ", metrics.mean_absolute_error(y_test,y_pred))
st.write("Linear Regression R^2 Score : ", metrics.r2_score(y_test,y_pred))
st.write("L'accuracy de ce modèle est très satisfaisante, même si pour un domaine aussi subtile que la chimie, il faudrait faire davantage de recherche pour atteindre des résultats aussi haut que possible. De plus le R^2 est très mauvais.")

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
st.write("Decision Tree Regressor : ")
st.write("Decision Tree Regressor Accuracy : ", metrics.mean_absolute_error(y_test,y_pred))
st.write("Decision Tree Regressor R^2 Score : ", metrics.r2_score(y_test,y_pred))
st.write("L'accuracy est encore meilleur que le modèle précédents K-Means. Le NCISC est une des features avec le moins de corrélation. Une séléction des features pourrait être source de meilleurs résultats.")

st.markdown("#### 3")
st.info("Compte-Rendu et Analyse")
st.write("Les informations ont été intégrés dans le streamlit au fil de l'avancée du TP. Avec plus de temps et un sujet aussi intéressant que celui-ci, nul doute que le travail consistant à améliorer la recherche du bon modèle sera utile et aura des applications très porteuses.")

st.header("")
st.subheader("Marwan BENLKHIR")