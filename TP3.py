import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
from sklearn.naive_bayes import CategoricalNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


st.header("Machine Learning I")
st.header("TP3 - Classification Supervisée")
st.header("")
st.subheader("Marwan BENLKHIR")

st.header("Part I - Titanic")

dftitt = pd.read_csv("9, 10/titanic_train.csv", sep=',')

st.subheader("Analyse préliminaire")

st.info("1")
st.write("titanic train info :")
st.write(dftitt.dtypes)
#st.write(dftitt.summary())
st.write("titanic train description :")
st.write(dftitt.describe())
fig, ax = plt.subplots(figsize=(30, 5))
plot = sns.countplot(hue='Survived', x='Age', data=dftitt)
st.write("titanic train survived over age :")
st.pyplot(plot.figure)
st.write("titanic train description of null data per column :")
st.write(dftitt.isnull().sum())

st.info("2")
st.write("Died : ", dftitt['Survived'].value_counts()[0])
st.write("Survivor : ", dftitt['Survived'].value_counts()[1])
st.write("Died : ", dftitt['Survived'].value_counts(normalize=True)[0]*100," %")
st.write("Survivor : ", dftitt['Survived'].value_counts(normalize=True)[1]*100," %")

st.subheader("Les femmes et les enfants d'abord !")

st.info("1")
st.write("Men survivor : ", dftitt.where(dftitt['Sex']=='male')['Survived'].value_counts(normalize=True)[1]*100, "%")
st.write("Women survivor : ", dftitt.where(dftitt['Sex']=='female')['Survived'].value_counts(normalize=True)[1]*100, "%")

st.info("2")
dftitt["Child"] = dftitt["Age"].apply(lambda x : 'child' if x<18 else 'adult')
st.write(dftitt['Child'])
st.write("Adult survivor : ", dftitt.where(dftitt['Child']=='adult')['Survived'].value_counts(normalize=True)[1]*100, "%")
st.write("Child survivor : ", dftitt.where(dftitt['Child']=='child')['Survived'].value_counts(normalize=True)[1]*100, "%")

st.info("3")
st.write("L'hypothèse potentiellement biaisée qui a été considéré est qu'il n'existe que deux genres, puis un autre biai peut être présent s'il on ne nettoie pas les données, c'est celui de la qualité des instruments de mesure.")

st.info("4")
st.write("Boy and girls survivor : ")
st.write(dftitt[["Sex",'Child','Survived']].groupby(['Child','Sex'])['Survived'].value_counts(normalize=True)*100)
st.write("La politique a certainement eu un impact sur les statistiques de survie")

st.info("5")
dftite = pd.read_csv("9, 10/titanic_test.csv", sep=',')
dftite["Child"] = dftite["Age"].apply(lambda x : 'child' if x<18 else 'adult')
dftitt["Child"] = dftitt["Child"].astype('category')
dftitt["Survived"] = dftitt["Survived"].astype('category')
dftite["Child"] = dftite["Child"].astype('category')
dftite["Survived"] = dftite["Survived"].astype('category')
st.write("Fait")

st.info("6")
st.write("Child, Sex et Survived :")
X_train = pd.get_dummies(dftitt[["Child","Sex"]])
y_train = dftitt["Survived"]
classifier = CategoricalNB()
classifier.fit(X_train.values, y_train.values)
st.write("class_log_prior_ :")
st.write(classifier.class_log_prior_)
X_test = pd.get_dummies(dftite[["Child","Sex"]])
y_pred = classifier.predict(X_test)
y_test = dftite["Survived"]
pred = classifier.predict(X_test)
st.write("Classification report of the classifier :")
st.write(classification_report(y_test, pred))

st.subheader("Taux de survie et classe sociale")

st.info("1")
fig2, ax2 = plt.subplots(figsize=(30, 5))
plot1 = sns.countplot(hue='Survived', x='Pclass', data=dftitt)
st.write("titanic train survived over class :")
st.pyplot(plot1.figure)

st.info("2")
dftitt["Pclass"] = dftitt["Pclass"].astype('category')
dftitt["Fare2"] = pd.cut(dftitt["Fare"], bins=[-1,10,20,30,600], labels=["-10","10-20","20-30","30+"])
dftitt["Fare2"] = dftitt["Fare2"].astype('category')

st.info("3")
st.write("Chi2 between Pclass and Fare2 :")
cont_table = pd.crosstab(dftitt["Pclass"], dftitt["Fare2"])
chi2, p, dof, expected = chi2_contingency(cont_table)
st.write("Chi² statistic =", chi2)
st.write("p-value =", p, " , alors les deux features sont indépendantes.")
st.write("Degrees of freedom =", dof)
st.write("Expected frequencies =\n", expected)

st.info("4")
for cat in dftitt["Fare2"].unique():
    st.write(cat," survivor : ", dftitt.where(dftitt['Fare2']==cat)['Survived'].value_counts(normalize=True)[1]*100, "%")
st.write("La classe n'a pas de lien avec la survie.")

st.info("5")
st.write("Pclass, Fare2 et Survived :")
X_train2 = pd.get_dummies(dftitt[["Pclass","Fare2"]])
y_train = dftitt["Survived"]
classifier = CategoricalNB()
classifier.fit(X_train2.values, y_train.values)
st.write("class_log_prior_ :")
st.write(classifier.class_log_prior_)
dftite["Fare2"] = pd.cut(dftite["Fare"], bins=[-1,10,20,30,600], labels=["-10","10-20","20-30","30+"])
dftite["Fare2"] = dftite["Fare2"].astype('category')
dftite["Pclass"] = dftite["Pclass"].astype('category')
X_test2 = pd.get_dummies(dftite[["Pclass","Fare2"]])
y_pred = classifier.predict(X_test2)
y_test = dftite["Survived"]
pred = classifier.predict(X_test2)
st.write("Classification report of the classifier :")
st.write(classification_report(y_test, pred))
st.write("Une classification sur les features Sex et isChild permet de mieux déterminer la survie qu'avec Pclass et Fare2.")

st.subheader("Modèle mixte et arbres de décision")

st.info("1")
# set train
st.write(dftitt.columns)
X_train3 = pd.concat([pd.get_dummies(dftitt[["Sex"]]), X_train, X_train2], axis=1).dropna()
y_train3 = dftitt["Survived"]
# train
classifier3 = CategoricalNB()
classifier3.fit(X_train3.values, y_train3.values)
st.write("class_log_prior_ model with all data available :")
st.write(classifier3.class_log_prior_)
# set test
X_test3 = pd.concat([pd.get_dummies(dftite[["Sex"]]), X_test, X_test2], axis=1).dropna()
y_test3 = dftite["Survived"]
y_pred3 = classifier3.predict(X_test3)
# predict
st.write("Classification report of the classifier train with all features:")
st.write(classification_report(y_test3, y_pred3))
st.write("Ce modèle à la même accuracy que le premier modèle composé de isChil et Sex. Néanmoins ici la moyenne de poids est plus basse.")

st.info("2")
#Pclass, Fare, Age, Sex
st.write("Number of missing value in column Age over total number of value. ")
st.write(pd.concat([dftitt[["Pclass","Fare","Age","Sex"]].isnull().sum(),dftitt[["Pclass","Fare","Age","Sex"]].count()], axis=1))
st.write("Etant donné qu'il ne manque aucune autre valeur dans les autres colonnes, je choisis de remplacer les valeurs manquantes par des données générées aléatoirement à partir de la moyenne et de l'écart-type. Pareil pour X_tes : ")
dftitt.loc[dftitt["Age"].isnull(), "Age"] = np.random.normal(dftitt["Age"].mean(), dftitt["Age"].std(), dftitt["Age"].isnull().sum())
X_train4 = dftitt[["Pclass","Fare","Age","Sex"]]
X_train4["Sex"] = X_train4["Sex"].map({"male": 1, "female": 0})
y_train4 = dftitt["Survived"]
st.write(pd.concat([dftite[["Pclass","Fare","Age","Sex"]].isnull().sum(),dftite[["Pclass","Fare","Age","Sex"]].count()], axis=1))
dftite.loc[dftite["Age"].isnull(), "Age"] = np.random.normal(dftite["Age"].mean(), dftite["Age"].std(), dftite["Age"].isnull().sum())
X_test4 = dftite[["Pclass","Fare","Age","Sex"]]
X_test4["Sex"] = X_test4["Sex"].map({"male": 1, "female": 0})
y_test4 = dftite["Survived"]
DT = DecisionTreeClassifier().fit(X_train4,y_train4)
y_pred4 = DT.predict(X_test4)
figure4 = plt.figure(figsize=(20,10))
thistree = tree.plot_tree(DT)
st.pyplot(figure4)
st.write("Accuracy : ", metrics.accuracy_score(y_test4, y_pred4))
st.write("On obtient pratiquement les mêmes performances.")

st.info("3")
st.write("Number of missing value in column Age over total number of value. ")
st.write(pd.concat([dftitt[["Pclass","Fare","Age","Sex","Embarked","SibSp","Parch","Cabin"]].isnull().sum(),dftitt[["Pclass","Fare","Age","Sex","Embarked","SibSp","Parch","Cabin"]].count()], axis=1))
st.write("Il est évident que de remplacer la valeur manquante par son mode pour Embarked est une bonne idée. Parcontre pour Cabin il est préférable de supprimer la colonne.")
dftitt.loc[dftitt["Age"].isnull(), "Age"] = np.random.normal(dftitt["Age"].mean(), dftitt["Age"].std(), dftitt["Age"].isnull().sum())
X_train5 = dftitt[["Pclass","Fare","Age","Sex","Embarked","SibSp","Parch"]]
dftitt.loc[dftitt["Embarked"].isnull(), "Embarked"] = dftitt["Embarked"].mode()[0]
X_train5 = pd.concat([pd.get_dummies(dftitt[["Embarked"]]), X_train5], axis=1)
X_train5 = X_train5.drop(columns=["Embarked"])
X_train5["Sex"] = X_train5["Sex"].map({"male": 1, "female": 0})
y_train5 = dftitt["Survived"]
# df test
dftite.loc[dftite["Age"].isnull(), "Age"] = np.random.normal(dftite["Age"].mean(), dftite["Age"].std(), dftite["Age"].isnull().sum())
X_test5 = dftite[["Pclass","Fare","Age","Sex","Embarked","SibSp","Parch"]]
dftite.loc[dftite["Embarked"].isnull(), "Embarked"] = dftite["Embarked"].mode()[0]
X_test5 = pd.concat([pd.get_dummies(dftite[["Embarked"]]), X_test5], axis=1)
X_test5 = X_test5.drop(columns=["Embarked"])
X_test5["Sex"] = X_test5["Sex"].map({"male": 1, "female": 0})
y_test5 = dftite["Survived"]
DT2 = DecisionTreeClassifier().fit(X_train5,y_train5)
y_pred5 = DT2.predict(X_test5)
figure5 = plt.figure(figsize=(20,10))
thistree = tree.plot_tree(DT2)
st.pyplot(figure5)
st.write("Accuracy : ", metrics.accuracy_score(y_test5, y_pred5))
st.write("C'est inédit car on a jamais obtenu une meilleure accuracy avec ce modèle que tous les modèles précédents. ")


st.header("Part II - Risque Cardiaque")

dfhrt = pd.read_csv("9, 10/heart-disease-UCI.csv", sep=',')
# age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target

st.subheader("Analyse préliminaire et facteurs de risque")

st.info("1")
st.write("heart disease info :")
st.write(dfhrt.dtypes)
#st.write(dfhrt.summary())
st.write("heart disease description :")
st.write(dfhrt.describe())
fig10, ax10 = plt.subplots(figsize=(30, 5))
plot10 = sns.countplot(hue='target', x='age', data=dfhrt)
st.write("heart disease over age :")
st.pyplot(plot10.figure)
st.write("heart disease description of null data per column :")
st.write(dfhrt.isnull().sum())

st.info("2")
st.write("Number of heart disease cases : ", dfhrt['target'].value_counts()[1], " and in percentage : ", dfhrt['target'].value_counts(normalize=True)[1]*100, " %")
st.write("Etant donné ces propotions, on peux dire que le dataset n'entrainera pas un important biais du survivant, c'est à dire que qu'on aurait pu trouver que 90% des gens sont atteints de heart disease alors que c'est uniquement dû au fait que les données sont collectées chez des personnes potentiellement atteintes. Ici, environ la moitié est atteinte. Une chose est sûr c'est qu'il n'y pas assez de données pour faire de l'inférence, pour généraliser à la population.")

st.info("3")
X_train6, X_test6, y_train6, y_test6 = train_test_split(dfhrt.drop(["target"], axis=1), dfhrt["target"], test_size=0.2, random_state=42)

st.subheader("Taux de risque selon le sexe et l'âge")

st.info("1")
st.write("Number of heart disease cases of male : ", dfhrt[dfhrt['sex'] == 1]['target'].value_counts()[1], " and in percentage : ", dfhrt[dfhrt['sex'] == 1]['target'].value_counts(normalize=True)[1]*100, " %")
st.write("Number of heart disease cases of female : ", dfhrt[dfhrt['sex'] == 0]['target'].value_counts()[1], " and in percentage : ", dfhrt[dfhrt['sex'] == 0]['target'].value_counts(normalize=True)[1]*100, " %")
fig11, ax11 = plt.subplots(figsize=(30, 5))
plot11 = sns.countplot(hue='target', x='sex', data=dfhrt)
st.write("heart disease over sex :")
st.pyplot(plot11.figure)

st.info("2")
dfhrt["Age_Group"] = pd.cut(dfhrt["age"], bins=[25,40,55,100], labels=["-40","40-55","+55"])

st.info("3")
for cat in dfhrt["Age_Group"].unique():
    st.write("Age group ", cat, " heart disease cases : ", dfhrt.where(dfhrt['Age_Group']==cat)['target'].value_counts(normalize=True)[1]*100, "%")

st.info("4")
st.write(dfhrt[["sex",'Age_Group','target']].groupby(['Age_Group','sex'])['target'].value_counts(normalize=True)*100)

st.subheader("Taux de risque et cholestérol")

st.info("1")
for cat in dfhrt["cp"].unique():
    st.write("Chest pain type ", cat, " heart disease cases : ", dfhrt.where(dfhrt['cp']==cat)['target'].value_counts(normalize=True)[1]*100, "%")
st.write("Des informations sur l'encodage de cette variable sont essentielles. Est-cequ'il s'agit de catégire où chaque valeur est un type de douleur ou un niveau d'intensité de la douleur. S'il s'agit d'un niveau d'intensité, les données montrent des tendances qu'on ne pourrait pas apriori penser puisque les chances de maladie cardiaque ne correspondent pas au type de douleur à la poitrine hormis pour l'absence de douleur où le taux est bien plus bas que pour les autres. Pour le reste du TP, chest pain sera traité comme un niveau d'intensité de la douleur")
st.info("2")
dfhrt["Cholesterol_Range"] = pd.cut(dfhrt["chol"], bins=[-1, 200, 240, 600], labels=["-200","200-240","240+"])

st.info("3") # Sex, age-group, cp, cholrange
X = dfhrt.drop(["target"], axis=1)
for cat in ["Age_Group", "Cholesterol_Range"]:
    X[cat] = X[cat].astype('category')
    X = pd.concat([pd.get_dummies(X[[cat]]), X], axis=1)
X = X.drop(columns=['age','Age_Group', 'Cholesterol_Range','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'])
X_train7, X_test7, y_train7, y_test7 = train_test_split(X, dfhrt["target"], test_size=0.2, random_state=42) # This drop categorical columns
classifier10 = CategoricalNB()
classifier10.fit(X_train7.values, y_train7.values)
y_pred10 = classifier10.predict(X_test7.values)

st.info("4")
st.write(classification_report(y_test7, y_pred10))

st.subheader("Modèle mixte et arbres de décision")

st.info("1")
# age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
X2 = dfhrt.drop(["target"], axis=1)
scaler = StandardScaler()
for cat in ["Age_Group", "Cholesterol_Range"]:
    X2[cat] = X2[cat].astype('category')
    X2 = pd.concat([pd.get_dummies(X2[[cat]]), X2], axis=1)
X2[['cp','trestbps','restecg','thalach','oldpeak','slope','ca','thal']] = scaler.fit_transform(X2[['cp','trestbps','restecg','thalach','oldpeak','slope','ca','thal']])
X2 = X2.drop(columns=['age','Age_Group', 'Cholesterol_Range','chol'])
st.write(X2)
X_train8, X_test8, y_train8, y_test8 = train_test_split(X2, dfhrt["target"], test_size=0.2, random_state=42)

st.info("2")
DT3 = DecisionTreeClassifier().fit(X_train8,y_train8)
y_pred6 = DT3.predict(X_test8)
st.write("Accuracy : ", metrics.accuracy_score(y_test8, y_pred6))

st.info("3")
figure5 = plt.figure(figsize=(20,10))
thistree2 = tree.plot_tree(DT3)
st.pyplot(figure5)

st.info("4")
st.write("Categorical NB : ",classification_report(y_test7, y_pred10))
st.write("Decision Tree : ", metrics.accuracy_score(y_test8, y_pred6))
st.write("Cette fois-ci le modèle de Naive Bayes categoriel avec un sélection restreinte de données est meilleur que le Decision Tree avec le maximum de données disponible. De plus, des modèle avec une accuracy de 87% sur la détection de maladies cardiques ne sont pas intéressants. Sur un sujet tel que celui la, il est nécessaire d'avoir une accuracy maximale, en particulier sur la prédiction de VraiPositifs et FauxNegatifs, donc Recall et Miss Rate. Il n'y a pas de modèle par essence meilleur que les autres. Il s'agit d'appilquer la bonne sélection de données et d'utiliser le modèle adapté pour obtenir les meilleurs performances. ")

st.subheader("Marwan BENLKHIR")
