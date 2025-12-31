import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score 
import numpy as np 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error

# windows import file
#df = pd.read_csv('cleaned_cereal.csv')
# macos import file
df = pd.read_csv('3, 4, 5/cleaned_cereal.csv')

#delete raw without rating
df = df.dropna(subset=['rating'])
rating = df['rating']
df = df.drop('rating', axis=1)
# categorical to integer and keep the mapping
mappings = {}
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    cats = df[col].astype('category')
    df[col + '_mapped'] = cats.cat.codes
    mappings[col] = dict(enumerate(cats.cat.categories))
    df = df.drop(col, axis=1)

st.write("df shape:", df.shape)
st.write("rating shape:", rating.shape)

df, df_test, rating, rating_test = train_test_split(df, rating, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(df, rating)

df_coef_rat = pd.DataFrame({
    'feature': df.columns,
    'coef': model.coef_
})
st.write("coef of all feature on rating")
st.write(df_coef_rat.sort_values(by=['coef'], ascending=0))

#test des interactions
    # create all interactions + all features model + all features^2
poly = PolynomialFeatures(degree=2, include_bias=False)
df_inter = poly.fit_transform(df)
feature_names = poly.get_feature_names_out(df.columns)

    #testing the previous model with rating
model2 = LinearRegression()
model2.fit(df_inter, rating)
coef_dfIntéraction = pd.DataFrame({'feature': feature_names, 'coef': model2.coef_})
st.write("coef of all feature and all interactions on rating")
st.write(coef_dfIntéraction.sort_values(by=['coef'], ascending=0))

# Scores
st.write("Score of Linear Regression")
st.write(model.score(df_test,rating_test))

# prediction 
rating_test_pred = model.predict(df_test)
st.write("MAE")
st.write(mean_absolute_error(rating_test, rating_test_pred))
st.write("RMSE")
st.write(root_mean_squared_error(rating_test, rating_test_pred))
#st.write("Score of Nonlinear regression")
#st.write(model2.score(df_test,rating_test))

modelcv = LinearRegression()
cvscores = cross_val_score(modelcv, df, rating, cv=10)
st.write("Cross validation 5folds R^2 mean")
st.write(cvscores.mean())

'''
# accuracy
print('Accuracy: %.3f ,\nStandard Deviations :%.3f' %
      (mean(scores), std(scores)))


st.write("New models ")

#Modèle 2 et split 2 à écrire

model2 = LinearRegression()
model2.fit(df2, rating2)
st.write(model2.columns)

st.write("Model scores")
scores = cross_val_score(model2, df2_test, rating2_test, cv=(KFold(n_splits=5, shuffle=True, random_state=42)), scoring='r2')
average_r2 = np.mean(scores) 
st.write(f"R² Score for each fold: {[round(score, 4) for score in scores]}")
st.write(f"Average R² across 5 folds: {average_r2:.2f}")


plt.plot(df2, model2.predict(df2), color='red')
plt.show()

model3 = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model3.fit(df2, rating2)
y_pred = model2.predict(df2)
plt.plot(df2, y_pred, color='red', label='Polynomial fit')
plt.legend()
plt.show()
'''