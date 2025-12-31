import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import statsmodels.api as sm

st.header("Machine Learning I")
st.header("TP2.2 - Régression Linéaire")
st.header("")
st.subheader("Marwan BENLKHIR")

# windows import file
#dfbat = pd.read_csv("tabBats.txt")
# macos import file
dfbat = pd.read_csv("6, 7, 8/tabBats.txt", sep=' ')
st.header("")
st.header("")

st.info("Partie A - 2")
st.write("First lines")
st.write(dfbat.head())
st.write("Features types")
st.dataframe(dfbat.dtypes)
st.write("Basic statistics")
st.write(dfbat.describe())

st.info("Partie A - 3")
dfLight = dfbat[['Species','BOW','BRW']]
st.write("Dataset first lines without unrelevant features")
st.write(dfLight.head())

st.info("Partie B - 1")
unique = np.unique(dfLight["Species"])
#dfLight['Species_mapped'] = dfLight["Species"].astype('category')
fig, ax = plt.subplots(figsize=(15,15))
scatter = ax.scatter(dfLight['BRW'], dfLight['BOW'], c=dfLight['Species'].astype('category').cat.codes, cmap="nipy_spectral")
cl =plt.colorbar(scatter, ticks=range(len(unique)), label="Category")
cl.ax.set_yticklabels(unique)
ax.set_title("Scatter Plot")
ax.set_xlabel("Axe BRW (Brain mass)")
ax.set_ylabel("Axe BOW (Body mass)")
st.pyplot(fig)
st.write("On peut estimer de manière empirique que la brain mass et la body mass evoluent de manière linéaire. On peut affirmer que body mass et brain mass ont une covariance positive. Aussi une espèce particulière se différencie énormement des autres, on la qualifie donc d'outlier.")

st.info("Partie B - 3")
dfLight["One"] = 1
model = sm.OLS( dfLight['BOW'], dfLight[['BRW','One']]).fit()
st.write(model.summary())
st.write("BRW a un coefficient de 0.1056 donc a un effet positif et le coef de Beta 0 est de -61.4443. Le t-test et P>|t| montrent que BRW et l'intercept agissent significativement sur BOW. Selon R^2, BRW explique 95% de la variance de BOW. Le skewness montre que le résidu est asymétrique à gauche, le kutosis que le résidu est aplati mais les analyse de résidu montre qu'il n'y a rien de très important.")
x = np.linspace(1, 8000, 100)
y = 0.1056 * x -61.4443
plt.plot(x,y, label='Line')
scatter = ax.scatter(dfLight['BRW'], dfLight['BOW'], c=dfLight['Species'].astype('category').cat.codes, cmap="nipy_spectral")
st.info("Partie B - 4")
st.pyplot(fig)

st.info("Partie C - 1")
dfLightLessP = dfLight.where(dfLight['Species'] != 'Pteropus vampyrus').dropna()
st.write(dfLightLessP)

st.info("Partie C - 2")
dfLight['Pteropus vampyrus'] = 1
dfLightLessP['Pteropus vampyrus'] = 0
dfall = pd.concat([dfLight,dfLightLessP])
fig, ax = plt.subplots(figsize=(15,15))
scatter = ax.scatter(dfall['BRW'], dfall['BOW'], c=dfall['Pteropus vampyrus'])
plt.legend(handles=scatter.legend_elements()[0], labels=['without &Pteropus vampyrus', 'Pteropus vampyrus'])
ax.set_title("Scatter Plot")
ax.set_xlabel("Axe BRW (Brain mass)")
ax.set_ylabel("Axe BOW (Body mass)")
st.pyplot(fig)

st.info("Partie C - 3")
dfLightLessP["One"] = 1
model2 = sm.OLS( dfLightLessP['BOW'], dfLightLessP[['BRW','One']]).fit()
st.write(model2.summary())

#stats = pd.DataFrame([model.params.to_frame().T, model.rsquared, model.resid.T], index='OLS with Pteropus vampyrus')
#stats.insert( [model.params.to_frame().T, model.rsquared, model.resid.T], index='OLS with Pteropus vampyrus')

#stats = pd.DataFrame({'OLS with Pteropus vampyrus' : [model.params, model.rsquared, model.resid], 'OLS without Pteropus vampyrus' : [model2.params, model2.rsquared, model2.resid]})
statsCompare = pd.DataFrame({
    'OLS with Pteropus vampyrus': model.params,
    'OLS without Pteropus vampyrus' : model2.params
})
statsCompare.loc['R²']= [model.rsquared, model2.rsquared]
statsCompare.loc['MSE']= [model.mse_resid, model2.mse_resid]
statsCompare.loc['SSR']= [model.ssr, model2.ssr]
st.write(statsCompare)


st.info("Partie C - 4")
x1 = np.linspace(1, 8000, 100)
y1 = 0.0674 * x -22.1491
plt.plot(x,y, label='OLS with Pteropus vampyrus')
plt.plot(x1,y1, color='red', label='OLS without Pteropus vampyrus')
plt.legend()
scatter = ax.scatter(dfLight['BRW'], dfLight['BOW'])
st.pyplot(fig)

st.header("")
st.subheader("Marwan BENLKHIR")