from numpy.random.mtrand import RandomState
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns


st.write("""
# Simple KNN App

""")

st.sidebar.header('User Input Parameters')

def user_input_hiper_p():
    nro_muestra = st.sidebar.slider('Numero de datos', 15, 300)
    K = st.sidebar.slider('Neighbours', 1, round(nro_muestra * 0.7))
    
    return nro_muestra, K

nro_muestra, K = user_input_hiper_p()


X, y = make_blobs(n_samples=nro_muestra, n_features=2, centers=[[2,2],[1,1]], shuffle=False, random_state=7)

df = pd.DataFrame(X, columns=['x1', 'x2'])
df['y'] = y


X_train, X_test, y_train, y_test = train_test_split(df.drop('y', axis=1), df['y'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=K)

knn.fit(X_train, y_train)


# Definimos el tamaño de la figura
fig = plt.figure(figsize=(10,8))


# Definimos una grilla de valores que abarcan todo el rango de cada variable
x1_min, x1_max = X_train['x1'].min() - 1, X_train['x1'].max() + 1
x2_min, x2_max = X_train['x2'].min() - 1, X_train['x2'].max() + 1
x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, .1), np.arange(x2_min, x2_max, .1))

# Predecimos a partir de los valores de la grilla
Z = knn.predict(np.c_[x1.ravel(), x2.ravel()])
Z = Z.reshape(x1.shape)
# Representamos las áreas de influencia de cada clase
plt.pcolormesh(x1, x2, Z, cmap = ListedColormap(sns.color_palette(n_colors=2)), alpha=0.2, shading='auto')

# Visualizamos los datos de entrenamiento
sns.scatterplot(x=X_train['x1'], y=X_train['x2'], hue=y_train, s=75)



# Definimos los rótulos del gráfico
plt.xlabel(f'$x_1$', fontsize=15)
plt.ylabel(f'$x_2$', fontsize=15)
plt.title('Fronteras de decisión', fontsize=15);



st.subheader('Prediction')
st.pyplot(fig)