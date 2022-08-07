    # on importe les librairies
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import base64
from sklearn.preprocessing import OrdinalEncoder


#### Arrière plan :  

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('fond.jpg')

### Chargement de notre modèle de prédiction ###
file1 = open('laptop_prediction.pkl', 'rb')
forest = pickle.load(file1)
file1.close()

# Création titre de la page
st.title('Bienvenue') 
st.write('Prédiction Laptop')
st.info('Veuillez selectionner les caractéristiques de votre appareil :')

### Ajout d'une image
img = Image.open('laptop.jpg')
st.image(img, width=600) 






df = pd.read_csv('laptop_data.csv', index_col=0)

### Transformation des features :

#Reduction des modalités 
Gpu_enc = []
for i in df.Gpu.str.split(' '):
    Gpu_enc.append(i[0])   
Cpu_enc = []
for i in df.Cpu.str.split(' '):
    Cpu_enc.append(i[0])  
df.Gpu, df.Cpu = Gpu_enc, Cpu_enc    

Screen_enc = df.ScreenResolution.str.rsplit(n = 1, expand=True)[0]
Memory_enc = df.Memory.str.rsplit(n=0, expand=True)[0]
df['Memory'] = Memory_enc

    #Nettoyage des données : 
df['Weight'] = df.Weight.str.replace('kg', '').astype(float)

    ##Separation des features continues et categoriques
var_cat, var_cont = [], []
for i in df.nunique().sort_values().index:
    if df[i].nunique()>40:
        var_cont.append(i)
    else:
        var_cat.append(i)

    #Suppression des features inutilisées
df = df.drop(['Weight', 'Price'], axis = 1)
data = df.copy()

##Récupération des features : 
var_choice = {}
for i in df[var_cat]:
    var_choice[i] = list(df[i].unique())

col_var_choice = var_cat.copy()

var_cat[0] = st.selectbox(str(var_cat[0]),
                                sorted(var_choice[var_cat[0]]))
var_cat[1] = st.selectbox(str(var_cat[1]),
                                sorted(var_choice[var_cat[1]]))
var_cat[2] = st.selectbox(str("Type"),
                               sorted(var_choice[var_cat[2]]))
var_cat[3] = st.selectbox(str(var_cat[3]),
                                sorted(var_choice[var_cat[3]]))
var_cat[4] = st.selectbox(str("Système d'exploitation"),
                                sorted(var_choice[var_cat[4]]))
var_cat[5] = st.selectbox(str("Capacité de stockage"),
                                sorted(var_choice[var_cat[5]]))
var_cat[6] = st.selectbox(str("Taille de l'écran"),
                                sorted(var_choice[var_cat[6]]))
var_cat[7] = st.selectbox(str("Compagnie"),
                                sorted(var_choice[var_cat[7]]))
var_cat[8] = st.selectbox(str("Résolution d'écran"),
                                sorted(var_choice[var_cat[8]]))
#on transforme var_cat en format compatible 
var_cat = pd.DataFrame(np.array(var_cat).reshape(1,len(var_cat)), columns=col_var_choice,index = ["utilisateur"])

#on met les colonne de df dans le meme ordre 
df = df[col_var_choice] 

#on concatene les valeurs utilisateurs contenus dans var_cat et df :
df = pd.concat([df,var_cat])

#on remet inches au bon types :
df['Inches'] = df['Inches'].astype(float)

#encodage des données 
var_a_enc = df.describe(include=('object')).columns

encoder = OrdinalEncoder()
df[var_a_enc] = encoder.fit_transform(df[var_a_enc])

#normalisation

var_a_norm = []
for i in df:
    if df[i].nunique()>10:
        var_a_norm.append(i)


#Normalisation du df
df[var_a_norm]  = (df[var_a_norm] - df[var_a_norm].mean())/df[var_a_norm].std()


#remettre les colonnes dans l'ordre de départ :

df = df[data.columns]

#recuperation des données entrées par l'utilisateur 

utilisateur = df.iloc[-1, :]


## Prediction 

if(st.button('Prédiction :')):
    prediction = forest.predict([utilisateur])[0]
    st.success(f"Prix estimé de votre appareil : {round(prediction,2)} $  \n Soit {round(prediction*133.88,2)} ₹")

