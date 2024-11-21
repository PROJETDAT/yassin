



import streamlit as st

st.markdown(
    """
    <style>
    /* Style pour personnaliser l'apparence de la barre latérale */
    .css-1d391kg { /* Cette classe représente la barre latérale */
        background-color: #ADD8E6;  /* Change ici la couleur d'arrière-plan de la barre latérale */
        border: 1px solid #2E8B57;  /* Optionnel : ajouter une bordure */
    }

    /* Changer la couleur du texte dans la barre latérale */
    .css-1d391kg .css-3o5yew {
        color: #ADD8E6;  /* Change ici la couleur du texte des éléments dans le sommaire */
    }

    /* Changer la couleur des titres dans la barre latérale (comme "Sommaire") */
    .css-1hynsf2 { 
        color: #ADD8E6;  /* Change ici la couleur du texte des titres */
    }
    </style>
    """,
    unsafe_allow_html=True

)




import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import plotly.express as px


from scipy.stats import skew, kurtosis



# Importation des modèles nécessaires
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Importation des métriques d'évaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



import statsmodels.api as sm



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import streamlit as st








import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, pearsonr
from sklearn.linear_model import LogisticRegression


from scipy.stats import pearsonr, chi2_contingency, ttest_ind, f_oneway


from scipy.stats import pearsonr, chi2_contingency, f_oneway



# Charger les données de la base

df = pd.read_csv("donnees_marketing_banque.csv", sep =";")

# supprimer les valeurs manquantes

df_1 = df.dropna(axis = 0)




descriptions = {
    "age": "Âge du client.",
    "job": "Type de métier du client (ex. 'manager', 'blue-collar', 'retired').",
    "marital": "État civil du client (ex. 'married', 'single', 'divorced').",
    "education": "Niveau d'études du client (ex. 'secondary', 'primary', 'tertiary').",
    "default": "Présence ou absence de défaut de paiement de crédit (ex. 'yes', 'no').",
    "balance": "Solde annuel moyen sur le compte du client.",
    "housing": "Présence ou absence de prêt immobilier (ex. 'yes', 'no').",
    "loan": "Présence ou absence de prêt personnel (ex. 'yes', 'no').",
    "contact": "Moyen de communication avec le client (ex. 'telephone', 'cellular', 'unknown').",
    "duration": "Durée de la dernière communication, en secondes.",
    "campaign": "Nombre de contacts effectués pendant cette campagne pour ce client.",
    "previous": "Nombre de contacts réalisés avant cette campagne pour ce client.",
    "poutcome": "Résultat de la campagne marketing précédente (ex. 'failure', 'nonexistent', 'success').",
    "y": "Le client a-t-il souscrit un dépôt à terme ? (variable cible, ex. 'yes', 'no')."
}





# Créer des pages
st.sidebar.title("Sommaire")
pages = ["Contexte du projet","Description des variables","Exploration des données",  "Analyse de données", "Modélisation et Prédiction"]
page = st.sidebar.radio("Aller vers la page :", pages)









# Page 1: Contexte du projet
if page == pages[0]:
    # Bannière d'introduction avec un texte accrocheur
    st.markdown("""
        <div style="background-color:#2E8B57;padding:20px;border-radius:10px">
        <h1 style="color:white;text-align:center;">📈 Projet Marketing Bancaire : Améliorer les Campagnes de Souscription</h1>
        </div>
    """, unsafe_allow_html=True)

    st.write(" ")

    # Présentation avec un résumé stylisé
    st.markdown("""
    <div style="border-left: 6px solid #2E8B57; padding-left: 15px; margin: 20px 0; font-size: 16px;">
        <p style="font-family: Arial, sans-serif; font-size: 18px;">
            Ce projet analyse l'impact des <b>campagnes de marketing direct</b> menées par une banque pour
            accroître la souscription aux <b>dépôts à terme</b>. 
            L'objectif est d'identifier les éléments de succès et de mettre en place des <i>modèles prédictifs</i>
            capables d'optimiser les résultats de ces campagnes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write(" ")

    # Objectifs du Projet avec décorations
    st.subheader("🎯 Objectifs du Projet")

    # Créer une liste des objectifs avec des descriptions détaillées
    objectives = {
        "📝 Description des Variables": "Comprendre les différentes caractéristiques des clients et les variables utilisées dans le projet, comme l'âge, le statut marital, et d'autres critères démographiques.",
        "🔍 Exploration des Données": "Examiner les données pour détecter des patterns et des tendances, ainsi que pour identifier les anomalies ou les valeurs manquantes.",
        "📊 Analyse des Données": "Comprendre le profil des clients et évaluer les performances des campagnes passées pour en tirer des enseignements.",
        "🤖 Modélisation": "Développer des modèles prédictifs pour estimer la probabilité qu'un client souscrive à un dépôt à terme basé sur ses caractéristiques."
    }

    for objective, description in objectives.items():
        st.markdown(f"""
        <div style="border: 1px solid #2E8B57; padding: 15px; border-radius: 8px; margin: 10px 0; background-color: #F9F9F9;">
            <h4 style="color: #2E8B57; margin-bottom: 5px;">{objective}</h4>
            <p style="font-family: Arial, sans-serif; color: #555;">{description}</p>
        </div>
        """, unsafe_allow_html=True)

    # Image avec légende et séparation visuelle
    st.image("photo.jpg", caption="Illustration d'une campagne de marketing direct", use_column_width=True)
    st.markdown("<hr style='border:1px solid #2E8B57'>", unsafe_allow_html=True)


















# Page 2: Description des variables
elif page == pages[1]:
    st.write("### Description des variables")
    
    # Sélection de la variable
    selected_variable = st.selectbox("Choisissez une variable pour voir sa signification :", list(descriptions.keys()))
    st.write(f"**{selected_variable}** : {descriptions[selected_variable]}")

    # Vérification du type de variable (numérique ou catégorielle)
    if pd.api.types.is_numeric_dtype(df_1[selected_variable]):

        # Affichage de l'histogramme avec Plotly
        #hist_chart = px.histogram(df_1, x=selected_variable, nbins=20, title=f"Distribution de {selected_variable}")




        #x = df_1[selected_variable]
        #x_density = np.linspace(min(x), max(x), 1000)

        # Création de la courbe de densité avec Plotly
        #hist_chart = go.Figure()
        #hist_chart.add_trace(go.Histogram(x=x, histnorm='probability density', nbinsx=20, name='Histogramme'))
        #hist_chart.add_trace(go.Scatter(x=x_density, y=np.exp(-0.5 * ((x_density - np.mean(x)) / np.std(x))**2) / (np.std(x) * np.sqrt(2 * np.pi)), mode='lines', name='Courbe Normale'))




        # Calcul de la skewness et du kurtosis
        x = df_1[selected_variable]
        skewness = skew(x)
        kurt = kurtosis(x)

        # Tracer une courbe de densité pour vérifier la distribution normale
        x_density = np.linspace(min(x), max(x), 1000)

        # Création de la courbe de densité avec Plotly
        hist_chart = go.Figure()
        hist_chart.add_trace(go.Histogram(x=x, histnorm='probability density', nbinsx=20, name='Histogramme'))
        hist_chart.add_trace(go.Scatter(x=x_density, y=np.exp(-0.5 * ((x_density - np.mean(x)) / np.std(x))**2) / (np.std(x) * np.sqrt(2 * np.pi)), mode='lines', name='Courbe Normale'))





        


        # Affichage du boxplot avec Plotly
        boxplot_chart = go.Figure()
        boxplot_chart.add_trace(go.Box(y=df_1[selected_variable], boxmean="sd", name=f"Boxplot de {selected_variable}"))
        boxplot_chart.update_layout(title=f"Boxplot de {selected_variable}")

        # Affichage des deux graphiques (Histogramme et Boxplot)
        st.plotly_chart(hist_chart, use_container_width=True)
        st.write(f"Skewness: {skewness:.2f}")
        st.write(f"Kurtosis: {kurt:.2f}")
        st.plotly_chart(boxplot_chart, use_container_width=True)
        
        
        

    else:
        # Graphique circulaire pour les variables catégorielles (Pie chart avec Plotly)
        pie_chart = px.pie(
            df_1,
            names=selected_variable,
            title=f"Répartition de {selected_variable}",
            hole=0.3  # Pour un style de graphique en anneau (facultatif)
        )
        
        # Affichage du graphique circulaire avec Streamlit
        st.plotly_chart(pie_chart, use_container_width=True)




       

    



    # Dictionnaire des commentaires pour chaque variable
    commentaires = {
        'job': "Plus de la moitié des clients occupent des postes tels que : 21,5 % en tant qu'ouvriers, 21 % en management, 17 % comme techniciens et 11 % comme administrateurs. Les 30 % restants se répartissent sur d'autres types de postes.",
        'marital': "Parmi les clients, 60 % sont mariés, 29 % sont célibataires et 12 % sont divorcés.",
        'education': "En ce qui concerne le niveau d'éducation des clients, 51 % ont un niveau secondaire, 30 % un niveau tertiaire, et 15 % un niveau primaire. Le reste est inconnu.",
        'default': "En ce qui concerne le niveau d'éducation des clients, 51 % ont un niveau secondaire, 30 % un niveau tertiaire, et 15 % un niveau primaire. Le reste est inconnu.",
        'housing': "En ce qui concerne la présence ou l'absence de prêt immobilier, 56 % des clients en possèdent un, tandis que 44 % n'en ont pas.",
        'loan': "En ce qui concerne la présence ou l'absence de prêt personnel, 84 % des clients n'en ont pas, tandis que 16 % en ont un.",
        'contact': "En ce qui concerne le moyen de communication avec les clients, 65 % utilisent un téléphone cellulaire, 6 % un téléphone fixe, et 29 % des cas sont inconnus.",
        'poutcome': "Pour les résultats de la campagne de marketing précédente, 3 % ont été un succès, 11 % un échec, 4 % ont eu d'autres résultats, et le reste est inconnu.",
        'y': "En ce qui concerne la souscription à un dépôt à terme, 88 % des clients n'en ont pas souscrit, tandis que 12 % en ont souscrit un.",
        'age': """
- Moyenne (40,93 ans) : L'âge moyen des individus est d'environ 41 ans.
- Écart-type (10,62 ans) : Les âges sont assez dispersés autour de la moyenne, avec une variation de ±10,62 ans.
- Valeur minimale (10 ans) : Le plus jeune individu a 10 ans.
- Valeur maximale (95 ans) : Le plus vieux individu a 95 ans.
- 25e percentile (33 ans) : 25% des individus sont âgés de 33 ans ou moins.
- Médiane (39 ans) : La moitié des individus ont un âge inférieur ou égal à 39 ans, ce qui est proche de la moyenne.
- 75e percentile (48 ans) : 75% des individus ont 48 ans ou moins.""",
        'duration': """
- Moyenne (258,16 secondes) : La durée moyenne de la dernière communication est de 258,16 secondes (environ 4 minutes et 18 secondes).
- Écart-type (257,54 secondes) : Les durées des communications sont largement dispersées autour de la moyenne, avec une variation de ±257,54 secondes.
- Valeur minimale (0 seconde) : La durée minimale est de 0 seconde, ce qui peut indiquer des communications inexistantes ou de très courte durée.
- Valeur maximale (4918 secondes) : La durée maximale est de 4918 secondes, soit environ 82 minutes.
- 25e percentile (103 secondes) : 25% des communications durent moins de 103 secondes.
- Médiane (50e percentile, 180 secondes) : La moitié des communications durent moins de 3 minutes.
- 75e percentile (319 secondes) : 75% des communications durent moins de 319 secondes (environ 5 minutes et 19 secondes).""",
        'balance': """
- Moyenne (1362,40) : Le solde annuel moyen sur le compte des clients est de 1 362,40 unités monétaires.
- Écart-type (3044,91) : Les soldes sont très dispersés autour de la moyenne, avec une variation importante (±3 044,91).
- Valeur minimale (-8019) : Le solde le plus bas est de -8 019, ce qui indique des clients avec des soldes négatifs importants.
- Valeur maximale (102 127) : Le solde le plus élevé est de 102 127, représentant des clients avec des soldes très élevés.
- 25e percentile (72) : 25% des clients ont un solde inférieur ou égal à 72.
- Médiane (448) : La moitié des clients ont un solde inférieur ou égal à 448.
- 75e percentile (1 428) : 75% des clients ont un solde inférieur ou égal à 1 428.""",
        'campaign': """
- Moyenne (2,76) : En moyenne, chaque client a eu 2,76 contacts pendant cette campagne.
- Écart-type (3,10) : Le nombre de contacts varie beaucoup, avec une dispersion de ±3,10 contacts.
- Valeur minimale (1) : Le client ayant eu le moins de contacts a été contacté 1 fois.
- Valeur maximale (63) : Le client ayant eu le plus de contacts a été contacté 63 fois.
- 25e percentile (1) : 25% des clients ont eu 1 contact ou moins.
- Médiane (2) : La moitié des clients ont eu 2 contacts ou moins.
- 75e percentile (3) : 75% des clients ont eu 3 contacts ou moins.""",
        'previous': """
- Moyenne (0,58) : En moyenne, chaque client a eu 0,58 contact avant cette campagne.
- Écart-type (2,30) : Le nombre de contacts avant la campagne varie beaucoup, avec une dispersion de ±2,30 contacts.
- Valeur minimale (0) : Le client ayant eu le moins de contacts n'a eu aucun contact avant cette campagne.
- Valeur maximale (275) : Le client ayant eu le plus de contacts avant cette campagne a eu 275 contacts.
- 25e percentile (0) : 25% des clients n'ont eu aucun contact avant cette campagne.
- Médiane (0) : La moitié des clients n'ont eu aucun contact avant cette campagne.
- 75e percentile (0) : 75% des clients n'ont eu aucun contact avant cette campagne."""
    }






    # Dictionnaire des interprétations
    interpretations = {
        'age': """L'échantillon présente une large gamme d'âges, allant des jeunes (10 ans) aux personnes âgées (95 ans).
La distribution des âges semble relativement équilibrée, mais l'écart-type élevé indique une certaine variabilité.
La médiane proche de la moyenne suggère que la répartition des âges n'est pas trop biaisée.""",
        'duration': """ Les durées des communications varient considérablement, avec la majorité des communications étant
 relativement courtes (moins de 3 minutes), mais certaines peuvent être exceptionnellement longues, comme le montre la valeur maximale de 82 minutes. 
 La médiane (180 secondes) est bien plus faible que la moyenne (258 secondes), ce qui suggère que la distribution est légèrement biaisée par les valeurs extrêmes. """,
        'balance': """ Les clients ont des soldes très variés, allant de soldes négatifs (-8 019) à soldes très élevés (102 127). La moyenne (1 362,40) est influencée par 
les valeurs extrêmes, tandis que la médiane (448) est plus faible, indiquant que la majorité des clients ont des soldes plus modestes. L'écart-type élevé montre une grande 
dispersion des soldes. """,
        'campaign': """ La majorité des clients ont eu entre 1 et 3 contacts, mais il y a une petite proportion de clients qui ont été contactés de façon beaucoup plus 
fréquente, allant jusqu'à 63 contacts. La moyenne (2,76) est légèrement plus élevée que la médiane (2), ce qui suggère que quelques clients ont été contactés beaucoup plus
 souvent que la plupart.""",
        'previous': """ La majorité des clients n'ont eu aucun contact avant cette campagne, ce qui explique que la moyenne soit proche de 0. Cependant, certains clients 
ont été contactés plusieurs fois avant cette campagne, avec un maximum de 275 contacts, ce qui génère une grande dispersion dans les données. """
    }


    
    # Checkbox pour afficher le résumé des statistiques descriptives
    if st.checkbox("résumé statistique"):
        st.write(f"**{selected_variable}**")
        st.dataframe(df_1[selected_variable].describe())

    
    
    # Checkbox pour afficher le commentaire
    if st.checkbox("Analyse"):
        if selected_variable in commentaires:
            st.write(f"**{selected_variable}** : {commentaires[selected_variable]}")



    # Checkbox pour afficher l'interprétation
    if st.checkbox("Interprétation"):
        if selected_variable in interpretations:
            st.write(f"**{selected_variable}** : {interpretations[selected_variable]}")



    













# Page 3: Exploration des données
elif page == pages[2]:
    st.write("### Exploration des données")

    # Affichage des données
    st.subheader("Aperçu des données")
    st.dataframe(df_1.head())

    # les dimension du dataframe

    if st.checkbox("Afficher les dimensions du dataframe 📏"):
        st.write(f"**Dimensions :** nous avons un total de {df_1.shape[0]} clients et {df_1.shape[1]} variables à analyser.")
        
        
    # les valeurs manquantes

    if st.checkbox("Afficher les valeurs manquantes ❓ "): 
         missing_values = df_1.isna().sum()
         st.write("### Valeurs manquantes :")
         st.dataframe(missing_values[missing_values > 0])  # Afficher uniquement les colonnes avec des valeurs manquantes
         if missing_values.sum() == 0:
            st.write("Nous avons **zéro (0)** valeur manquante dans le dataset. 🎉")
         else:
            st.write(f"Nous avons un total de **{missing_values.sum()}** valeurs manquantes dans le dataset.")

    
    if st.checkbox("Afficher les doublons 🔍 "): 
        num_duplicates = df_1.duplicated().sum()
        st.write(f"### Nombre de doublons : {num_duplicates}")
        if num_duplicates == 0:
            st.write("Nous avons **zéro (0)** doublon dans le dataset. 🎉")
        else:
            st.write(f"Nous avons un total de **{num_duplicates}** doublons dans le dataset.")



















# Page 4: Analyse Bivariée entre les Variables et y (avec y qualitative)
elif page == pages[3]:
    st.write("### Analyse de données")
    # Analyse Bivariée avec la variable cible 'y'
    st.subheader("Analyse Bivariée avec la Variable Cible")
    st.write("Choisissez une variable à analyser par rapport à la variable cible `y` :")

    # Liste déroulante pour sélectionner la variable à comparer avec 'y'
    variable_bivarie = st.selectbox("Choisissez une variable :", [col for col in df_1.columns if col != 'y' and col != 'id'])

    # Vérification du type de la variable sélectionnée
    if variable_bivarie in df_1.select_dtypes(include=['float64', 'int64']).columns:
        # Analyse pour les variables numériques
        st.write(f"### Répartition de `{variable_bivarie}` selon la variable cible `y`")
        st.write(df_1.groupby('y')[variable_bivarie].describe())  # Affichage des stats descriptives par groupe

        # Visualisation : Boîte à moustaches pour comparer les distributions
        fig = px.box(df_1, x='y', y=variable_bivarie, title=f"Distribution de `{variable_bivarie}` par rapport à `y`")
        st.plotly_chart(fig)

        # Test t de Student ou ANOVA
        unique_categories = df_1["y"].nunique()  # Nombre de catégories dans 'y'

        if unique_categories == 2:
            # Si `y` a 2 catégories : Test t de Student
            st.write(f"### Test t de Student pour `{variable_bivarie}` en fonction de `y`")

            # Séparation des données en 2 groupes
            group1 = df_1[df_1["y"] == df_1["y"].unique()[0]][variable_bivarie].dropna()
            group2 = df_1[df_1["y"] == df_1["y"].unique()[1]][variable_bivarie].dropna()

            # Test t de Student
            t_stat, p_value = ttest_ind(group1, group2)
            st.write(f"**Statistique t** : {t_stat:.4f}")
            st.write(f"**Valeur p** : {p_value:.4f}")

            if p_value < 0.05:
                st.write(f"Il existe une différence significative entre les groupes pour `{variable_bivarie}` et `y`.")
            else:
                st.write(f"Aucune différence significative entre les groupes pour `{variable_bivarie}` et `y`.")

        else:
            # Si `y` a plus de 2 catégories : Test ANOVA
            st.write(f"### Test ANOVA pour `{variable_bivarie}` en fonction de `y`")

            # Séparation des données en groupes en fonction des catégories de `y`
            groups = [df_1[df_1["y"] == category][variable_bivarie].dropna() for category in df_1["y"].unique()]

            # Test ANOVA
            f_stat, p_value = f_oneway(*groups)
            st.write(f"**Statistique F** : {f_stat:.4f}")
            st.write(f"**Valeur p** : {p_value:.4f}")

            if p_value < 0.05:
                st.write(f"Il existe une différence significative entre les groupes pour `{variable_bivarie}` et `y`.")
            else:
                st.write(f"Aucune différence significative entre les groupes pour `{variable_bivarie}` et `y`.")

    elif variable_bivarie in df_1.select_dtypes(include=['object']).columns:
        # Analyse pour les variables catégorielles
        st.write(f"### Distribution de `{variable_bivarie}` par rapport à `y`")
        contingency_table = pd.crosstab(df_1[variable_bivarie], df_1['y'])
        st.dataframe(contingency_table)

        # Visualisation : Histogramme empilé
        fig = go.Figure(data=[
            go.Bar(name='Yes', x=contingency_table.index, y=contingency_table['yes']),
            go.Bar(name='No', x=contingency_table.index, y=contingency_table['no'])
        ])
        fig.update_layout(barmode='stack', title=f"Distribution de `{variable_bivarie}` par rapport à `y`")
        st.plotly_chart(fig)

        # Test du Chi-Carré
        st.write(f"### Test du Chi-Carré pour `{variable_bivarie}` et `y`")
        
        # Création de la table de contingence
        contingency_table = pd.crosstab(df_1[variable_bivarie], df_1['y'])
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        st.write(f"**Statistique du Chi-Carré** : {chi2_stat:.4f}")
        st.write(f"**Valeur p** : {p_value:.4f}")

        if p_value < 0.05:
            st.write(f"Il existe une relation significative entre `{variable_bivarie}` et `y`.")
        else:
            st.write(f"Aucune relation significative entre `{variable_bivarie}` et `y`.")





# Analyse de Corrélation
    st.subheader("Analyse de Corrélation")
    st.write("Exploration des relations entre variables numériques.")
    
    if st.checkbox("Afficher la matrice de corrélation 📊"):
        correlation_matrix = df_1.select_dtypes(include=['float64', 'int64']).corr()
        st.dataframe(correlation_matrix)

        # Visualisation du Heatmap des Corrélations
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='Viridis',
            colorbar=dict(title='Correlation'),
            text=correlation_matrix.values,
            hoverinfo='text'
        ))
        fig.update_layout(title="Heatmap de Corrélation")
        st.plotly_chart(fig)



     

    # Analyse Avancée - Nuage de Points Interactif
    st.write("### Analyse Avancée : Nuage de Points ")

    # Choix des variables numériques pour les axes
    variable_x = st.selectbox("Choisissez une variable pour l'axe X :", df_1.select_dtypes(include=['float64', 'int64']).columns)
    variable_y = st.selectbox("Choisissez une variable pour l'axe Y :", df_1.select_dtypes(include=['float64', 'int64']).columns)

    # Option pour ajouter une variable catégorielle pour la couleur
    variable_hue = st.selectbox("Choisissez une variable catégorielle pour la couleur (optionnel) :", 
                            [None] + list(df_1.select_dtypes(include=['object']).columns))

    # Création du nuage de points interactif avec Plotly
    fig = px.scatter(
        df_1, 
        x=variable_x, 
        y=variable_y, 
        color=variable_hue, 
        title=f"Nuage de Points  : {variable_x} vs {variable_y}",
        labels={variable_x: variable_x, variable_y: variable_y},
        template="plotly_white"
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig)











































if page == pages[4]:
    st.write("### Modélisation ")

    # Suppression des colonnes inutiles et sélection des colonnes d'intérêt
    df1 = df_1[['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']]

    # 1. Préparation des données
    target = 'y'  # Nom de la variable cible binaire
    X = df1.drop(columns=[target])  # Suppression de la colonne cible
    y = df1[target]

    # Séparation des variables quantitatives et qualitatives
    quantitative_vars = X.select_dtypes(include=['number']).columns
    qualitative_vars = X.select_dtypes(exclude=['number']).columns

    # 2. Choix des variables par l'utilisateur
    st.write("Sélectionner les variables à utiliser pour la prédiction:")
    
    # Sélectionner les variables quantitatives et qualitatives 
    selected_quant_vars = st.multiselect("Sélectionner les variables quantitatives", quantitative_vars, default=quantitative_vars)
    selected_qual_vars = st.multiselect("Sélectionner les variables qualitatives", qualitative_vars, default=qualitative_vars)

    # Mettre à jour X en fonction des variables sélectionnées
    X_selected = X[selected_quant_vars + selected_qual_vars]

    # 3. Traitement des variables qualitatives et quantitatives
    # Pipeline pour les variables quantitatives : imputation + standardisation
    quant_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Imputation des valeurs manquantes
        ('scaler', StandardScaler())  # Standardisation des données
    ])

    # Pipeline pour les variables qualitatives : imputation + encodage one-hot
    qual_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputation des valeurs manquantes
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Encodage One-Hot
    ])

    # Création du transformateur qui applique les pipelines aux bonnes colonnes
    preprocessor = ColumnTransformer(
        transformers=[
            ('quant', quant_pipeline, selected_quant_vars),
            ('qual', qual_pipeline, selected_qual_vars)
        ])

    # 4. Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

    # Application du prétraitement sur les données d'entraînement et de test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 5. Sélection du modèle via Streamlit
    model_choice = st.selectbox("Choisissez le modèle", ["Régression Logistique", "Random Forest"])

    # Entraînement et évaluation des modèles
    if model_choice == "Régression Logistique":
        # Modèle de Régression Logistique
        logreg_model = LogisticRegression(max_iter=1000)
        logreg_model.fit(X_train_processed, y_train)
        y_pred_logreg = logreg_model.predict(X_test_processed)

        # Évaluation du modèle de régression logistique
        logreg_accuracy = accuracy_score(y_test, y_pred_logreg)
        logreg_report = classification_report(y_test, y_pred_logreg, output_dict=True)
        logreg_conf_matrix = confusion_matrix(y_test, y_pred_logreg)

        # Affichage des résultats sous forme de tableau
        st.write("== Régression Logistique ==")
        st.write(f"Accuracy: {logreg_accuracy:.4f}")
        
        # Conversion du classification_report en DataFrame
        logreg_report_df = pd.DataFrame(logreg_report).T  # Utilisation de 'pd.DataFrame' pour créer le DataFrame
        st.dataframe(logreg_report_df)

        # Visualisation de la matrice de confusion avec Plotly (interactif)
        fig = go.Figure(data=go.Heatmap(
            z=logreg_conf_matrix,
            x=['Classe 0', 'Classe 1'],
            y=['Classe 0', 'Classe 1'],
            colorscale='Blues',
            colorbar=dict(title='Nombre de prédictions')
        ))
        fig.update_layout(
            title='Matrice de Confusion - Régression Logistique',
            xaxis_title='Prédictions',
            yaxis_title='Vrais Labels',
            autosize=True
        )
        st.plotly_chart(fig)

    elif model_choice == "Random Forest":
        # Modèle Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_processed, y_train)
        y_pred_rf = rf_model.predict(X_test_processed)

        # Évaluation du modèle Random Forest
        rf_accuracy = accuracy_score(y_test, y_pred_rf)
        rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
        rf_conf_matrix = confusion_matrix(y_test, y_pred_rf)

        # Affichage des résultats sous forme de tableau
        st.write("== Random Forest ==")
        st.write(f"Accuracy: {rf_accuracy:.4f}")
        
        # Conversion du classification_report en DataFrame
        rf_report_df = pd.DataFrame(rf_report).T  # Utilisation de 'pd.DataFrame' pour créer le DataFrame
        st.dataframe(rf_report_df)

        # Visualisation de la matrice de confusion avec Plotly (interactif)
        fig = go.Figure(data=go.Heatmap(
            z=rf_conf_matrix,
            x=['Classe 0', 'Classe 1'],
            y=['Classe 0', 'Classe 1'],
            colorscale='Blues',
            colorbar=dict(title='Nombre de prédictions')
        ))
        fig.update_layout(
            title='Matrice de Confusion - Random Forest',
            xaxis_title='Prédictions',
            yaxis_title='Vrais Labels',
            autosize=True
        )
        st.plotly_chart(fig)

    # 6. Prédiction pour un client spécifique
    st.write("### Prédiction pour un client spécifique")
    
    # Formulaire pour entrer les caractéristiques du client
    client_data = {}
    for col in selected_quant_vars:
        client_data[col] = st.number_input(f"Entrez la valeur pour {col}", value=0)

    for col in selected_qual_vars:
        client_data[col] = st.selectbox(f"Entrez la valeur pour {col}", options=["yes", "no"], index=0)

    # Prédiction en fonction des données saisies
    if st.button("Faire la prédiction"):
        # Appliquer le préprocesseur et faire la prédiction
        client_data_transformed = preprocessor.transform(pd.DataFrame([client_data]))
        if model_choice == "Régression Logistique":
            prediction = logreg_model.predict(client_data_transformed)
        elif model_choice == "Random Forest":
            prediction = rf_model.predict(client_data_transformed)
        
        if prediction[0] == 1:
            st.write("Le client va s'inscrire au dépôt à terme.")
        else:
            st.write("Le client ne va pas s'inscrire au dépôt à terme.")
