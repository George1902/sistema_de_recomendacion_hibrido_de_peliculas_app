import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Recomendador de Películas", page_icon="🎬")

st.title("🎬 Sistema de Recomendación de Películas")
st.write("Modelo híbrido: SVD + Contenido")

# -------------------------------
# CARGA DE DATOS (cache)
# -------------------------------
@st.cache_data
def cargar_datos():
    df_ratings = pd.read_csv("data/u.data", sep="\t",
                             names=['user_id','movie_id','rating','timestamp'])
    
    columnas_peliculas = [
        'movie_id', 'titulo', 'fecha_estreno', 'video_estreno',
        'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
        'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
        'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
        'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    
    df_peliculas = pd.read_csv(
        "data/u.item",
        sep="|",
        names=columnas_peliculas,
        encoding="latin-1"
    )

    generos = columnas_peliculas[5:]

    return df_ratings, df_peliculas, generos

df_ratings, df_peliculas, generos = cargar_datos()

# -------------------------------
# MODELO (cache)
# -------------------------------
@st.cache_data
def entrenar_modelo(df_ratings, df_peliculas, generos):

    # matriz
    user_item = df_ratings.pivot(
        index='user_id',
        columns='movie_id',
        values='rating'
    )

    user_means = user_item.mean(axis=1)
    user_item_centered = user_item.sub(user_means, axis=0).fillna(0)

    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=50, random_state=42)

    user_factors = svd.fit_transform(user_item_centered)
    item_factors = svd.components_

    predicciones = np.dot(user_factors, item_factors)
    predicciones = predicciones + user_means.values.reshape(-1,1)

    predicciones = pd.DataFrame(
        predicciones,
        index=user_item.index,
        columns=user_item.columns
    )

    # contenido
    def generar_generos(row):
        return ' '.join([g for g in generos if row[g] == 1])

    df_peliculas['generos_str'] = df_peliculas.apply(generar_generos, axis=1)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df_peliculas['generos_str'])
    similitud = cosine_similarity(tfidf_matrix)

    return predicciones, similitud

predicciones, similitud = entrenar_modelo(df_ratings, df_peliculas, generos)

# -------------------------------
# FUNCIONES
# -------------------------------
def get_candidates(movie_id):
    idx = df_peliculas[df_peliculas['movie_id'] == movie_id].index[0]
    scores = list(enumerate(similitud[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores[1:50]

def hybrid(user_id, movie_id):

    candidatos = get_candidates(movie_id)
    resultados = []

    for idx, score_cont in candidatos:

        movie_id_cand = df_peliculas.iloc[idx]['movie_id']

        if user_id in predicciones.index:
            score_colab = predicciones.loc[user_id, movie_id_cand]
        else:
            score_colab = 0

        score_colab = score_colab / 5

        final = 0.6 * score_colab + 0.4 * score_cont

        resultados.append((movie_id_cand, final))

    resultados = sorted(resultados, key=lambda x: x[1], reverse=True)

    return resultados[:10]

# -------------------------------
# UI
# -------------------------------
usuarios = df_ratings['user_id'].unique()
user_id = st.selectbox("Selecciona usuario", usuarios)

peliculas = df_peliculas[['movie_id','titulo']]
pelicula_nombre = st.selectbox("Selecciona película base", peliculas['titulo'])

movie_id = peliculas[peliculas['titulo'] == pelicula_nombre]['movie_id'].values[0]

# -------------------------------
# BOTÓN
# -------------------------------
if st.button("🎯 Recomendar"):

    recs = hybrid(user_id, movie_id)

    st.subheader("📌 Recomendaciones")

    for movie_id_rec, score in recs:
        titulo = df_peliculas[
            df_peliculas['movie_id'] == movie_id_rec
        ]['titulo'].values[0]

        st.write(f"🎬 {titulo} — ⭐ {round(score,3)}")