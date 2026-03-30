import streamlit as st
import pandas as pd
import numpy as np
import requests
import os

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="Movie Recommender AI",
    page_icon="🍿",
    layout="wide"
)

# -------------------------------
# 🎨 ESTILO NETFLIX
# -------------------------------
st.markdown("""
<style>

/* CONTENEDOR */
.movie-container {
    position: relative;
    overflow: hidden;
    border-radius: 12px;
    cursor: pointer;
}

/* IMAGEN */
.movie-img {
    width: 100%;
    border-radius: 12px;
    transition: transform 0.4s ease;
}

/* ZOOM */
.movie-container:hover .movie-img {
    transform: scale(1.1);
}

/* OVERLAY */
.movie-overlay {
    position: absolute;
    bottom: 0;
    width: 100%;
    padding: 10px;
    background: linear-gradient(to top, rgba(0,0,0,0.9), rgba(0,0,0,0));
    opacity: 0;
    transition: opacity 0.3s ease;
}

/* MOSTRAR OVERLAY */
.movie-container:hover .movie-overlay {
    opacity: 1;
}

/* TITULO */
.movie-title {
    color: white;
    font-size: 16px;
    font-weight: 700;
    line-height: 1.2;
}

/* SCORE */
.movie-score {
    color: #e50914;
    font-size: 14px;
    font-weight: 500;
}

/* PLACEHOLDER */
.no-image {
    height: 250px;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: #1c1c1c;
    color: white;
    font-size: 30px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🍿 Movie Recommender AI</h1>", unsafe_allow_html=True)
st.write("Sistema híbrido: SVD + Contenido")

# -------------------------------
# 🔐 API SEGURA
# -------------------------------
API_KEY = st.secrets.get("TMDB_API_KEY") or os.getenv("TMDB_API_KEY")

# -------------------------------
# 🎬 POSTERS
# -------------------------------
def get_poster(title):
    try:
        title = title.split('(')[0]

        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={title}"
        response = requests.get(url)

        if response.status_code != 200:
            return None

        data = response.json()

        if data.get("results"):
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        return None

    return None

# -------------------------------
# 📂 DATOS
# -------------------------------
@st.cache_data
def cargar_datos():
    df_ratings = pd.read_csv(
        "data/u.data",
        sep="\t",
        names=['user_id','movie_id','rating','timestamp']
    )

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
# 🤖 MODELO
# -------------------------------
@st.cache_data
def entrenar_modelo(df_ratings, df_peliculas, generos):

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

    # CONTENIDO
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
# 🔥 SISTEMA HÍBRIDO
# -------------------------------
def hybrid(user_id, movie_id, top_n=15):

    user_pred = predicciones.loc[user_id]

    idx = df_peliculas[df_peliculas['movie_id'] == movie_id].index[0]
    sim_scores = list(enumerate(similitud[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:50]

    movie_indices = [i[0] for i in sim_scores]
    movie_ids_similares = df_peliculas.iloc[movie_indices]['movie_id']

    scores = []
    for m_id in movie_ids_similares:
        score = user_pred.get(m_id, 0)
        scores.append((m_id, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    return scores[:top_n]

# -------------------------------
# 🎛️ UI
# -------------------------------
usuarios = df_ratings['user_id'].unique()
user_id = st.selectbox("👤 Usuario", usuarios)

peliculas = df_peliculas[['movie_id','titulo']]
pelicula_nombre = st.selectbox("🎬 Película base", peliculas['titulo'])

movie_id = peliculas[peliculas['titulo'] == pelicula_nombre]['movie_id'].values[0]

# -------------------------------
# 🚀 RECOMENDADOR
# -------------------------------
if st.button("🚀 Recomendar"):

    recs = hybrid(user_id, movie_id)

    st.subheader("🔥 Recomendaciones")

    cols = st.columns(5)

    for i, (movie_id_rec, score) in enumerate(recs):

        titulo = df_peliculas[
            df_peliculas['movie_id'] == movie_id_rec
        ]['titulo'].values[0]

        poster = get_poster(titulo)

        col = cols[i % 5]

        with col:

            if poster:
                st.markdown(f"""
                <div class="movie-container">
                    <img src="{poster}" class="movie-img"/>
                    <div class="movie-overlay">
                        <div class="movie-title">{titulo[:30]}</div>
                        <div class="movie-score">⭐ {round(score,2)}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="movie-container">
                    <div class="no-image">🎬</div>
                    <div class="movie-overlay">
                        <div class="movie-title">{titulo[:30]}</div>
                        <div class="movie-score">⭐ {round(score,2)}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
