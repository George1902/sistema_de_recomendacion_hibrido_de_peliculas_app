# 🎬 Movie Recommender AI
## Sistema de Recomendación Híbrido de Películas

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-green)
![TMDB](https://img.shields.io/badge/API-TMDB-01b4e4)
![MovieLens](https://img.shields.io/badge/Datos-MovieLens-orange)
![Estado](https://img.shields.io/badge/Estado-Activo-brightgreen)

---

## 🚀 Demo en vivo

👉 **[Probar la app aquí](https://sistemaderecomendacionhibridodepeliculasapp-george-1902.streamlit.app/)**

---

## 📌 Descripción

**Movie Recommender AI** es una aplicación interactiva
de recomendación de películas con interfaz estilo Netflix,
que combina técnicas avanzadas de Machine Learning para
generar recomendaciones personalizadas en tiempo real.

> *"Un buen sistema de recomendación no solo predice
> lo que el usuario quiere ver — descubre películas
> que el usuario no sabía que quería ver."*

---

## ✨ Características principales

- Selección de usuario y película base
- Recomendaciones en tiempo real
- Pósters oficiales vía **API de TMDB**
- Interfaz visual tipo plataforma de streaming
- Grid de 5 columnas con tarjetas dinámicas
- Diseño responsivo y moderno

---

## ⚙️ Arquitectura del sistema
```
Datos MovieLens 100K
        ↓
Matriz Usuario-Ítem → SVD (50 factores latentes)
        ↓
TF-IDF (similitud entre películas)
        ↓
Sistema Híbrido (60% colaborativo + 40% contenido)
        ↓
Recomendaciones personalizadas
        ↓
Pósters vía API TMDB → Interfaz Streamlit
```

---

## 🤖 Modelo de Machine Learning

### Filtrado Colaborativo — SVD
- Técnica: **Truncated SVD**
- Reducción a **50 factores latentes**
- Matriz usuario-ítem normalizada
- Detecta patrones ocultos de preferencia

### Filtrado por Contenido — TF-IDF
- Extracción de géneros por película
- Vectorización con **TF-IDF**
- Similitud mediante **cosine similarity**
- Encuentra películas similares entre sí

### Sistema Híbrido
- **60%** filtrado colaborativo (SVD)
- **40%** filtrado por contenido (TF-IDF)
- Scores normalizados con MinMaxScaler

---

## 📊 Métricas de evaluación

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| RMSE | **0.7268** | Error promedio de ~0.73 estrellas |
| Precision@10 | **65.3%** | 6.5 de 10 recomendaciones relevantes |
| Coverage | **19.0%** | Películas únicas recomendadas |

---

## 🎬 Ejemplos de recomendaciones

Para un usuario con preferencia por Sci-Fi y Acción:

| # | Película | Score |
|---|---------|-------|
| 1 | Aliens (1986) | 0.977 |
| 2 | Empire Strikes Back (1980) | 0.955 |
| 3 | Terminator, The (1984) | 0.953 |
| 4 | Return of the Jedi (1983) | 0.947 |
| 5 | Terminator 2 (1991) | 0.917 |

---

## 🎨 Diseño UX/UI

- Interfaz oscura tipo **Netflix**
- Tarjetas de películas con efecto hover
- Overlay con degradado oscuro
- Título y score visibles sobre el póster
- Layout ancho y responsivo

---

## 🔐 Seguridad

La API Key de TMDB está protegida usando:
- `st.secrets` de Streamlit Cloud
- Variables de entorno (`os.getenv`)

No se exponen credenciales en el repositorio.

---

## 📁 Estructura del proyecto
```
│
├── app.py
├── README.md
└── requirements.txt
```
⚠️ Nota: Este proyecto muestra el desarrollo del modelo.
La implementación productiva se encuentra en:

👉 movie_recommender_app_netflix_style

---

## 🛠️ Tecnologías utilizadas

- **Python 3.12**
- **Streamlit** — interfaz web interactiva
- **Scikit-learn** — SVD, TF-IDF, cosine similarity
- **SciPy** — matrices sparse
- **Pandas / NumPy** — procesamiento de datos
- **TMDB API** — pósters de películas
- **Streamlit Cloud** — despliegue gratuito
- **GitHub** — control de versiones

---

## ▶️ Cómo ejecutar localmente

1. Clona el repositorio:
```bash
git clone https://github.com/George1902/sistema_de_recomendacion_hibrido_de_peliculas_app.git
cd sistema_de_recomendacion_hibrido_de_peliculas_app
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Configura tu API Key de TMDB en `.streamlit/secrets.toml`:
```toml
TMDB_API_KEY = "tu_api_key_aqui"
```

4. Descarga el dataset desde:
   [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)

5. Ejecuta la app:
```bash
streamlit run app.py
```

---

## 📋 requirements.txt
```
streamlit
pandas
numpy
scikit-learn
scipy
requests
```

---

## 🔮 Próximas mejoras

- 🔍 Buscador por nombre de película
- 🎭 Filtro por género (terror, comedia, etc.)
- 🧠 Recomendaciones sin usuario (content-first)
- ⭐ Sistema de favoritos
- 📊 Dashboard de métricas en tiempo real
- 🤖 Embeddings avanzados (Word2Vec / BERT)

---

## 📊 Dataset

**MovieLens 100K**
GroupLens Research — University of Minnesota
100,000 ratings — 943 usuarios — 1,682 películas
🔗 https://grouplens.org/datasets/movielens/100k/

---

## 🔗 Proyecto relacionado

Este proyecto es la evolución de un sistema previo:

👉 [Sistema de Recomendación Híbrido — Notebook completo](https://github.com/George1902/sistema-recomendacion-peliculas)

---

## 👨‍💻 Autor

**Jorge Ojeda**
Estudiante — Oracle Next Education (ONE) — Alura LATAM
Especialización: Ciencia de Datos
📅 2026

---

## ⚠️ Aviso importante

Esta aplicación utiliza el dataset MovieLens 100K
con fines educativos. Los pósters de películas se
obtienen vía API pública de TMDB.

---

## 📄 Licencia

Proyecto de uso educativo y libre distribución.
Los datos pertenecen a GroupLens Research y están
disponibles públicamente bajo su licencia de uso.
