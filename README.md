# 🎬 Movie Recommender AI

## Sistema de Recomendación de Películas

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
de recomendación de películas con una interfaz inspirada en Netflix,
que permite descubrir contenido de forma visual e intuitiva.

La aplicación combina técnicas de Machine Learning desarrolladas
en un entorno de análisis, adaptadas para ejecución en tiempo real.

> *"Un buen sistema de recomendación no solo predice lo que el usuario quiere ver — descubre películas que el usuario no sabía que quería ver."*

---

## ✨ Características principales

* 👤 Selección de usuario y película base
* 🎬 Recomendaciones en tiempo real
* 🖼️ Pósters oficiales vía API de TMDB
* 🎨 Interfaz estilo plataforma de streaming
* 📱 Diseño responsivo
* ⚡ Renderizado rápido

---

## ⚙️ Arquitectura del sistema (implementada en notebook)

El sistema de recomendación fue desarrollado y evaluado en
un entorno de análisis (Jupyter Notebook), donde se implementaron:

* Filtrado colaborativo mediante **SVD**
* Filtrado por contenido con **TF-IDF**
* Combinación híbrida de ambos enfoques

La aplicación desplegada en Streamlit utiliza una versión
adaptada para ejecución interactiva.

---

## 🧠 Enfoque del modelo

El proyecto sigue un enfoque híbrido conceptual:

* **Colaborativo (SVD):** detecta patrones entre usuarios
* **Contenido (TF-IDF):** encuentra similitud entre películas
* **Combinación:** recomendaciones más robustas

⚠️ Nota: El entrenamiento completo del modelo se encuentra en el notebook del proyecto.

---

## 📁 Estructura del proyecto

```
├── app.py
├── README.md
├── requirements.txt
```

---

## 🔐 Seguridad

La API Key de TMDB se gestiona mediante:

* `st.secrets` en Streamlit Cloud
* Variables de entorno (`os.getenv`)

---

## 📊 Dataset

**MovieLens 100K**
GroupLens Research — University of Minnesota

🔗 https://grouplens.org/datasets/movielens/100k/

---

## 🔗 Proyecto relacionado

Este proyecto evoluciona hacia una versión más orientada a producto:

👉 movie_recommender_app_netflix_style

---

## 👨‍💻 Autor

**Jorge Ojeda**
Oracle Next Education (ONE) — Alura LATAM
📅 2026

---

## 📄 Licencia

Proyecto de uso educativo.
Datos proporcionados por MovieLens.
Pósters obtenidos vía TMDB API.
