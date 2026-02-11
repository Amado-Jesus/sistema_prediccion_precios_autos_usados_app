# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import sklearn
import xgboost
import joblib

# -------------------------
# Configuración general
# -------------------------
st.set_page_config(page_title="Car Price Predictor", layout="wide")

st.title("🚗 Predicción de Precio de Autos Usados")


# -------------------------
# Cargar modelo
# -------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    return model

model = load_model()

# -------------------------
# Subir archivo
# -------------------------
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Vista previa del dataset")
        st.dataframe(df.head())

        required_columns = [
            'year', 'manufacturer', 'condition', 'cylinders', 'fuel',
            'odometer', 'title_status', 'transmission',
            'drive', 'type', 'paint_color'
        ]

        # Validar columnas
        if not all(col in df.columns for col in required_columns):
            st.error("El archivo no contiene las columnas necesarias")
        else:
            st.success("✅ Columnas validadas correctamente.")

            # Predicciones
            predictions = model.predict(df)

            df["predicted_price"] = predictions

            st.subheader("Resultados con Predicción")
            st.dataframe(df.head(10))

         

            # Descargar resultado
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar archivo con predicciones",
                data=csv,
                file_name="predicciones_autos.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
