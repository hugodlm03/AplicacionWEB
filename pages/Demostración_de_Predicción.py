import streamlit as st
import pandas as pd
import random
import re
import tempfile
import os
from pathlib import Path
import joblib
import xgboost as xgb

# CONFIGURACIÓN DE STREAMLIT 
st.set_page_config(
    page_title="🎯 Demo RF vs XGB (sin centralizados)",
    page_icon="🔮",
    layout="wide"
)

# RUTAS 
project_root     = Path(__file__).resolve().parents[1]
CSV_PATH         = project_root / "BaseDeDatos" / "adidas_centralizado.csv"
LIBSVM_PATH      = project_root / "BaseDeDatos" / "adidas.libsvm"
RFR_RESULTS_PATH = project_root / "Resultados" / "RandomForestFederado"
RFR_MODELS_PATH  = project_root / "Modelos" / "RFR"
XGB_MODELS_PATH  = project_root / "Modelos" / "Xgbllr"

#  CARGA DE MODELOS Random Forest (solo global y local) 
@st.cache_resource
def load_rfr_global():
    return joblib.load(RFR_MODELS_PATH / "Global" / "model.pkl")

@st.cache_resource
def load_rfr_local(client_id: str):
    df_silos = pd.read_csv(RFR_RESULTS_PATH / "ResultadoPorSiloMejorRun.csv")
    fila     = df_silos[df_silos["client"] == client_id].iloc[0]
    epoch    = int(fila["epoch"])
    fname    = f"{client_id}_epoch_{epoch}.pkl"
    return joblib.load(RFR_MODELS_PATH / "Locales" / fname)

#  CARGA DE Boosters XGBoost (solo local y global
@st.cache_resource
def load_xgb_local(client_id: str):
    m = re.search(r"\d+", client_id)
    if not m:
        raise ValueError(f"client_id inválido: {client_id}")
    num = m.group(0)
    path = XGB_MODELS_PATH / "Locales" / "trees" / f"tree_client_{num}_r0.json"
    bst  = xgb.Booster()
    bst.load_model(str(path))
    return bst

@st.cache_resource
def load_xgb_global(client_id: str):
    m = re.search(r"\d+", client_id)
    if not m:
        raise ValueError(f"client_id inválido: {client_id}")
    num = m.group(0)
    path = XGB_MODELS_PATH / "Global" / "trees" / f"tree_agg_client_{num}_r85.json"
    bst  = xgb.Booster()
    bst.load_model(str(path))
    return bst

#  ESTADO INICIAL 
if "rf_sample" not in st.session_state:
    st.session_state.rf_sample       = None
    st.session_state.rf_client_id    = ""
    st.session_state.xgb_libsvm_line = ""
    st.session_state.predictions     = None

#  INTERFAZ 
st.title(" Muestra Aleatoria y Predicción")
st.markdown(
    "- **RF** usa una fila del CSV (`adidas_centralizado.csv`)\n"
    "- **XGB** usa una línea del LIBSVM (`adidas.libsvm`)\n"
)

# Generar muestras
if st.button("1️⃣ Generar Muestras"):
    # RF: fila aleatoria + cliente aleatorio
    df_all       = pd.read_csv(CSV_PATH)
    df_silos     = pd.read_csv(RFR_RESULTS_PATH / "ResultadoPorSiloMejorRun.csv")
    cliente_azar = random.choice(df_silos["client"].tolist())
    muestra_rf   = df_all.sample(1)
    # XGB: línea aleatoria de .libsvm
    lines         = [l.strip() for l in open(LIBSVM_PATH) if l.strip()]
    linea_libsvm  = random.choice(lines)

    st.session_state.rf_client_id    = cliente_azar
    st.session_state.rf_sample       = muestra_rf
    st.session_state.xgb_libsvm_line = linea_libsvm
    st.session_state.predictions     = None

# Mostrar muestras
if st.session_state.rf_sample is not None:
    df_s   = st.session_state.rf_sample
    client = st.session_state.rf_client_id
    valor  = int(df_s["Units Sold"].iloc[0])

    st.subheader(f"🚀 RF — Silo: `{client}`")
    st.dataframe(df_s, use_container_width=True)
    st.info(f"Valor real de **Units Sold**: **{valor}**")

    st.subheader(" XGB — Instancia LIBSVM")
    st.code(st.session_state.xgb_libsvm_line, language="text")

    # Realizar predicciones
    if st.button("2️⃣ Realizar Predicciones"):
        with st.spinner("🔮 …calculando…"):
            try:
                # --- Random Forest ---
                rf_g = load_rfr_global()
                rf_l = load_rfr_local(client)
                feats = list(rf_g.feature_names_in_)
                X_rf  = df_s[feats]

                # --- XGBoost: escribir temp LIBSVM + DMatrix ---
                tmp = tempfile.NamedTemporaryFile(mode="w+", suffix=".libsvm", delete=False)
                tmp.write(st.session_state.xgb_libsvm_line + "\n")
                tmp.flush(); tmp.close()
                dmat = xgb.DMatrix(tmp.name + "?format=libsvm")
                os.unlink(tmp.name)

                xgb_l = load_xgb_local(client)
                xgb_g = load_xgb_global(client)

                # --- Recolectar predicciones sin centralizados ---
                preds = {
                    # Random Forest
                    "RF Ensemble Local":       rf_l.predict(X_rf)[0],
                    "RF 1 Árbol Local":        rf_l.estimators_[0].predict(X_rf)[0],
                    "RF Ensemble Global":      rf_g.predict(X_rf)[0],
                    "RF 1 Árbol Global":       rf_g.estimators_[0].predict(X_rf)[0],
                    # XGBoost
                    "XGB Ensemble Local":      xgb_l.predict(dmat)[0],
                    "XGB 1 Árbol Local":       xgb_l.predict(dmat, iteration_range=(0,1))[0],
                    "XGB Ensemble Global":     xgb_g.predict(dmat)[0],
                    "XGB 1 Árbol Global":      xgb_g.predict(dmat, iteration_range=(0,1))[0],
                }
                st.session_state.predictions = preds
            except Exception as e:
                st.error(f"❌ Error en predicción: {e}")
                st.session_state.predictions = None

# Mostrar resultados
if st.session_state.predictions:
    st.subheader("📊 Resultados")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔧 Random Forest")
        for label in [
            "RF Ensemble Local",
            "RF 1 Árbol Local",
            "RF Ensemble Global",
            "RF 1 Árbol Global",
        ]:
            st.metric(label, f"{preds[label]:.2f}")
    with col2:
        st.subheader("🌲 XGBoost")
        for label in [
            "XGB Ensemble Local",
            "XGB 1 Árbol Local",
            "XGB Ensemble Global",
            "XGB 1 Árbol Global",
        ]:
            st.metric(label, f"{preds[label]:.2f}")