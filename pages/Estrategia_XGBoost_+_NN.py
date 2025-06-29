import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import numpy as np
import xgboost as xgb
import torch
import json
import matplotlib.pyplot as plt

# CONFIGURACIÓN DE LA PÁGINA 
st.set_page_config(
    page_title="Estrategia XGBoost + NN",
    page_icon="🚀",
    layout="wide"
)

# FUNCIONES DE CARGA DE DATOS 
@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Error: No se pudo encontrar el archivo en la ruta: {path}. Por favor, verifica la estructura de carpetas.")
        return None

#  TÍTULO DE LA PÁGINA 
st.title("🚀 Estrategia 2: XGBoost + Red Neuronal (XGBllr)")
st.markdown("""
Esta sección se centra en la segunda estrategia experimental, que utiliza una arquitectura híbrida combinando 
modelos **XGBoost** en los clientes con una **Red Neuronal Convolucional (CNN)** en el servidor para la agregación de los árboles.
""")

#  Rutas a los archivos de resultados 
project_root = Path(__file__).parent.parent
xgb_results_path = project_root / "Resultados" / "XGBllr"
modelos_xgb_path = project_root / "Modelos" / "XGBllr"

# PESTAÑAS 
tab1, tab2, tab3 = st.tabs([
    "📊 Modelo Centralizado (Línea Base)",
    "🌐 Modelo Federado (XGBllr)",
    "🧠 Interpretación de Modelos"
])

# MODELO CENTRALIZADO 
with tab1:
    st.header("Análisis del Modelo XGBoost Centralizado")
    st.markdown("Como punto de referencia, primero evaluamos un modelo XGBoost estándar entrenado de forma centralizada con una búsqueda de hiperparámetros.")
    
    df_central_xgb = load_csv(xgb_results_path / "ResultadosCentralizado.csv")
    
    if df_central_xgb is not None:
        df_central_xgb['rmse'] = np.sqrt(df_central_xgb['result_test'])
        
        st.subheader("Búsqueda Interactiva de Hiperparámetros")
        top_n_central = st.slider("Número de mejores experimentos (centralizados) a mostrar:", min_value=5, max_value=len(df_central_xgb), value=20, step=5, key="slider_central")
        df_to_plot_central = df_central_xgb.sort_values('rmse').head(top_n_central)

        st.info("Interactúa con el gráfico: haz clic y arrastra en los ejes verticales para filtrar y ver qué rangos de parámetros producen los mejores resultados (líneas más oscuras).")
        
        fig_paralelas_xgb = px.parallel_coordinates(
            df_to_plot_central, 
            dimensions=['n_estimators_client', 'xgb_max_depth', 'subsample', 'learning_rate', 'alpha', 'gamma', 'min_child_weight', 'rmse'],
            color="rmse", color_continuous_scale=px.colors.sequential.Plasma_r,
            title=f"Top {top_n_central} Mejores Configuraciones del Modelo Centralizado",
            labels={col: col.replace('_', ' ').title() for col in df_to_plot_central.columns}
        )
        st.plotly_chart(fig_paralelas_xgb, use_container_width=True)
        
        best_central_xgb_rmse = df_central_xgb['rmse'].min()
        st.success(f"**Mejor RMSE del modelo XGBoost centralizado:** `{best_central_xgb_rmse:.2f}`")

# MODELO FEDERADO 
with tab2:
    st.header("Análisis del Modelo Federado (XGBllr)")
    
    st.subheader("Búsqueda Interactiva de Hiperparámetros (Federado)")
    df_fed_params = load_csv(xgb_results_path / "ResultadosFederadoPorROnda.csv")
    if df_fed_params is not None:
        df_fed_params['rmse'] = np.sqrt(df_fed_params['best_res'])

        top_n_fed = st.slider("Número de mejores experimentos (federados) a mostrar:", min_value=5, max_value=len(df_fed_params), value=10, step=5, key="slider_fed")
        df_to_plot_fed = df_fed_params.sort_values('rmse').head(top_n_fed)

        st.info("Usa el gráfico para explorar qué parámetros del modelo XGBoost y de la CNN de agregación consiguen un menor error en el entorno federado.")
        
        fig_paralelas_fed = px.parallel_coordinates(
            df_to_plot_fed,
            dimensions=['n_estimators_client', 'num_rounds', 'xgb_max_depth', 'cnn_lr', 'best_res_round_num', 'rmse'],
            color="rmse", color_continuous_scale=px.colors.sequential.Viridis,
            title=f"Top {top_n_fed} Mejores Configuraciones del Modelo Federado",
            labels={"n_estimators_client": "Nº Estimadores", "num_rounds": "Rondas", "xgb_max_depth": "Prof. Máx.", "cnn_lr": "LR (CNN)", "best_res_round_num": "Mejor Ronda", "rmse": "Mejor RMSE"}
        )
        st.plotly_chart(fig_paralelas_fed, use_container_width=True)

    st.divider()
    
    st.subheader("Análisis de la Convergencia del Mejor Modelo Federado")
    st.markdown("Una vez encontrada la mejor configuración, analizamos su rendimiento a lo largo de todas las rondas de entrenamiento.")
    
    df_todas_epocas = load_csv(xgb_results_path / "TodasEpocasMejorRun.csv")
    if df_todas_epocas is not None:
        df_todas_epocas['rmse'] = np.sqrt(df_todas_epocas['result_value'])
        fig_convergencia = px.line(df_todas_epocas, x='round_num', y='rmse', title="Convergencia del Error (RMSE) del Modelo Global por Ronda", labels={'round_num': 'Ronda Federada', 'rmse': 'RMSE Global'}, markers=True)
        best_round_df = df_todas_epocas.sort_values('rmse').iloc[0]
        best_round_num = int(best_round_df['round_num'])
        best_round_rmse = best_round_df['rmse']
        fig_convergencia.add_vline(x=best_round_num, line_dash="dot", line_color="red", annotation_text=f"Mejor resultado (Ronda {best_round_num})", annotation_position="top left")
        st.plotly_chart(fig_convergencia, use_container_width=True)
        st.success(f"El mejor modelo federado se obtuvo en la **ronda {best_round_num}** con un **RMSE de {best_round_rmse:.2f}**.")
        st.info("Se observa la clásica curva de aprendizaje federado: un error muy alto al principio que desciende drásticamente en las primeras rondas hasta estabilizarse.")
# INTERPRETACIÓN DEL MODELO (VERSIÓN PULIDA FINAL)
with tab3:
    st.header("Interpretación de los Modelos")
    st.markdown("Analizamos los componentes de los modelos para entender qué han aprendido.")

    #  ANÁLISIS DE LA CNN (SIN CAMBIOS) 
    st.subheader("Análisis de la Red de Agregación (CNN)")
    st.markdown("Visualizamos los pesos de los filtros de la CNN para intuir a qué patrones presta más atención el servidor.")
    try:
        cnn_path = modelos_xgb_path / "Global" / "cnns" / "cnn_global_round_85.pt"
        state_dict = torch.load(cnn_path, map_location=torch.device('cpu'))
        weights = state_dict['conv1d.weight'].numpy()
        
        fig_cnn_weights = px.imshow(weights[0], aspect="auto",
                                    title="Mapa de Calor de los Pesos del Primer Filtro de la CNN (Ronda 85)",
                                    labels=dict(x="Dimensión del Embedding", y="Canales de Entrada", color="Peso"))
        st.plotly_chart(fig_cnn_weights, use_container_width=True)

    except (FileNotFoundError, KeyError) as e:
        st.warning(f"No se pudo cargar el modelo de la CNN para visualizar los pesos. Detalle: {e}")
    except Exception as e:
        st.error(f"Ocurrió un error inesperado al cargar el modelo CNN: {e}")

    st.divider()

    #  VISUALIZACIÓN DE UN ÁRBOL FEDERADO REAL (MEJORADO) 
    st.subheader("Visualización de un Árbol Federado Real (Ronda 85)")
    st.markdown("Cargamos un árbol individual del modelo global agregado en la ronda 85 para inspeccionar su estructura de reglas.")
    
    nombre_del_arbol_json = "esemble_global_r85.json"
    
    try:
        tree_path = modelos_xgb_path / "Global" / "trees" / nombre_del_arbol_json
        
        if not tree_path.is_file():
            st.error(f"¡Archivo no encontrado! No existe ningún archivo en la ruta '{tree_path}'.")
            st.warning("Por favor, abre el código y modifica la variable 'nombre_del_arbol_json' con el nombre de un archivo que sí tengas.")
        else:
            # Mensaje de éxito eliminado para una interfaz más limpia
            bst = xgb.Booster()
            bst.load_model(str(tree_path))
            
            fig, ax = plt.subplots(figsize=(40, 15)) # Ajustamos el tamaño para un layout horizontal
            xgb.plot_tree(bst, num_trees=0, ax=ax, rankdir='LR')
            
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Ocurrió un error al cargar o visualizar el árbol de XGBoost: {e}")