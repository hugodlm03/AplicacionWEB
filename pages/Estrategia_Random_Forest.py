import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

#  CONFIGURACIÓN DE LA PÁGINA 
st.set_page_config(
    page_title="Estrategia Random Forest",
    page_icon="🌳",
    layout="wide"
)

#  FUNCIONES DE CARGA DE DATOS 
@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Error: No se pudo encontrar el archivo en la ruta: {path}. Por favor, verifica la estructura de carpetas.")
        return None

#  TÍTULO DE LA PÁGINA 
st.title(" Estrategia 1: Random Forest Federado (FRF)")
st.markdown("""
Esta sección detalla la evaluación de la estrategia basada en *Federated Random Forest*. 
Analizamos el rendimiento de un modelo centralizado como línea base, exploramos los resultados del entrenamiento 
federado a través de sus épocas y, finalmente, interpretamos el modelo agregado global para extraer conocimiento.
""")

#  Rutas a los archivos 
project_root = Path(__file__).parent.parent
rfr_results_path = project_root / "Resultados" / "RandomForestFederado"
modelos_rfr_path = project_root / "Modelos" / "RFR"

#  PESTAÑAS PARA ORGANIZAR EL CONTENIDO 
tab1, tab2, tab3 = st.tabs([
    "📊 Modelo Centralizado (Línea Base)", 
    "🌐 Modelo Federado (FRF)", 
    "🧠 Interpretación del Modelo Ganador"
])

# MODELO CENTRALIZADO 
with tab1:
    st.header("Análisis del Modelo Centralizado")
    st.markdown("Antes de evaluar el enfoque federado, entrenamos un modelo de Random Forest de forma tradicional (centralizada) usando un Grid Search para encontrar los mejores hiperparámetros. Este será nuestro punto de referencia.")
    
    df_centralizado = load_csv(rfr_results_path / "ResultadosCentralizados.csv")
    
    if df_centralizado is not None:
        st.subheader("Resultados del Grid Search Centralizado")
        st.dataframe(df_centralizado.sort_values('rank_test_score').head())
        
        best_central_score = df_centralizado.sort_values('mean_test_score', ascending=False).iloc[0]
        best_central_rmse = (-best_central_score['mean_test_score'])**0.5
        st.success(f"**Mejor RMSE del modelo centralizado:** `{best_central_rmse:.2f}`")

        st.subheader("Influencia de los Hiperparámetros (Replicando Figura 4.2 del TFG)")
        
        df_centralizado['rmse'] = (-df_centralizado['mean_test_score'])**0.5
        df_centralizado['param_max_depth'] = df_centralizado['param_max_depth'].fillna('None')
        pivot_table = df_centralizado.pivot_table(values='rmse', index='param_max_depth', columns='param_min_samples_leaf')
        
        # --- CAMBIO 1: Rellenamos valores nulos y forzamos formato para que se vean todos los números ---
        pivot_table = pivot_table.fillna(0)
        fig_heatmap = px.imshow(pivot_table, text_auto=True, aspect="auto",
                                title="RMSE medio según `max_depth` y `min_samples_leaf`",
                                labels=dict(x="Muestras Mínimas por Hoja", y="Profundidad Máxima", color="RMSE"),
                                color_continuous_scale='YlGnBu_r')
        st.plotly_chart(fig_heatmap, use_container_width=True)

#  MODELO FEDERADO 
with tab2:
    st.header("Análisis del Modelo Federado")
    st.markdown("Aquí exploramos el proceso de encontrar el mejor modelo federado y analizamos su comportamiento durante el entrenamiento.")

    st.subheader("Búsqueda Interactiva de Hiperparámetros")
    
    df_params_fed = load_csv(rfr_results_path / "ResultadosComparacionDeParametros.csv")

    if df_params_fed is not None:
        # Limpieza de datos
        df_params_fed.columns = df_params_fed.columns.str.strip()
        df_params_fed['bootstrap'] = df_params_fed['bootstrap'].astype(int)
        df_params_fed['max_depth'] = df_params_fed['max_depth'].fillna(0) # Rellenar NaN para visualización

        #  Slider para controlar el número de experimentos a mostrar 
        st.markdown("Usa el slider para ajustar cuántos de los mejores experimentos (ordenados por menor RMSE) quieres visualizar en el gráfico.")
        top_n = st.slider("Número de mejores experimentos a mostrar:", min_value=5, max_value=len(df_params_fed), value=10, step=5)
        
        df_to_plot = df_params_fed.sort_values('rmse').head(top_n)

        # Instrucciones para usar el gráfico 
        st.info("""
        **¿Cómo leer este gráfico?**
        - Cada **línea de color** es un experimento con una combinación de parámetros.
        - Las líneas más **oscuras/frías (azules)** tienen un error (RMSE) más bajo y son mejores.
        - **¡Interactúa!** Haz clic y arrastra el ratón sobre un eje vertical para filtrar por un rango de valores y ver qué patrones emergen. Por ejemplo, filtra en `min_samples_leaf` para ver solo los valores de `1`.
        """)

        #  Gráfico mejorado con etiquetas y datos filtrados 
        fig_paralelas = px.parallel_coordinates(df_to_plot,
                                                dimensions=['n_base_estimators', 'train_size', 'max_depth', 'min_samples_leaf', 'bootstrap', 'rmse'],
                                                color="rmse",
                                                color_continuous_scale=px.colors.sequential.Plasma_r, # Paleta de color invertida (frío=mejor)
                                                title=f"Top {top_n} Mejores Configuraciones del Modelo Federado",
                                                labels={
                                                    "n_base_estimators": "Nº Árboles Base",
                                                    "train_size": "Proporción Train",
                                                    "max_depth": "Profundidad Máx.",
                                                    "min_samples_leaf": "Hojas Mín.",
                                                    "bootstrap": "Usa Bootstrap",
                                                    "rmse": "Error (RMSE)"
                                                })
        st.plotly_chart(fig_paralelas, use_container_width=True)

        st.subheader("Tabla con las 5 Mejores Configuraciones Federadas")
        st.dataframe(df_params_fed.sort_values('rmse').head())

    st.divider()

    df_federado = load_csv(rfr_results_path / "ResultadoPorSiloMejorRun.csv")
    
    if df_federado is not None:

        # Gráfico de media por silo en lugar de la diferencia 
        st.subheader("Rendimiento Medio por Silo (entre épocas)")
        st.markdown("Este gráfico resume el rendimiento promedio de cada silo durante todo el proceso de entrenamiento federado. Nos da una idea de qué silos son, en media, más fáciles o difíciles de predecir.")

        avg_rmse_per_silo = df_federado.groupby('client')['rmse'].mean().reset_index()
        
        fig_avg_rmse = px.bar(avg_rmse_per_silo.sort_values('rmse', ascending=False), 
                              x='client', y='rmse',
                              title='RMSE Medio por Silo (promedio de todas las épocas)',
                              labels={'client': 'Silo (Cliente)', 'rmse': 'RMSE Medio'},
                              color='rmse',
                              color_continuous_scale='Blues_r')
        st.plotly_chart(fig_avg_rmse, use_container_width=True)

# Interpretación del modelo
with tab3:
    st.header("Interpretación del Modelo Federado Ganador")
    st.markdown("Una vez entrenado el modelo agregado, podemos 'abrirlo' para entender qué variables son más importantes para sus predicciones y cómo son las reglas de decisión que ha aprendido.")

    st.subheader("Ranking de Importancia de las Variables (Figura 4.3 del TFG)")
    feature_importance_data = {
        'Variable': ["Sales Method_Online", "Operating Margin", "Product_Men's Street Footwear", "Price per Unit", "Sales Method_Outlet", "inv_day", "City", "inv_month", "Retailer ID", "Product_Women's Apparel", "State", "inv_year", "Product_Men's Athletic Footwear", "Product_Women's Athletic Footwear", "Product_Women's Street Footwear"],
        'Importancia': [0.21, 0.21, 0.14, 0.12, 0.08, 0.05, 0.04, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01]
    }
    df_importance = pd.DataFrame(feature_importance_data)
    fig_importance = px.bar(df_importance.sort_values('Importancia', ascending=True), x='Importancia', y='Variable', orientation='h', title="Top-15 Variables más Importantes")
    st.plotly_chart(fig_importance, use_container_width=True)

    st.subheader("Visualización de un Árbol de Decisión (Figura 4.4 del TFG)")
    st.markdown("Podemos incluso visualizar uno de los árboles que componen el bosque final para entender sus reglas.")
    
    try:
        modelo_path = modelos_rfr_path / "Global" / "model.pkl"
        modelo_ganador = joblib.load(modelo_path)
        arbol_a_visualizar = modelo_ganador.estimators_[0]
        
        fig, axes = plt.subplots(figsize=(30,10), dpi=300)
        plot_tree(arbol_a_visualizar, feature_names=modelo_ganador.feature_names_in_, filled=True, rounded=True, max_depth=4, fontsize=4)
        st.pyplot(fig)
        st.info("Nota: El árbol se muestra con una profundidad limitada a 4 niveles para facilitar su lectura en la web.")
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo del modelo en '{modelo_path}'.")
    except Exception as e:
        st.error(f"Ocurrió un error al cargar o visualizar el modelo: {e}")