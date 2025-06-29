import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

#  CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="Análisis Exploratorio",
    page_icon="📊",
    layout="wide"
)

#  INYECTAMOS CSS PERSONALIZADO (AÑADIMOS ESTILO PARA MÉTRICAS) 
st.markdown("""
<style>
/* Estilo para las métricas centradas */
.centered-metric {
    background-color: rgba(38, 39, 48, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
}
.centered-metric p {
    margin-bottom: 0.5rem;
    color: #a0a0a0;
    font-size: 0.9rem;
}
.centered-metric h2 {
    margin-top: 0;
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)


# FUNCIÓN PARA CARGAR DATOS 
@st.cache_data
def load_clean_data(path):
    df = pd.read_csv(path)
    if 'Invoice Date' in df.columns:
        df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])
    return df

# TÍTULO Y CARGA DE DATOS 
st.title("Análisis Exploratorio de los Datos")
st.markdown("En esta sección, exploramos el conjunto de datos de ventas de Adidas ya preprocesado para identificar patrones, distribuciones y relaciones clave que guiarán el modelado.")

try:
    project_root = Path(__file__).parent.parent
    file_path = project_root / "BaseDeDatos" / "adidas_sales_limpio.csv"
    df_adidas = load_clean_data(file_path)

    #  VISIÓN GENERAL 
    st.header("Visión General del Conjunto de Datos")
    st.dataframe(df_adidas.head())
    
    #   MÉTRICAS CENTRALIZADAS 
    col1, col2, col3, col4 = st.columns(4) # Usamos 4 columnas para dejar espacio y centrar
    
    with col2:
        st.markdown(f"""
        <div class="centered-metric">
            <p>Número de Registros</p>
            <h2>{df_adidas.shape[0]:,}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="centered-metric">
            <p>Número de Columnas</p>
            <h2>{df_adidas.shape[1]}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Estadísticas Descriptivas de las Variables Numéricas")
    st.write(df_adidas.describe())

    st.divider()

    #  ANÁLISIS UNIVARIADO INTERACTIVO 
    st.header("Análisis Univariado: Explorando Cada Variable")
    st.markdown("Usa el menú desplegable dentro de cada pestaña para seleccionar una variable y visualizar su distribución.")

    variables_categoricas = ['Retailer', 'Region', 'State', 'City', 'Product', 'Sales Method']
    variables_numericas = ['Price per Unit', 'Units Sold', 'Total Sales', 'Operating Profit', 'Operating Margin']

    tab1, tab2 = st.tabs(["Variables Numéricas", "Variables Categóricas"])

    with tab1:
        selected_numerical = st.selectbox("Selecciona una variable numérica:", variables_numericas)
        
        if selected_numerical:
            fig_num = px.histogram(df_adidas, x=selected_numerical, nbins=50,
                                   title=f"Distribución de {selected_numerical}",
                                   color_discrete_sequence=['#00F260'])
            fig_num.update_layout(bargap=0.1)
            st.plotly_chart(fig_num, use_container_width=True)

    with tab2:
            st.subheader("Frecuencia de las Categorías Principales")
            
            # Selector para la variable categórica (sin cambios)
            selected_categorical = st.selectbox("Selecciona una variable categórica:", variables_categoricas)
            
            if selected_categorical:
                # Contamos el número de categorías únicas para la variable seleccionada
                unique_count = df_adidas[selected_categorical].nunique()
                
                # --- LÓGICA CONDICIONAL PARA EL TÍTULO Y LOS DATOS ---
                if unique_count > 20:
                    # Si hay muchas categorías, mostramos solo el top 20
                    chart_title = f"Frecuencia de las 20 principales categorías en '{selected_categorical}'"
                    counts = df_adidas[selected_categorical].value_counts().nlargest(20).reset_index()
                    st.info(f"Nota: La variable '{selected_categorical}' tiene {unique_count} categorías únicas. Se muestran solo las 20 más frecuentes para una mejor visualización.")
                else:
                    # Si hay pocas categorías, las mostramos todas
                    chart_title = f"Frecuencia por categoría en '{selected_categorical}'"
                    counts = df_adidas[selected_categorical].value_counts().reset_index()

                # Renombramos las columnas para el gráfico
                counts.columns = [selected_categorical, 'count']
                
                # Creamos la figura con el título y los datos correctos
                fig_cat = px.bar(counts, 
                                x=selected_categorical, 
                                y='count',
                                title=chart_title,
                                labels={'count':'Número de Registros'},
                                color=selected_categorical, 
                                color_discrete_sequence=px.colors.qualitative.Vivid) # He cambiado la paleta de color a una más viva
                
                st.plotly_chart(fig_cat, use_container_width=True)


    st.divider()

    # ANÁLISIS MULTIVARIADO
    st.header("Análisis Multivariado: Encontrando Relaciones")
    
    # código de correlación y scatter plot se mantiene igual
    st.subheader("Matriz de Correlación entre Variables Numéricas")
    numeric_df = df_adidas.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_df.corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Heatmap de Correlación", color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Relación Interactiva entre Variables")
    numeric_cols_list = numeric_df.columns.tolist()
    categorical_cols = [col for col in df_adidas.columns if df_adidas[col].dtype == 'object']
    x_axis = st.selectbox("Elige la variable para el eje X:", numeric_cols_list, index=numeric_cols_list.index('Price per Unit'))
    y_axis = st.selectbox("Elige la variable para el eje Y:", numeric_cols_list, index=numeric_cols_list.index('Units Sold'))
    try:
        cat_index = categorical_cols.index('Product')
    except (ValueError, IndexError):
        cat_index = 0
    color_by = st.selectbox("Elige la variable para colorear los puntos:", categorical_cols, index=cat_index)
    fig_scatter = px.scatter(df_adidas, x=x_axis, y=y_axis, color=color_by, title=f"Relación entre {x_axis} y {y_axis}", hover_data=['Retailer', 'State'])
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    #  ANÁLISIS DE PARTICIONES FEDERADAS 
    st.header("Análisis de las Particiones Federadas (Silos)")
    st.markdown("Para simular un entorno de Aprendizaje Federado, el conjunto de datos se dividió en **28 silos** o nodos. Cada silo representa un `Retailer` en una `Region` específica. Esta sección nos permite visualizar la distribución y heterogeneidad de estos silos.")

    # Creamos la columna 'Silo' para facilitar la visualización
    df_adidas['Silo'] = df_adidas['Retailer'] + ' - ' + df_adidas['Region']
    silo_counts = df_adidas['Silo'].value_counts().reset_index()
    silo_counts.columns = ['Silo', 'Número de Registros']

    # Gráfico de Treemap para ver la composición de los silos
    st.subheader("Composición Jerárquica de los Silos")
    fig_treemap = px.treemap(df_adidas, path=[px.Constant("Todos los Silos"), 'Region', 'Retailer'],
                             title="Treemap de Regiones y Minoristas",
                             color_continuous_scale='RdBu',
                             color='Operating Margin')
    fig_treemap.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    st.plotly_chart(fig_treemap, use_container_width=True)

    # Gráfico de barras para ver el tamaño de cada silo (replicando Tabla 4.1 de tu TFG)
    st.subheader("Tamaño de Cada Silo (Nº de Registros)")
    fig_silo_size = px.bar(silo_counts.sort_values('Número de Registros', ascending=False),
                           x='Silo', y='Número de Registros',
                           title='Número de Registros por Silo Federado',
                           color='Número de Registros',
                           labels={'Número de Registros': 'Cantidad de Datos'})
    st.plotly_chart(fig_silo_size, use_container_width=True)
    st.info("Como se observa, existe una marcada heterogeneidad en el tamaño de los silos. Algunos nodos tienen muchos datos (>800 registros), mientras que otros apenas cuentan con unas pocas decenas. Este es un reto clave en el Aprendizaje Federado.")

    #  NUEVA SECCIÓN: EXPLORADOR INTERACTIVO DE SILOS 
    st.subheader("Explorador Interactivo de Silos Individuales")
    st.markdown("Selecciona un silo del menú desplegable para analizar su distribución interna de productos y métodos de venta. Esto demuestra la **heterogeneidad non-IID** de los datos.")

    # Selector para elegir un silo
    selected_silo = st.selectbox("Selecciona un Silo para inspeccionar:", sorted(df_adidas['Silo'].unique()))

    if selected_silo:
        silo_df = df_adidas[df_adidas['Silo'] == selected_silo].copy()
        
        st.metric("Registros en este Silo", f"{silo_df.shape[0]}")
        
        col1, col2 = st.columns(2)
        with col1:
            fig_pie_product = px.pie(silo_df, names='Product', title=f"Distribución de Productos en {selected_silo}", hole=0.4)
            st.plotly_chart(fig_pie_product, use_container_width=True)
        with col2:
            fig_pie_method = px.pie(silo_df, names='Sales Method', title=f"Métodos de Venta en {selected_silo}", hole=0.4)
            st.plotly_chart(fig_pie_method, use_container_width=True)

    st.divider()

    #  NUEVA SECCIÓN: ANÁLISIS TEMPORAL 
    st.header("Análisis Temporal de las Ventas")
    st.markdown("Visualizamos la evolución de las ventas totales a lo largo del tiempo para detectar tendencias o estacionalidad.")

    # Agrupamos los datos por mes
    ventas_mensuales = df_adidas.set_index('Invoice Date').resample('M')['Total Sales'].sum().reset_index()
    
    fig_temporal = px.line(ventas_mensuales, x='Invoice Date', y='Total Sales',
                           title="Evolución Mensual de las Ventas Totales (Figura 3.15 del TFG)",
                           labels={'Invoice Date': 'Mes', 'Total Sales': 'Ventas Totales ($)'},
                           markers=True)
    fig_temporal.update_layout(xaxis_title="Fecha", yaxis_title="Ventas Totales ($)")
    st.plotly_chart(fig_temporal, use_container_width=True)


except FileNotFoundError:
    st.error(f"Error: No se encontró el archivo 'adidas_sales_limpio.csv' en la ruta '{file_path}'. Por favor, revisa la estructura de carpetas y el nombre del archivo.")
except Exception as e:
    st.error(f"Ha ocurrido un error inesperado al procesar el archivo: {e}")