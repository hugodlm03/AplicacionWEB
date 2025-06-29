import streamlit as st
import base64 # Necesario para codificar las imágenes
from PIL import Image

# --- FUNCIÓN PARA CARGAR IMÁGENES LOCALES ---
def get_image_as_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="TFG Hugo De Lamo - Aprendizaje Federado",
    page_icon="🏆",
    layout="wide"
)

# --- INYECTAMOS CSS PERSONALIZADO ---
st.markdown("""
<style>
/* --- NUEVO TÍTULO ANIMADO --- */
/* Definimos la animación del degradado */
@keyframes gradient-animation {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Aplicamos la animación a nuestro texto */
.animated-gradient-text {
    font-weight: bold;
    font-size: 2.8rem; /* Un poco más grande para que impacte */
    background: linear-gradient(to right, #00F260, #0575E6, #00F260);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient-animation 4s ease infinite;
}

/* --- ESTILO PARA LAS TARJETAS (Sin cambios) --- */
.card {
    background-color: #1a1a2e;
    border-radius: 15px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2), 0 6px 20px 0 rgba(0,0,0,0.19);
    transition: 0.3s;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    height: 100%;
}
.card:hover {
    box-shadow: 0 8px 16px 0 rgba(0,255,127,0.2);
    transform: scale(1.02);
}
.card-text {
    color: #e0e0e0;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)


# --- TÍTULO Y AUTORÍA ---
st.markdown('<h1 class="animated-gradient-text">Aplicación de Aprendizaje Federado en datos de Supermercados y Almacenes</h1>', unsafe_allow_html=True)
st.caption("Un Trabajo de Fin de Grado realizado por Hugo De Lamo Martínez.")
st.caption("Tutorizado por Pablo Torrijos Arenas y José Antonio Gámez Martín.")

st.divider()

# --- MOTIVACIÓN ---
st.header("El Desafío: IA Interpretable más allá de la Nube")
st.markdown("""
El mundo del Aprendizaje Automático está dominado por redes neuronales profundas, modelos potentísimos pero con dos grandes exigencias: **necesitan una cantidad masiva de datos y su lógica interna es una 'caja negra'**.
Esta situación representa una barrera para muchas empresas, como supermercados o almacenes.

> *Este TFG nace para abordar esa brecha: demostrar que se pueden crear modelos colaborativos, privados e interpretables utilizando los datos donde residen.*
""")

# --- LA SOLUCIÓN ---
st.header("💡 Nuestra Solución: Árboles de Decisión en un Entorno Federado")

st.success(
    "**Objetivo Principal:** Aplicar aprendizaje federado con algoritmos "
    "interpretables (árboles de decisión) en datos tabulares distribuidos, "
    "preservando la privacidad y garantizando la calidad explicativa de los modelos."
)

# --- TARJETAS PERSONALIZADAS ---
# Cargamos las imágenes
img_privacidad = get_image_as_base64("assets/icono_privacidad.png")
img_arbol = get_image_as_base64("assets/icono_arbol.png")
img_datos = get_image_as_base64("assets/icono_datos.png")

col1, col2, col3 = st.columns(3, gap="large")

# Tarjetas sin emojis en los títulos <h3>
with col1:
    st.markdown(f"""
    <div class="card">
        <img src="data:image/png;base64,{img_privacidad}" width="80">
        <h3>Privacidad por Diseño</h3>
        <p class="card-text">
        A diferencia de los métodos tradicionales, los datos sensibles nunca abandonan los dispositivos. 
        Solo se comparten los parámetros del modelo, no la información bruta, 
        mejorando la confidencialidad y la seguridad.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <img src="data:image/png;base64,{img_arbol}" width="80">
        <h3>Inteligencia Interpretable</h3>
        <p class="card-text">
        Dejamos atrás las 'cajas negras'. Este trabajo se centra en árboles de decisión, 
        modelos que formulan reglas legibles y que estan preparados especialmente para datos tabulares.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card">
        <img src="data:image/png;base64,{img_datos}" width="80">
        <h3>Validación con Datos Reales</h3>
        <p class="card-text">
        Todo el análisis se realiza sobre el 'Adidas US Sales Dataset', 
        un conjunto de datos real con 9,648 registros que simulamos en 28 silos 
        para recrear un escenario realista.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# --- OBJETIVOS ESPECÍFICOS ---
with st.expander("🎯 Ver los Objetivos Específicos del TFG"):
    st.markdown("""
    - **(OE1)** Estudiar a fondo la metodología del Aprendizaje Federado y su arquitectura.
    - **(OE2)** Investigar cómo adaptar algoritmos interpretables.
    - **(OE3)** Diseñar una arquitectura federada formal para datos tabulares.
    - **(OE4)** Implementar los algoritmos usando **Flower** y **Scikit-learn**.
    - **(OE5)** Realizar un Análisis Exploratorio de Datos (EDA) detallado.
    - **(OE6)** Ejecutar un estudio experimental riguroso para evaluar el rendimiento.
    - **(OE7)** Comparar los resultados federados con los enfoques centralizados.
    - **(OE8)** Presentar conclusiones claras sobre la viabilidad de este enfoque.
    """)

# --- GUÍA DE LA APLICACIÓN ---
st.info("""
### 🧭 Guía de Navegación
En el menú de la izquierda encontrarás el corazón de este TFG. ¡Te invito a explorar!

- **📊 Análisis Exploratorio de Datos:** Sumérgete en los datos de Adidas.
- **🌳 Estrategia Random Forest:** Analiza los resultados del primer enfoque federado.
- **🚀 Estrategia XGBoost + NN:** Explora la segunda estrategia híbrida.
- **🔮 Demostración de Predicción:** ¡Pon a prueba los modelos en tiempo real!
""", icon="ℹ️")

st.divider()
st.markdown("<h5 style='text-align: center;'>Hecho con 🧠 y ☕ por Hugo De Lamo Martínez</h5>", unsafe_allow_html=True)