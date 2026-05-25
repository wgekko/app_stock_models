import base64
from pathlib import Path
import streamlit.components.v1 as components
import streamlit as st
import yfinance as yf
import polars as pl
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
import datetime
import xgboost as xgb
import os
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")



# --- Configuración página ---
st.set_page_config(page_title="Quant Platform - Calculus & AI", layout="wide", page_icon=":material/terminal:")


# --- Estilos Globales (Ocultar elementos de Streamlit) ---
# st.markdown("""
#     <style>
#     #MainMenu {visibility: hidden;}
#     header {visibility: hidden;}
#     footer {visibility: hidden;}
#     .stDeployButton {visibility: hidden;}
#     [data-testid="stSidebar"] { display: none; }
#     [data-testid="stAppViewContainer"] { margin-left: 0px; }
#     </style>
# """, unsafe_allow_html=True)


col_izq, col_central, col_der = st.columns([1, 10, 1])

with col_central:    
    st.subheader(":material/terminal: Terminal Quant de Alta Performance", anchor=False, text_alignment="center")

    # 2. CSS para centrar el texto dentro de los componentes st.info (o alertas)
    st.markdown("""
        <style>
        .stAlert > div {
            text-align: center;
            display: flex;
            justify-content: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # (Aquí continuarían tus animaciones y botones...)
    v1, v2 = st.columns(2)

# --- Carga de Animaciones ---
def load_html(file_name):
    return Path(file_name).read_text(encoding="utf-8")

try:
    opcion1_html = load_html("static/index1.html") #
    opcion2_html = load_html("static/index2.html") #
except:
    opcion1_html = opcion2_html = ""

# --- Función para cargar HTML como Data URL para st.iframe ---
def get_html_data_url(file_path):
    try:
        content = Path(file_path).read_text(encoding="utf-8")
        b64 = base64.b64encode(content.encode()).decode()
        return f"data:text/html;base64,{b64}"
    except:
        return ""

# Obtener las URLs de datos
opcion1_html = get_html_data_url("static/index1.html")
opcion2_html = get_html_data_url("static/index2.html")

# --- RENDERIZADO ---
col_izq, col_central, col_der = st.columns([1, 10, 1]) #
with col_central:
    # 1. Animaciones en ventanas paralelas
    v1, v2 = st.columns(2)
    with v1:        
        #components.html(tunnel_html, height=400, scrolling=False)
        if opcion1_html:
            st.iframe(opcion1_html, height=310)
    with v2:
        #components.html(crt_html, height=400, scrolling=False)
        if opcion2_html:
            st.iframe(opcion2_html, height=310)
    st.write("") 

    # (Aquí continuarían tus animaciones 2da etapa...)
    v3, v4 = st.columns(2)

# --- Carga de Animaciones ---
def load_html(file_name):
    return Path(file_name).read_text(encoding="utf-8")
try:
    opcion3_html = load_html("static/texto-1.html") #
    opcion4_html = load_html("static/texto-2.html") #
except:
    opcion3_html = opcion4_html = ""
# --- Función para cargar HTML como Data URL para st.iframe ---
def get_html_data_url(file_path):
    try:
        content = Path(file_path).read_text(encoding="utf-8")
        b64 = base64.b64encode(content.encode()).decode()
        return f"data:text/html;base64,{b64}"
    except:
        return ""
# Obtener las URLs de datos
opcion3_html = get_html_data_url("static/texto-1.html")
opcion4_html = get_html_data_url("static/texto-2.html")

# --- RENDERIZADO ---
col_izq, col_central, col_der = st.columns([1, 10, 1]) #

with col_central:
    # 1. Animaciones en ventanas paralelas
    v3, v4 = st.columns(2)
    with v3:        
        #components.html(tunnel_html, height=400, scrolling=False)
        if opcion3_html:
            st.iframe(opcion3_html, height=325)
    with v4:
        #components.html(crt_html, height=400, scrolling=False)
        if opcion4_html:
            st.iframe(opcion4_html, height=325)

    st.write("") 

    with st.expander(":material/terminal: Expandir Modelo de Análisis Quant de Alta Performance"):
        @st.cache_data
        def obtener_datos(ticker, dias_historia, today_date):
            # today_date serves as a cache key invalidation when the day changes!
            start_date = today_date - datetime.timedelta(days=dias_historia)
            
            df_pd = yf.download(ticker, start=start_date, end=today_date, progress=False)
            
            if df_pd.empty:
                return None
            if isinstance(df_pd.columns, pd.MultiIndex):
                df_pd.columns = df_pd.columns.get_level_values(0)
            return pl.from_pandas(df_pd.reset_index())

        # ==========================================
        # 2. Motor de Cálculo Diferencial y Estocástico (Polars / Rust)
        # ==========================================
        def aplicar_calculo_financiero(df, ventana_derivadas=10):
            # Step 1: Base columns that only depend on Close, High, Low
            df = df.with_columns([
                (pl.col("Close") / pl.col("Close").shift(1)).log().alias("Retorno_Diario"),
                (pl.col("Close") - pl.col("Close").shift(1)).alias("Derivada_1_Pura"),
                pl.col("Close").rolling_mean(window_size=20).alias("SMA_20"),
                pl.col("Close").rolling_std(window_size=20).alias("STD_20"),
                pl.col("Low").rolling_min(window_size=14).alias("Lowest_Low_14"),
                pl.col("High").rolling_max(window_size=14).alias("Highest_High_14"),
            ])
            
            # Step 2: Columns depending on Step 1 outputs
            df = df.with_columns([
                (pl.col("Retorno_Diario").rolling_std(window_size=20) * np.sqrt(252)).alias("Volatilidad_20d"),
                pl.col("Derivada_1_Pura").rolling_mean(ventana_derivadas).alias("Velocidad"),
                (pl.col("SMA_20") + (pl.col("STD_20") * 2)).alias("Bollinger_Upper"),
                (pl.col("SMA_20") - (pl.col("STD_20") * 2)).alias("Bollinger_Lower"),
                pl.when(pl.col("Derivada_1_Pura") > 0).then(pl.col("Derivada_1_Pura")).otherwise(0).alias("Gain"),
                pl.when(pl.col("Derivada_1_Pura") < 0).then(pl.col("Derivada_1_Pura").abs()).otherwise(0).alias("Loss"),
                (100 * (pl.col("Close") - pl.col("Lowest_Low_14")) / 
                (pl.col("Highest_High_14") - pl.col("Lowest_Low_14") + 1e-10)).alias("Stoch_K"),
            ])
            
            # Step 3: Columns depending on Step 2 outputs
            df = df.with_columns([
                (pl.col("Velocidad") - pl.col("Velocidad").shift(1)).alias("Derivada_2_Pura"),
                pl.col("Gain").rolling_mean(14).alias("Avg_Gain"),
                pl.col("Loss").rolling_mean(14).alias("Avg_Loss"),
                pl.col("Stoch_K").rolling_mean(window_size=9).alias("Stoch_D"),
            ])
            
            # Step 4: Final derivations
            df = df.with_columns([
                pl.col("Derivada_2_Pura").rolling_mean(ventana_derivadas).alias("Aceleracion"),
                (100 - (100 / (1 + (pl.col("Avg_Gain") / (pl.col("Avg_Loss") + 1e-10))))).alias("RSI"),
            ])
            
            return df.drop_nulls()

        # --- Función Auxiliar para las Griegas (Black-Scholes) ---
        def calcular_griegas_bs(S, K, T, r, sigma):
            """Calcula Delta y Gamma analíticos usando NumPy/Scipy."""
            if sigma <= 0 or T <= 0:
                return 0.5, 0.0
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            return delta, gamma


        def calcular_griegas_bs_vec(S, K, T, r, sigma):
            """Calcula Delta y Gamma analíticos utilizando operaciones vectorizadas de NumPy."""
            S = np.asarray(S, dtype=np.float64)
            K = np.asarray(K, dtype=np.float64)
            sigma = np.asarray(sigma, dtype=np.float64)
            
            # Crear máscara para evitar divisiones por cero o valores no válidos
            mask = (sigma > 0) & (T > 0)
            
            delta = np.full_like(S, 0.5)
            gamma = np.zeros_like(S)
            
            if np.any(mask):
                S_v = S[mask]
                K_v = K[mask]
                sigma_v = sigma[mask]
                
                # T y r pueden ser arrays o escalares
                T_v = T[mask] if isinstance(T, np.ndarray) else T
                r_v = r[mask] if isinstance(r, np.ndarray) else r
                
                sqrt_T = np.sqrt(T_v)
                d1 = (np.log(S_v / K_v) + (r_v + 0.5 * sigma_v ** 2) * T_v) / (sigma_v * sqrt_T)
                
                delta[mask] = norm.cdf(d1)
                gamma[mask] = norm.pdf(d1) / (S_v * sigma_v * sqrt_T)
                
            return delta, gamma

        # ==========================================
        # 3. Motores de Inferencia (LSTM, XGBoost, GARCH)
        # ==========================================
        def _hash_ndarray(arr: np.ndarray) -> int:
            """Hashing de array de numpy por valor en lugar de ID de memoria para que st.cache_resource funcione."""
            return hash(arr.tobytes())

        @st.cache_resource(hash_funcs={np.ndarray: _hash_ndarray})
        def entrenar_modelo_lstm(datos_cierre, ventana, epocas):
            scaler = MinMaxScaler(feature_range=(0, 1))
            datos_escalados = scaler.fit_transform(datos_cierre.reshape(-1, 1))
            X, y = [], []
            for i in range(ventana, len(datos_escalados)):
                X.append(datos_escalados[i-ventana:i, 0])
                y.append(datos_escalados[i, 0])
            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            modelo = Sequential()
            modelo.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
            modelo.add(Dropout(0.2))
            modelo.add(LSTM(units=50, return_sequences=False))
            modelo.add(Dropout(0.2))
            modelo.add(Dense(units=1))
            modelo.compile(optimizer='adam', loss='mean_squared_error')
            modelo.fit(X, y, epochs=epocas, batch_size=32, verbose=0)
            return modelo, scaler


        def entrenar_xgboost_1(df_features):
            """Entrena XGBoost para predecir si el precio subirá (1) o bajará (0) mañana."""
            # 1. Asegurar limpieza de nulos antes de cualquier cosa
            df_ml = df_features.copy().dropna()
            
            # 2. Definir target
            df_ml['Target'] = np.where(df_ml['Retorno_Diario'].shift(-1) > 0, 1, 0)
            df_ml = df_ml.dropna() 
            
            features = ['RSI', 'Velocidad', 'Aceleracion', 'Volatilidad_20d', 'Delta', 'Gamma']
            
            # 3. CONVERSIÓN EXPLÍCITA A TIPO FLOAT64
            X = df_ml[features].astype(np.float64) 
            y = df_ml['Target'].astype(int)
            
            modelo = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
            modelo.fit(X, y)
            
            # Extraer importancia
            importancia = pd.DataFrame({'Indicador': features, 'Importancia': modelo.feature_importances_})
            importancia = importancia.sort_values(by='Importancia', ascending=True)
            
            return modelo, importancia, features


        def predecir_riesgo_garch(retornos):
            """Pronostica la volatilidad de mañana usando GARCH(1,1)."""
            # Escalar retornos x100 ayuda a la convergencia matemática del optimizador
            retornos_escalados = retornos.dropna() * 100 
            modelo = arch_model(retornos_escalados, vol='Garch', p=1, q=1, mean='Zero')
            resultado = modelo.fit(disp='off')
            pronostico = resultado.forecast(horizon=1)
            
            # Extraer varianza proyectada, volver a escala normal y anualizar
            var_proyectada = pronostico.variance.iloc[-1, 0]
            vol_diaria = np.sqrt(var_proyectada) / 100
            vol_anualizada = vol_diaria * np.sqrt(252)
            return vol_anualizada

        # ==========================================
        # 4. Interfaz Gráfica (Streamlit)
        # ==========================================
        st.subheader(":material/terminal: Terminal Quant de Alta Performance")

        st.sidebar.subheader(":material/data_thresholding: Parámetros de Mercado")
        ticker = st.sidebar.text_input("Símbolo de Activo", value="AAPL").upper()

        # --- Selector de Periodos de 6 meses a 5 años ---
        opciones_periodo = {
            "6 Meses": 182,
            "1 Año": 365,
            "1.5 Años": 547,
            "2 Años": 730,
            "2.5 Años": 912,
            "3 Años": 1095,
            "3.5 Años": 1277,
            "4 Años": 1460,
            "4.5 Años": 1642,
            "5 Años": 1825
        }
        etiqueta_periodo = st.sidebar.selectbox("Historial de Análisis", list(opciones_periodo.keys()), index=1) 
        dias_periodo = opciones_periodo[etiqueta_periodo]
        # --------------------------------------------------------

        ui_deriv_window = st.sidebar.slider("Suavizado de Derivadas (Días)", 3, 20, 10)

        st.sidebar.header(":material/settings: Configuración de IA Temporal")
        ui_ventana = st.sidebar.slider("Ventana de Memoria (Lookback)", 10, 120, 60, step=10)
        ui_epocas = st.sidebar.slider("Épocas de Optimización (Techo 50)", 1, 50, 10)

        # Inicializar y comprobar estado para conservar simulaciones
        if "prev_params" not in st.session_state:
            st.session_state.prev_params = {}
        if "lstm_results" not in st.session_state:
            st.session_state.lstm_results = None
        if "comite_results" not in st.session_state:
            st.session_state.comite_results = None

        # Recopilar parámetros actuales para detectar cambios
        current_params = {
            "ticker": ticker,
            "dias_periodo": dias_periodo,
            "ui_deriv_window": ui_deriv_window,
            "ui_ventana": ui_ventana,
            "ui_epocas": ui_epocas
        }

        # Si cambian los parámetros principales, invalidamos las predicciones guardadas
        if st.session_state.prev_params != current_params:
            st.session_state.prev_params = current_params
            st.session_state.lstm_results = None
            st.session_state.comite_results = None


        if ticker:
            # today_date passed as a cache key invalidation when the day changes!
            df_raw = obtener_datos(ticker, dias_periodo, datetime.date.today())
            
            if df_raw is not None:
                df_proc = aplicar_calculo_financiero(df_raw, ui_deriv_window)
                df_pd = df_proc.to_pandas()
                
                # --- CÁLCULO DE LAS GRIEGAS EN VECTORIZADO EN LUGAR DE ITERROWS ---
                df_pd['Delta'], df_pd['Gamma'] = calcular_griegas_bs_vec(
                    df_pd['Close'].to_numpy(),
                    df_pd['Close'].to_numpy(),
                    30/252,
                    0.04,
                    df_pd['Volatilidad_20d'].to_numpy()
                )
                
                # ==========================================
                # PANEL 1: ESTADÍSTICA DESCRIPTIVA AVANZADA
                # ==========================================
                st.subheader(" Métricas Descriptivas y Análisis de Distribución")
                
                vol_anual = df_pd['Retorno_Diario'].std() * np.sqrt(252) * 100
                skewness = df_pd['Retorno_Diario'].skew()
                kurtosis = df_pd['Retorno_Diario'].kurtosis()
                var_95 = np.percentile(df_pd['Retorno_Diario'], 5) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Volatilidad Realizada (Anual)", f"{vol_anual:.2f}%")
                col2.metric("Sesgo de Retornos (Skewness)", f"{skewness:.3f}", help="Negativo significa colas izquierdas más largas (caídas abruptas).")
                col3.metric("Curtosis (Riesgo de Cola)", f"{kurtosis:.3f}", help="Mayor a 3 indica colas pesadas (eventos de Cisne Negro frecuentes).")
                col4.metric("Value at Risk (VaR 95% Diario)", f"{var_95:.2f}%", help="Pérdida máxima esperada para un día con 95% de confianza.")
                
                st.divider()
                
                # ==========================================
                # PANEL 2: INTERFAZ DE PESTAÑAS (TABS)
                # ==========================================
                tab_precio, tab_calculo, tab_ia , tab_ml_garch = st.tabs(["Gráfico del Precio & RSI & STOCHASTIC", " Cinemática y Derivadas Parciales", "Inferencia Predictiva", " Comité de Modelos (Ensemble ML & GARCH)"])
                
                with tab_precio:
                    st.subheader("Acción del Precio, RSI & Oscilador Estocástico")
                    
                    fig_p = make_subplots(
                        rows=3, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_width=[0.2, 0.2, 0.6]
                    )
                    
                    # --- FILA 1: Velas y Bandas de Bollinger ---
                    fig_p.add_trace(go.Candlestick(x=df_pd['Date'], open=df_pd['Open'], high=df_pd['High'], low=df_pd['Low'], close=df_pd['Close'], name='Precio'), row=1, col=1)
                    fig_p.add_trace(go.Scatter(x=df_pd['Date'], y=df_pd['Bollinger_Upper'], line=dict(color='rgba(255,255,255,0.3)', dash='dash'), name='Banda Sup'), row=1, col=1)
                    fig_p.add_trace(go.Scatter(x=df_pd['Date'], y=df_pd['Bollinger_Lower'], line=dict(color='rgba(255,255,255,0.3)', dash='dash'), name='Banda Inf', fill='tonexty', fillcolor='rgba(255,255,255,0.05)'), row=1, col=1)
                    
                    # --- FILA 2: RSI ---
                    fig_p.add_trace(go.Scatter(x=df_pd['Date'], y=df_pd['RSI'], line=dict(color='#9b59b6', width=2), name='RSI'), row=2, col=1)
                    fig_p.add_hline(y=70, line_dash="dot", line_color="#e74c3c", row=2, col=1)
                    fig_p.add_hline(y=30, line_dash="dot", line_color="#2ecc71", row=2, col=1)
                    
                    # --- FILA 3: Oscilador Estocástico (14, 9) ---
                    fig_p.add_trace(go.Scatter(x=df_pd['Date'], y=df_pd['Stoch_K'], line=dict(color='#00bcff', width=1.5), name='%K Estocástico'), row=3, col=1)
                    fig_p.add_trace(go.Scatter(x=df_pd['Date'], y=df_pd['Stoch_D'], line=dict(color='#f1c40f', width=1.5), name='%D (Sma 9)'), row=3, col=1)
                    fig_p.add_hline(y=80, line_dash="dot", line_color="#e74c3c", row=3, col=1)
                    fig_p.add_hline(y=20, line_dash="dot", line_color="#2ecc71", row=3, col=1)
                    
                    fig_p.update_layout(xaxis_rangeslider_visible=False, height=750, template="plotly_dark", margin=dict(t=10, b=10))
                    st.plotly_chart(fig_p, width='stretch')
                    st.markdown("---") 
                    
                with tab_calculo:
                    st.subheader("Análisis Matemático Avanzado")
                    
                    fig_c = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                        subplot_titles=('Cinemática de Precios (1° Derivada vs 2° Derivada)', 'Cálculo Estocástico (Sensibilidad Teórica de Opciones - Delta / Gamma)'))
                    
                    # Fila 1: Velocidad y Aceleración
                    fig_c.add_trace(go.Scatter(x=df_pd['Date'], y=df_pd['Velocidad'], line=dict(color='#3498db', width=1.5), name='Velocidad (f\')'), row=1, col=1)
                    # CORRECCIÓN: Se actualiza 'y' para consumir la columna sin tilde 'Aceleracion'
                    fig_c.add_trace(go.Scatter(x=df_pd['Date'], y=df_pd['Aceleracion'], line=dict(color='#e67e22', width=1.5), name='Aceleración (f\'\')'), row=1, col=1)
                    fig_c.add_hline(y=0, line_dash="solid", line_color="gray", row=1, col=1)
                    
                    # Fila 2: Delta y Gamma (Ecuación Black-Scholes)
                    fig_c.add_trace(go.Scatter(x=df_pd['Date'], y=df_pd['Delta'], line=dict(color='#2ecc71', width=1.5), name='Delta (∂C/∂S)'), row=2, col=1)
                    fig_c.add_trace(go.Scatter(x=df_pd['Date'], y=df_pd['Gamma'], line=dict(color='#e74c3c', width=1.5), name='Gamma (∂²C/∂S²)', yaxis="y2"), row=2, col=1)
                    
                    fig_c.update_layout(height=650, template="plotly_dark", margin=dict(t=30, b=10))
                    st.plotly_chart(fig_c, width='stretch')
                    
                    st.info("**:material/high_quality: Tip Cuantitativo:** Cuando la línea de **Aceleración (naranja)** cruza la línea de cero hacia abajo mientras la **Velocidad (azul)** está en máximos, confirma matemáticamente un punto de inflexión bajista en el precio.")
                    st.markdown("---") 

                with tab_ia:
                    st.subheader("Entrenamiento e Inferencia Predictiva por Redes LSTM")
                    
                    # Mostrar resultados guardados en el estado si existen
                    if st.session_state.lstm_results is not None:
                        res = st.session_state.lstm_results
                        prediccion = res['prediccion']
                        retorno_proyectado = res['retorno_proyectado']
                        
                        col_a, col_b = st.columns(2)
                        col_a.metric(label="Precio de Cierre Proyectado", value=f"${prediccion:.2f}", delta=f"{retorno_proyectado:.2f}%")
                        
                        if retorno_proyectado > 0:
                            col_b.success(":material/terminal: **Señal Operativa LSTM:** Estructura Secuencial de Tendencia Alcista")
                        else:
                            col_b.error(":material/error: **Señal Operativa LSTM:** Estructura Secuencial de Tendencia Bajista")
                    
                    if st.button("Ejecutar Simulación por Deep Learning"):
                        with st.spinner("Inicializando tensores y ajustando pesos de la red LSTM..."):
                            precios = df_pd['Close'].values
                            modelo, scaler = entrenar_modelo_lstm(precios, ventana=ui_ventana, epocas=ui_epocas)
                            
                            # Predicción de t+1
                            inputs = precios[-ui_ventana:]
                            inputs_esc = scaler.transform(inputs.reshape(-1, 1))
                            X_input = np.reshape(inputs_esc, (1, ui_ventana, 1))
                            pred_esc = modelo.predict(X_input, verbose=0)
                            prediccion = scaler.inverse_transform(pred_esc)[0][0]
                            
                            precio_actual = precios[-1]
                            retorno_proyectado = ((prediccion - precio_actual) / precio_actual) * 100
                            
                            # Guardar resultados en el estado y recargar la aplicación para mostrarlos de forma persistente
                            st.session_state.lstm_results = {
                                'prediccion': prediccion,
                                'retorno_proyectado': retorno_proyectado
                            }
                            st.rerun()
                            st.markdown("---") 

                with tab_ml_garch:        
                    st.markdown("### El Comité Cuantitativo")
                    st.markdown("En lugar de confiar en un solo modelo, analizamos el mercado desde tres dimensiones matemáticas distintas: **Secuencia Geométrica (LSTM)**, **Condiciones Estructurales (XGBoost)** y **Riesgo Estocástico (GARCH)**.")
                    
                    # Mostrar resultados guardados en el estado si existen
                    if st.session_state.comite_results is not None:
                        res = st.session_state.comite_results
                        pred_lstm = res['pred_lstm']
                        senal_lstm = res['senal_lstm']
                        senal_xgb = res['senal_xgb']
                        alerta_riesgo = res['alerta_riesgo']
                        vol_manana_garch = res['vol_manana_garch']
                        df_importancia = res['df_importancia']
                        
                        # --- DASHBOARD DE RESULTADOS ---
                        st.divider()
                        col1, col2, col3 = st.columns(3)
                        
                        # Tarjeta LSTM
                        color_lstm = "green" if senal_lstm == "ALCISTA" else "red"
                        col1.markdown(f"#### Red Neuronal LSTM")
                        col1.markdown(f"*(Memoria de la Serie de Tiempo)*")
                        col1.markdown(f"<h3 style='color:{color_lstm};'>{senal_lstm}</h3>", unsafe_allow_html=True)
                        col1.metric("Precio Proyectado", f"${pred_lstm:.2f}")

                        # Tarjeta XGBoost
                        color_xgb = "green" if senal_xgb == "ALCISTA" else "red"
                        col2.markdown(f"#### 🌳 XGBoost")
                        col2.markdown(f"*(Interacción de Derivadas y RSI)*")
                        col2.markdown(f"<h3 style='color:{color_xgb};'>{senal_xgb}</h3>", unsafe_allow_html=True)
                        col2.markdown(f"Confianza Basada en Features")

                        # Tarjeta GARCH
                        color_garch = "orange" if alerta_riesgo == "MODERADO" else ("red" if alerta_riesgo.startswith("CRÍTICO") else "green")
                        col3.markdown(f"####  Oráculo GARCH")
                        col3.markdown(f"*(Heterocedasticidad de Varianza)*")
                        col3.markdown(f"<h3 style='color:{color_garch};'>{alerta_riesgo}</h3>", unsafe_allow_html=True)
                        col3.metric("Volatilidad Proyectada Anual", f"{vol_manana_garch*100:.1f}%")

                        st.divider()

                        # --- EXPLICABILIDAD DE IA (XAI) ---
                        st.subheader("Explicabilidad del Modelo (Feature Importance - XGBoost)")
                        st.markdown("¿Qué indicadores matemáticos están impulsando la decisión del mercado hoy? El siguiente gráfico revela el peso de cada variable en la decisión del árbol de Gradient Boosting.")
                        
                        fig_xgb = px.bar(df_importancia, x='Importancia', y='Indicador', orientation='h', 
                                        color='Importancia', color_continuous_scale='Blues',
                                        title="Peso de las Variables en la Predicción Actual")
                        fig_xgb.update_layout(template="plotly_dark", height=400)
                        st.plotly_chart(fig_xgb, width='stretch')
                        st.markdown("---") 
                    
                    if st.button("Ejecutar Simulaciones de IA y Riesgo", width='stretch'):
                        with st.spinner("Compilando redes, ensamblando árboles y calculando varianza condicional..."):                    
                            # --- MODELO 1: LSTM (Tendencia Base) ---
                            precios = df_pd['Close'].values
                            # Se llama a la función unificada entrenar_modelo_lstm en lugar de la duplicada entrenar_lstm_1
                            modelo_lstm, scaler = entrenar_modelo_lstm(precios, ventana=ui_ventana, epocas=ui_epocas)
                            inputs_esc = scaler.transform(precios[-ui_ventana:].reshape(-1, 1))
                            pred_lstm_esc = modelo_lstm.predict(np.reshape(inputs_esc, (1, ui_ventana, 1)), verbose=0)
                            pred_lstm = scaler.inverse_transform(pred_lstm_esc)[0][0]
                            senal_lstm = "ALCISTA" if pred_lstm > precios[-1] else "BAJISTA"  
                            
                            # --- MODELO 2: XGBoost (Interacción de Indicadores) ---
                            modelo_xgb, df_importancia, features = entrenar_xgboost_1(df_pd)                    
                            datos_hoy = df_pd.iloc[-1:][features].astype(np.float64)                    
                            pred_xgb = modelo_xgb.predict(datos_hoy)[0]
                            senal_xgb = "ALCISTA" if pred_xgb == 1 else "BAJISTA"

                            # --- MODELO 3: GARCH(1,1) (Volatilidad y Riesgo) ---
                            vol_hoy = df_pd['Volatilidad_20d'].iloc[-1]
                            vol_manana_garch = predecir_riesgo_garch(df_pd['Retorno_Diario'])
                            alerta_riesgo = "CRÍTICO (Alta Turbulencia)" if vol_manana_garch > vol_hoy * 1.2 else ("ESTABLE" if vol_manana_garch < vol_hoy else "MODERADO")

                            # Guardar resultados en el estado y recargar la aplicación para mostrarlos de forma persistente
                            st.session_state.comite_results = {
                                'pred_lstm': pred_lstm,
                                'senal_lstm': senal_lstm,
                                'senal_xgb': senal_xgb,
                                'alerta_riesgo': alerta_riesgo,
                                'vol_manana_garch': vol_manana_garch,
                                'df_importancia': df_importancia
                            }
                            st.rerun()
                            st.markdown("---") 

    st.markdown("---")            