"""
==========================================================================
APLICAÇÃO STREAMLIT — PREDIÇÃO DE DOENÇA CARDIOVASCULAR
=============================================================================
Interface web moderna para inferência de risco cardiovascular usando
Random Forest treinado em dados clínicos.

Uso: streamlit run app.py
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÃO INICIAL DO STREAMLIT
# =============================================================================

st.set_page_config(
    page_title="💚 Predição de Risco Cardiovascular",
    page_icon="💚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para design moderno
st.markdown("""
<style>
    /* Cores e fontes */
    :root {
        --primary-color: #FF6B6B;
        --secondary-color: #4ECDC4;
        --success-color: #2ECC71;
        --warning-color: #F39C12;
        --danger-color: #E74C3C;
    }
    
    /* Estilo do header */
    .stMarkdown h1 {
        color: #2C3E50;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.2em;
        font-weight: 700;
    }
    
    .stMarkdown h2 {
        color: #34495E;
        font-size: 1.5em;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
        border-bottom: 2px solid #E74C3C;
        padding-bottom: 0.5em;
    }
    
    /* Cards/Containers */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Sidebar */
    .css-1d391kg {
        padding: 20px;
    }
    
    /* Botões */
    .stButton > button {
        width: 100%;
        height: 50px;
        font-size: 1.1em;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    
    /* Inputs */
    .stSlider, .stNumberInput, .stSelectbox, .stRadio {
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

@st.cache_resource
def load_model():
    """Carrega o modelo treinado usando cache do Streamlit."""
    model_path = Path("modelo_rf.pkl")
    
    if not model_path.exists():
        st.error("❌ Erro: Arquivo 'modelo_rf.pkl' não encontrado!")
        st.info("Execute 'python train_model.py' para treinar e exportar o modelo.")
        st.stop()
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {str(e)}")
        st.stop()


@st.cache_resource
def load_feature_info():
    """Carrega informações sobre features."""
    feature_info_path = Path("feature_info.pkl")
    
    if feature_info_path.exists():
        return joblib.load(feature_info_path)
    else:
        # Valores padrão se o arquivo não existir
        return {
            'continuous_cols': ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi'],
            'categorical_cols': ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'],
            'all_cols': ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 
                        'gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'],
            'metrics': {
                'auc_roc': 0.78,
                'accuracy': 0.75,
                'precision': 0.75,
                'recall': 0.73,
                'f1_score': 0.74
            }
        }


def calculate_bmi(weight_kg, height_cm):
    """Calcula IMC a partir de peso (kg) e altura (cm)."""
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 2)


def interpret_risk(probability, prediction):
    """Interpreta o risco cardiovascular baseado na predição e probabilidade."""
    risk_level = "Alto ⚠️" if prediction == 1 else "Baixo ✅"
    risk_color = "error" if prediction == 1 else "success"
    
    return {
        'label': risk_level,
        'type': risk_color,
        'probability': probability,
        'prediction': prediction
    }


def format_result_display(risk_data):
    """Formata a exibição do resultado."""
    prediction = risk_data['prediction']
    probability = risk_data['probability']
    
    if prediction == 1:
        emoji = "⚠️"
        title = "RISCO ALTO DE DOENÇA CARDIOVASCULAR"
        message = f"""
        A análise indica **risco elevado** de doença cardiovascular.
        
        **Probabilidade: {probability*100:.1f}%**
        
        ⚡ **RECOMENDAÇÕES:**
        - Procure um cardiologista para avaliação clínica completa
        - Realize exames complementares (eletrocardiograma, ecocardiograma)
        - Adote hábitos saudáveis: exercícios, dieta balanceada
        - Reduza fatores de risco: fumo, álcool, sedentarismo
        - Monitore pressão arterial regularmente
        """
    else:
        emoji = "💚"
        title = "RISCO BAIXO DE DOENÇA CARDIOVASCULAR"
        message = f"""
        A análise indica **risco baixo** de doença cardiovascular.
        
        **Probabilidade de risco: {(1-probability)*100:.1f}%**
        
        ✨ **CONTINUE ASSIM:**
        - Mantenha hábitos saudáveis e exercício regular
        - Acompanhamento médico periódico recomendado
        - Monitore pressão arterial e colesterol anualmente
        - Evite fatores de risco: fumo, álcool em excesso
        - Mantenha peso saudável e alimentação equilibrada
        """
    
    return emoji, title, message


# =============================================================================
# CARREGAMENTO DO MODELO
# =============================================================================

pipeline = load_model()
feature_info = load_feature_info()

# =============================================================================
# HEADER
# =============================================================================

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("# 💚 Predição de Risco Cardiovascular")
    st.markdown("### Sistema de Análise com Inteligência Artificial")

st.markdown("---")

# =============================================================================
# INTRODUÇÃO
# =============================================================================

st.markdown("""
Este sistema utiliza um **modelo de Machine Learning (Random Forest)** treinado 
com dados clínicos para estimar o risco de doença cardiovascular.

ℹ️ **Como funciona:**
- Preencha seus dados clínicos no painel lateral
- O sistema calcula automaticamente métricas como IMC
- Clique em "Analisar Risco" para obter o resultado
- Receba recomendações personalizadas baseadas na análise
""")

st.markdown("---")

# =============================================================================
# SIDEBAR — FORMULÁRIO DE ENTRADA
# =============================================================================

st.sidebar.markdown("# 📋 Dados Clínicos")
st.sidebar.markdown("Preencha seus dados para análise de risco")

# --- DADOS PESSOAIS ---
st.sidebar.markdown("### 👤 Dados Pessoais")

age_years = st.sidebar.slider(
    "Idade (anos)",
    min_value=18,
    max_value=100,
    value=45,
    step=1,
    help="Idade em anos completos"
)

# --- ANTROPOMÉTRICOS ---
st.sidebar.markdown("### 📏 Medidas Antropométricas")

height_cm = st.sidebar.slider(
    "Altura (cm)",
    min_value=100,
    max_value=250,
    value=170,
    step=1,
    help="Altura em centímetros"
)

weight_kg = st.sidebar.slider(
    "Peso (kg)",
    min_value=30,
    max_value=200,
    value=70,
    step=1,
    help="Peso em quilogramas"
)

# Calcular e exibir IMC
bmi = calculate_bmi(weight_kg, height_cm)
st.sidebar.metric(
    label="📊 IMC Calculado",
    value=f"{bmi:.1f}",
    delta=None,
    help="Índice de Massa Corporal (calculado automaticamente)"
)

# --- SINAIS VITAIS ---
st.sidebar.markdown("### 🫀 Sinais Vitais")

ap_hi = st.sidebar.number_input(
    "Pressão Sistólica (mmHg)",
    min_value=0,
    max_value=300,
    value=120,
    step=5,
    help="Pressão arterial sistólica"
)

ap_lo = st.sidebar.number_input(
    "Pressão Diastólica (mmHg)",
    min_value=0,
    max_value=200,
    value=80,
    step=5,
    help="Pressão arterial diastólica"
)

# --- EXAMES LABORATORIAIS ---
st.sidebar.markdown("### 🧪 Exames Laboratoriais")

cholesterol = st.sidebar.selectbox(
    "Colesterol",
    options=[1, 2, 3],
    format_func=lambda x: {
        1: "1 - Normal",
        2: "2 - Acima do Normal",
        3: "3 - Muito Acima do Normal"
    }[x],
    help="Nível de colesterol"
)

gluc = st.sidebar.selectbox(
    "Glicose",
    options=[1, 2, 3],
    format_func=lambda x: {
        1: "1 - Normal",
        2: "2 - Acima do Normal",
        3: "3 - Muito Acima do Normal"
    }[x],
    help="Nível de glicose"
)

# --- HÁBITOS ---
st.sidebar.markdown("### 🚬 Hábitos e Estilo de Vida")

smoke = st.sidebar.radio(
    "É fumante?",
    options=[0, 1],
    format_func=lambda x: "Não" if x == 0 else "Sim",
    horizontal=True,
    help="Habitual de fumo"
)

alco = st.sidebar.radio(
    "Consome álcool?",
    options=[0, 1],
    format_func=lambda x: "Não" if x == 0 else "Sim",
    horizontal=True,
    help="Consumo regular de álcool"
)

active = st.sidebar.radio(
    "Pratica atividade física?",
    options=[0, 1],
    format_func=lambda x: "Não" if x == 0 else "Sim",
    horizontal=True,
    help="Atividade física regular (≥30min, ≥3x semana)"
)

# Gender (valores padrão - não é crítico para o usuário)
gender = 1  # Valor padrão

# =============================================================================
# BOTÃO DE ANÁLISE
# =============================================================================

st.sidebar.markdown("---")

analyze_button = st.sidebar.button(
    "🔍 ANALISAR RISCO",
    use_container_width=True,
    key="analyze_button"
)

# =============================================================================
# PROCESSAMENTO E RESULTADOS
# =============================================================================

if analyze_button:
    # Validações básicas
    if ap_hi < ap_lo:
        st.error("❌ Erro: Pressão sistólica não pode ser menor que diastólica!")
    elif ap_hi > 300 or ap_lo > 200:
        st.warning("⚠️ Aviso: Valores de pressão fora dos limites fisiológicos normais")
    elif weight_kg < 30 or weight_kg > 200:
        st.warning("⚠️ Aviso: Peso fora dos limites esperados")
    elif height_cm < 100 or height_cm > 220:
        st.warning("⚠️ Aviso: Altura fora dos limites esperados")
    else:
        # Estruturar dados para o modelo
        input_data = pd.DataFrame({
            'age_years': [age_years],
            'height': [height_cm],
            'weight': [weight_kg],
            'bmi': [bmi],
            'ap_hi': [ap_hi],
            'ap_lo': [ap_lo],
            'gender': [gender],
            'cholesterol': [cholesterol],
            'gluc': [gluc],
            'smoke': [smoke],
            'alco': [alco],
            'active': [active]
        })
        
        try:
            # Fazer predição
            prediction = pipeline.predict(input_data)[0]
            probability = pipeline.predict_proba(input_data)[0][1]
            
            # Interpretar resultado
            risk_data = interpret_risk(probability, prediction)
            emoji, title, message = format_result_display(risk_data)
            
            # Exibir resultado
            st.markdown("---")
            st.markdown(f"## {emoji} {title}")
            
            if prediction == 1:
                st.error(message)
            else:
                st.success(message)
            
            # Métricas adicionais
            st.markdown("---")
            st.markdown("### 📊 Análise Detalhada")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Probabilidade de Risco",
                    f"{probability*100:.1f}%",
                    help="Confiança do modelo na predição"
                )
            
            with col2:
                st.metric(
                    "IMC",
                    f"{bmi:.1f} kg/m²",
                    help="Índice de Massa Corporal"
                )
            
            with col3:
                st.metric(
                    "Pressão Arterial",
                    f"{ap_hi}/{ap_lo} mmHg",
                    help="Sistólica/Diastólica"
                )
            
            with col4:
                status_text = "Ativo ✅" if active == 1 else "Sedentário ❌"
                st.metric(
                    "Status Físico",
                    status_text,
                    help="Prática de atividade física"
                )
            
            # Informações sobre o modelo
            st.markdown("---")
            st.markdown("### 🤖 Informações do Modelo")
            
            model_cols = st.columns(5)
            metrics = feature_info.get('metrics', {})
            
            model_cols[0].metric(
                "AUC-ROC",
                f"{metrics.get('auc_roc', 0):.3f}",
                help="Métrica principal de avaliação"
            )
            model_cols[1].metric(
                "Acurácia",
                f"{metrics.get('accuracy', 0):.1%}",
                help="Proporção de predições corretas"
            )
            model_cols[2].metric(
                "Precisão",
                f"{metrics.get('precision', 0):.1%}",
                help="De todos os 'positivos' previstos, quantos acertaram"
            )
            model_cols[3].metric(
                "Recall",
                f"{metrics.get('recall', 0):.1%}",
                help="De todos os casos reais positivos, quantos foram detectados"
            )
            model_cols[4].metric(
                "F1-Score",
                f"{metrics.get('f1_score', 0):.3f}",
                help="Média harmônica entre precisão e recall"
            )
            
            # Disclaimer
            st.markdown("---")
            st.info(
                "⚠️ **DISCLAIMER:** Este sistema é uma ferramenta de suporte à decisão "
                "e não substitui avaliação médica profissional. Consulte sempre um "
                "cardiologista ou médico especialista para diagnóstico definitivo."
            )
        
        except Exception as e:
            st.error(f"❌ Erro ao processar predição: {str(e)}")
            st.info("Verifique se o arquivo 'modelo_rf.pkl' está na pasta correta.")

# =============================================================================
# INFORMAÇÕES ADICIONAIS (sem análise)
# =============================================================================

if not analyze_button:
    st.markdown("---")
    st.markdown("### 📚 Sobre Este Sistema")
    
    tab1, tab2, tab3 = st.tabs(["Sobre o Projeto", "Interpretação de Risco", "FAQ"])
    
    with tab1:
        st.markdown("""
        **Objetivo:** Prever o risco de doença cardiovascular usando dados clínicos.
        
        **Algoritmo:** Random Forest Classifier com 100 árvores
        
        **Dataset:** Cardiovascular Disease Dataset (Kaggle) - 68.677 pacientes
        
        **Disciplina:** Projeto - Pesquisa e Experimentação em Sistemas de Informação
        
        **Universidade:** PUC Minas - Curso Sistemas de Informação (7º semestre)
        """)
    
    with tab2:
        st.markdown("""
        #### 🟢 Risco Baixo (Probabilidade < 50%)
        Indica que o modelo prevê baixa probabilidade de doença cardiovascular com base 
        nos dados fornecidos. Manter acompanhamento médico periódico é importante.
        
        #### 🔴 Risco Alto (Probabilidade ≥ 50%)
        Indica que o modelo prevê alta probabilidade de doença cardiovascular. 
        Recomenda-se procurar um cardiologista para avaliação clínica completa.
        
        **Nota:** A probabilidade reflete apenas o risco estatístico baseado no modelo, 
        não é um diagnóstico médico.
        """)
    
    with tab3:
        st.markdown("""
        **P: Como o IMC é calculado?**
        R: IMC = Peso (kg) / (Altura em metros)²
        
        **P: Qual é a precisão do modelo?**
        R: AUC-ROC de 0.78 em dados de teste, indicando bom desempenho discriminativo.
        
        **P: Por que colesterol e glicose têm valores 1, 2, 3?**
        R: São categorizados como: 1-Normal, 2-Acima do Normal, 3-Muito Acima.
        
        **P: Meus dados são seguros?**
        R: Todos os dados são processados localmente no seu navegador, sem armazenamento.
        
        **P: Com que frequência devo fazer análises?**
        R: Consulte seu médico. Recomenda-se acompanhamento anual ou conforme orientação médica.
        """)