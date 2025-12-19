"""
app.py
======
Dashboard Streamlit multipage com sidebar para exploração de modelo de Selic 2026.

Páginas disponíveis:
1. Dashboard: Visualização em tempo real de Selic, Focus, Probabilidades
2. Modelo (Tutorial): Explicação passo-a-passo para leigos
3. Análise Detalhada: Gráficos, tabelas, estatísticas
4. Simulação MC: Trajetórias possíveis de Selic até dez/2026
5. Histórico: Série histórica (placeholder)

Autor: Finance Dashboard Team
Data: 2025-12-18
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging

# Imports do projeto
from data_fetcher import get_fetcher
from model import create_calibrated_model, MonteCarloSimulator
from notifier import get_notifier
from evaluation import (
    get_conditional_copom_probs,
    benchmark_vectorized_vs_loop,
    backtest_simple,
)
from config import (
    TIMEZONE,
    PARAM_UNCERTAINTY_ALPHA,
    SHOCK_PROB,
    SHOCK_MAG,
    FOCUS_MEDIAN_DEC_2026,
)

# ============================================================================
# CONFIG STREAMLIT
# ============================================================================

st.set_page_config(
    page_title="Selic Dashboard 2026",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado (melhor estética para cards de cenário)
st.markdown("""
<style>
    /* Reset pequeno para tipografia */
    .stApp { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; }

    .metric-card {
        background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
        padding: 20px 22px;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(16,24,40,0.06);
        border: 1px solid rgba(31,41,55,0.06);
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .metric-card h3 { margin: 0; font-size: 16px; color: #374151; font-weight: 600; }
    .metric-card .metric-value { font-size: 34px; font-weight: 700; color: #0f172a; margin: 6px 0; }
    .metric-card p { margin: 0; color: #6b7280; font-size: 13px; line-height: 1.3; }

    .metric-bar { height: 8px; background: rgba(99,102,241,0.08); border-radius: 999px; margin-top: 12px; overflow: hidden; }
    .metric-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg,#4f46e5,#06b6d4); box-shadow: 0 3px 8px rgba(79,70,229,0.18); }

    .success-card .metric-fill { background: linear-gradient(90deg,#10b981,#34d399); box-shadow: 0 3px 8px rgba(16,185,129,0.12); }
    .warning-card .metric-fill { background: linear-gradient(90deg,#fb7185,#fb923c); box-shadow: 0 3px 8px rgba(251,113,133,0.12); }

    /* Ajustes responsivos para colunas */
    @media (max-width: 900px) {
        .metric-card .metric-value { font-size: 26px; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SETUP LOGGING
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# FUNCTIONS AUXILIARES
# ============================================================================


@st.cache_resource
def load_data():
    """Carrega dados e modelo em cache (atualizado a cada 1 hora)."""
    fetcher = get_fetcher()
    data = fetcher.get_all_current_data()

    model = create_calibrated_model(
        focus_median=data["focus_selic"],
        selic_current=data["selic"],
    )

    return data, model


@st.cache_data(ttl=3600)  # 1 hora
def run_monte_carlo():
    """Executa simulação MC e cachea resultados."""
    simulator = MonteCarloSimulator()
    trajectories, df_terminal = simulator.simulate()
    return trajectories, df_terminal


def run_monte_carlo_advanced(
    n_sims: int,
    use_vectorized: bool = True,
    use_conditional_probs: bool = False,
    use_param_uncertainty: bool = False,
    use_shocks: bool = False,
    ipca: float = 4.46,
):
    """Executa simulação MC com opções avançadas."""
    simulator = MonteCarloSimulator(n_simulations=n_sims, n_steps=8)
    
    kwargs = {}
    
    if use_conditional_probs:
        probs = get_conditional_copom_probs(selic=15.0, ipca_12m=ipca)
        kwargs['copom_probs_base'] = probs
    
    if use_param_uncertainty:
        kwargs['param_uncertainty_alpha'] = PARAM_UNCERTAINTY_ALPHA
    else:
        kwargs['param_uncertainty_alpha'] = 0.0
    
    if use_shocks:
        kwargs['shock_prob'] = SHOCK_PROB
        kwargs['shock_mag'] = SHOCK_MAG
    else:
        kwargs['shock_prob'] = 0.0
    
    if use_vectorized:
        trajectories, df_terminal = simulator.simulate_vectorized(**kwargs)
    else:
        trajectories, df_terminal = simulator.simulate()
    
    return trajectories, df_terminal


def create_gauge_chart(value: float, min_val: float, max_val: float, title: str):
    """Cria gráfico de gauge (velocímetro)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, (max_val + min_val) / 3], 'color': "lightgray"},
                {'range': [(max_val + min_val) / 3, 2 * (max_val + min_val) / 3], 'color': "gray"},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 12.13,
            }
        }
    ))
    fig.update_layout(height=300)
    return fig


# ============================================================================
# SIDEBAR - NAVEGAÇÃO
# ============================================================================

st.sidebar.title("📊 Selic Dashboard 2026")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Selecione a página:",
    [
        "📈 Dashboard",
        "🎓 Tutorial (Leigo)",
        "📉 Análise Detalhada",
        "🎲 Simulação Monte Carlo",
        "📋 Histórico",
    ],
)

st.sidebar.markdown("---")

with st.sidebar.expander("ℹ️ Sobre este projeto"):
    st.markdown("""
    **Selic Probability Dashboard** é um modelo estatístico que:
    
    1. Coleta dados em tempo real de:
       - Taxa Selic (BC Brasil)
       - IPCA (Inflação - IBGE)
       - Focus Bulletin (Expectativas de mercado)
    
    2. Calibra uma distribuição de probabilidade para:
       - Selic terminal em dezembro de 2026
       - Trajetórias possíveis até lá
    
    3. Envia alertas via WhatsApp quando há mudanças significativas
    
    **Metodologia**: Distribuição discreta + Normal truncada + Monte Carlo
    
    **Atualização**: Cache de 1 hora (dados em tempo real, análise leve)
    """)

st.sidebar.markdown("---")

if st.sidebar.checkbox("🔄 Atualizar dados agora", value=False):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

# Opções avançadas para simulação
with st.sidebar.expander("⚙️ Opções Avançadas (Simulação)"):
    st.markdown("**Configurações do Monte Carlo**")
    
    use_conditional_probs = st.checkbox(
        "📊 Usar decisões condicionais (gaps macro)",
        value=False,
        help="Ajusta probs COPOM baseado em IPCA gap vs meta"
    )
    
    use_param_uncertainty = st.checkbox(
        "🔀 Incerteza de parâmetros (Dirichlet)",
        value=False,
        help="Adiciona ruído às probabilidades por simulação"
    )
    
    use_shocks = st.checkbox(
        "⚡ Incluir choques extremos",
        value=False,
        help="Pequena prob de choques de ±1.5pp"
    )
    
    use_vectorized = st.checkbox(
        "⚡ Usar MC vetorizado (fast)",
        value=True,
        help="Versão NumPy pura vs loop"
    )
    
    n_sims = st.slider(
        "Número de simulações",
        min_value=1000,
        max_value=100000,
        value=10000,
        step=1000,
        help="Mais simulações = mais preciso, mais lento"
    )

st.sidebar.markdown("---")
st.sidebar.text_area(
    "📞 Sugestões / Bugs",
    placeholder="Descreva aqui...",
    height=80,
    disabled=True,
)

# ============================================================================
# PÁGINA 1: DASHBOARD
# ============================================================================

if page == "📈 Dashboard":
    st.title("💰 Dashboard Selic 2026")
    st.markdown("*Visualização em tempo real de probabilidades e projeções*")

    # Carregar dados
    data, model = load_data()
    notifier = get_notifier()

    # Verificar mudanças significativas (se WhatsApp estiver configurado)
    notifier.check_and_notify_selic_change(data["selic"])
    notifier.check_and_notify_focus_change(data["focus_selic"])

    # KPIs - Primeira linha
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "💵 Selic Corrente",
            f"{data['selic']:.2f}%",
            delta=f"{data['selic'] - 15.00:.2f}pp desde 15,00%",
            delta_color="inverse",
        )

    with col2:
        st.metric(
            "📊 Focus 2026",
            f"{data['focus_selic']:.2f}%",
            delta=f"{data['focus_selic'] - data['selic']:.2f}pp vs Selic atual",
            delta_color="normal",
        )

    with col3:
        st.metric(
            "📈 IPCA 12m",
            f"{data['ipca_12m']:.2f}%",
            delta="Na meta" if 3.5 <= data['ipca_12m'] <= 4.5 else "Fora da meta",
            delta_color="normal",
        )

    with col4:
        expected = model.expected_value()
        st.metric(
            "🎯 E[Selic] dez/2026",
            f"{expected:.2f}%",
            delta=f"{abs(expected - data['focus_selic']):.2f}pp vs Focus",
            delta_color="off",
        )

    st.markdown("---")

    # Distribuição de probabilidade
    st.subheader("📊 Distribuição de Probabilidade - Selic em dez/2026")

    df_probs = model.get_probabilities()

    fig_dist = px.bar(
        df_probs,
        x="selic_level",
        y="pct_display",
        labels={"selic_level": "Selic (%)", "pct_display": "Probabilidade (%)"},
        title="Distribuição de Selic Terminal",
        color="pct_display",
        color_continuous_scale="Blues",
    )
    fig_dist.add_vline(x=data["focus_selic"], line_dash="dash", line_color="red", annotation_text="Focus")
    fig_dist.add_vline(x=expected, line_dash="dash", line_color="green", annotation_text="E[X]")
    st.plotly_chart(fig_dist, use_container_width=True)

    # Cenários macro
    st.subheader("🎭 Cenários Macroeconômicos")

    scenarios = model.interpret_distribution()

    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        prob_dovish, desc_dovish = scenarios["Dovish (corte agressivo)"]
        st.markdown(f"""
        <div class='metric-card success-card'>
            <div>
                <h3>🕊️ Dovish</h3>
                <div class='metric-value'>{prob_dovish*100:.1f}%</div>
                <p>{desc_dovish}</p>
            </div>
            <div class='metric-bar'><div class='metric-fill' style='width: {prob_dovish*100:.1f}%;'></div></div>
        </div>
        """, unsafe_allow_html=True)

    with col_s2:
        prob_central, desc_central = scenarios["Central (corte gradual)"]
        st.markdown(f"""
        <div class='metric-card'>
            <div>
                <h3>⚖️ Central</h3>
                <div class='metric-value'>{prob_central*100:.1f}%</div>
                <p>{desc_central}</p>
            </div>
            <div class='metric-bar'><div class='metric-fill' style='width: {prob_central*100:.1f}%;'></div></div>
        </div>
        """, unsafe_allow_html=True)

    with col_s3:
        prob_hawkish, desc_hawkish = scenarios["Hawkish (hold conservador)"]
        st.markdown(f"""
        <div class='metric-card warning-card'>
            <div>
                <h3>🦅 Hawkish</h3>
                <div class='metric-value'>{prob_hawkish*100:.1f}%</div>
                <p>{desc_hawkish}</p>
            </div>
            <div class='metric-bar'><div class='metric-fill' style='width: {prob_hawkish*100:.1f}%;'></div></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Estatísticas
    st.subheader("📈 Estatísticas da Distribuição")

    stats = model.get_stats()

    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)

    with col_s1:
        st.metric("Mínimo", f"{stats['min']:.2f}%")
    with col_s2:
        st.metric("P10", f"{stats['p10']:.2f}%")
    with col_s3:
        st.metric("Mediana (P50)", f"{stats['p50']:.2f}%")
    with col_s4:
        st.metric("P90", f"{stats['p90']:.2f}%")
    with col_s5:
        st.metric("Máximo", f"{stats['max']:.2f}%")

    st.markdown("---")

    # Última atualização
    st.caption(f"⏰ Última atualização: {data['last_update'].strftime('%d/%m/%Y %H:%M:%S')}")

# ============================================================================
# PÁGINA 2: TUTORIAL PARA LEIGOS
# ============================================================================

elif page == "🎓 Tutorial (Leigo)":
    st.title("🎓 Entendendo o Modelo (Explicado Simples)")

    st.markdown("""
    Este tutorial explica como o modelo de Selic funciona, **sem jargão técnico**.
    
    Se você está aqui pela primeira vez, comece pelo tópico 1 e vá em ordem!
    """)

    topic = st.radio(
        "Escolha um tópico:",
        [
            "1️⃣ O que é Selic?",
            "2️⃣ Por que prever Selic?",
            "3️⃣ O que é Focus?",
            "4️⃣ Como o modelo funciona?",
            "5️⃣ O que são probabilidades?",
            "6️⃣ Cenários dovish/hawkish",
        ],
    )

    if topic == "1️⃣ O que é Selic?":
        st.header("O que é Selic?")
        st.markdown("""
        **Selic** = Taxa de juros básica do Brasil.
        
        Pense assim:
        - Você vai a um banco pedir um empréstimo
        - O banco cobra juros, certo?
        - Essa taxa de juros base vem da Selic
        
        **Selic alta (15%) = empréstimo caro**
        - Você paga mais ao pedir dinheiro emprestado
        - As empresas investem menos (caro demais)
        - As pessoas poupam mais (vale a pena)
        
        **Selic baixa (10%) = empréstimo barato**
        - Você paga menos ao pedir dinheiro emprestado
        - As empresas investem mais (barato)
        - As pessoas poupam menos (não vale a pena)
        
        Quem decide a Selic é o **Banco Central** do Brasil.
        """)

        st.info("""
        💡 Exemplo prático:
        - Você quer comprar um carro no banco com Selic em 15%
        - O financiamento sai caro!
        - Se Selic cair para 12%, o mesmo carro fica mais barato de comprar
        """)

    elif topic == "2️⃣ Por que prever Selic?":
        st.header("Por que prever Selic?")
        st.markdown("""
        Se você sabe **para onde a Selic vai**, você pode tomar decisões melhores:
        
        **Se espera que Selic CAIA (15% → 12%)**
        - 📊 Aplicações de renda fixa vão render menos? Talvez buscar risco
        - 🏠 Crédito imobiliário fica mais barato? Pode ser hora de comprar casa
        - 📈 Ações tendem a subir quando juros caem? Geralmente sim
        
        **Se espera que Selic SUBA (15% → 17%)**
        - 📊 Renda fixa vai render mais? CDI fica atrativo
        - 🏠 Crédito fica mais caro? Melhor evitar endividamento longo
        - 📉 Ações tendem a sofrer com juros altos? Geralmente sim
        
        O mercado inteiro tenta antecipar a Selic: bancos, fundos, traders.
        """)

    elif topic == "3️⃣ O que é Focus?":
        st.header("O que é Focus?")
        st.markdown("""
        **Focus** = Pesquisa semanal do Banco Central com o mercado.
        
        Funciona assim:
        1. O BC pergunta a vários analistas: "Qual sua previsão de Selic para dezembro de 2026?"
        2. Cada um dá um número (12%, 11,5%, 13%...)
        3. O BC publica a **mediana** desses palpites
        4. Esse número é o Focus
        
        Focus é importante porque:
        - Resume a visão do mercado
        - Impacta preço de DI, NTN-B, câmbio, bolsa
        """)

    elif topic == "4️⃣ Como o modelo funciona?":
        st.header("Como o modelo funciona?")
        st.markdown("""
        O modelo segue 3 passos simples:
        
        1. **Coleta dados reais**
           - Selic atual
           - Focus para dez/2026
           - IPCA 12m
        
        2. **Cria cenários**
           - 🕊️ Dovish: Selic termina em 11–11,5%
           - ⚖️ Central: Selic termina em 12–12,5% (perto do Focus)
           - 🦅 Hawkish: Selic termina em 13–13,5%
        
        3. **Atribui probabilidades**
           - Exemplo: 22% de chance de 12,0%; 23% de chance de 12,5%; etc.
           - A média da distribuição fica igual ao Focus (~12,1%)
        """)

    elif topic == "5️⃣ O que são probabilidades?":
        st.header("O que são probabilidades?")
        st.markdown("""
        **Probabilidade** é a chance de algo acontecer.
        
        - Moeda: 50% cara, 50% coroa
        - Dado: ~16,7% para cada número
        
        No nosso caso:
        - 22% de chance da Selic terminar em 12,0%
        - 23% de chance da Selic terminar em 12,5%
        - 16% de chance da Selic terminar em 13,0%
        
        Como o futuro é incerto, usamos probabilidades, não certezas.
        """)

    elif topic == "6️⃣ Cenários dovish/hawkish":
        st.header("Dovish vs Hawkish: O que significa?")
        st.markdown("""
        Termos inspirados em animais:
        
        - **Dove (pomba)**: suave, pacífica → juros BAIXOS
        - **Hawk (falcão)**: agressivo, atento → juros ALTOS
        
        🕊️ **Dovish**:
        - BC corta juros agressivamente
        - Focado em estimular economia
        
        🦅 **Hawkish**:
        - BC mantém juros altos
        - Focado em controlar inflação
        
        ⚖️ **Central**:
        - Meio termo: corta com cuidado, sem exageros
        """)

# ============================================================================
# PÁGINA 3: ANÁLISE DETALHADA
# ============================================================================

elif page == "📉 Análise Detalhada":
    st.title("📉 Análise Detalhada - Modelo e Dados")

    data, model = load_data()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Tabela de Probabilidades", "📈 Estatísticas", "🎯 Intervalos", "📋 Dados Brutos"]
    )

    with tab1:
        df_probs = model.get_probabilities()
        df_probs_display = df_probs.copy()
        df_probs_display["probability"] = (df_probs_display["probability"] * 100).round(2).astype(str) + "%"
        df_probs_display["cumulative"] = (df_probs_display["cumulative"] * 100).round(2).astype(str) + "%"
        df_probs_display["pct_display"] = df_probs_display["pct_display"].round(2)

        df_probs_display.columns = [
            "Selic (%)",
            "Probabilidade",
            "Cumulativa",
            "Probabilidade (%)",
        ]
        st.dataframe(df_probs_display, use_container_width=True)

    with tab2:
        st.subheader("Estatísticas Descritivas")
        stats = model.get_stats()
        st.json(stats)

    with tab3:
        st.subheader("Intervalos de Confiança")

        col1, col2 = st.columns(2)

        with col1:
            prob_68 = model.probability_range(
                stats["mean"] - stats["std"],
                stats["mean"] + stats["std"],
            )
            st.metric(
                "68% CI (μ ±σ)",
                f"{stats['mean'] - stats['std']:.2f}% a {stats['mean'] + stats['std']:.2f}%",
                f"{prob_68*100:.1f}% real",
            )

        with col2:
            prob_95 = model.probability_range(
                stats["mean"] - 2 * stats["std"],
                stats["mean"] + 2 * stats["std"],
            )
            st.metric(
                "95% CI (μ ±2σ)",
                f"{stats['mean'] - 2*stats['std']:.2f}% a {stats['mean'] + 2*stats['std']:.2f}%",
                f"{prob_95*100:.1f}% real",
            )

    with tab4:
        st.subheader("Dados Brutos de Entrada")

        df_raw = pd.DataFrame({
            "Variável": ["Selic Corrente", "Focus Dez/2026", "IPCA 12m", "IPCA Meta", "Desvio Padrão"],
            "Valor": [
                f"{data['selic']:.2f}%",
                f"{data['focus_selic']:.2f}%",
                f"{data['ipca_12m']:.2f}%",
                "4,50% (teto)",
                f"{model.selic_std:.2f}pp",
            ],
            "Data": [
                data['selic_date'].strftime('%d/%m/%Y'),
                data['focus_date'].strftime('%d/%m/%Y'),
                data['ipca_date'].strftime('%d/%m/%Y'),
                "Meta BC",
                "Calibrado",
            ],
        })

        st.dataframe(df_raw, use_container_width=True, hide_index=True)

# ============================================================================
# PÁGINA 4: SIMULAÇÃO MONTE CARLO
# ============================================================================

elif page == "🎲 Simulação Monte Carlo":
    st.title("🎲 Simulação Monte Carlo - Trajetórias de Selic")

    st.markdown("""
    A simulação Monte Carlo roda **múltiplos cenários** possíveis de decisões do COPOM.
    
    Cada cenário é uma "história" diferente até dez/2026:
    - Pode o COPOM cortar em janeiro, março, maio, etc.
    - Pode cortar 25 ou 50 bps por reunião
    - Pode pausar os cortes em algum momento
    """)

    # Recuperar opções avançadas do sidebar
    use_conditional_probs = st.session_state.get('use_conditional_probs', False)
    use_param_uncertainty = st.session_state.get('use_param_uncertainty', False)
    use_shocks = st.session_state.get('use_shocks', False)
    use_vectorized = st.session_state.get('use_vectorized', True)
    n_sims = st.session_state.get('n_sims', 10000)

    # Status das opções
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"""
        **Configuração Ativa:**
        - Simulações: {n_sims:,}
        - Método: {'Vetorizado' if use_vectorized else 'Loop'}
        - Condicional COPOM: {'✓' if use_conditional_probs else '✗'}
        - Incerteza Params: {'✓' if use_param_uncertainty else '✗'}
        - Choques Extremos: {'✓' if use_shocks else '✗'}
        """)
    
    with col2:
        st.info("Ajuste as opções no painel esquerdo (⚙️ Opções Avançadas)")

    if st.button("🔄 Rodar simulação", key="run_mc"):
        with st.spinner(f"Simulando {n_sims:,} trajetórias..."):
            data, model = load_data()
            trajectories, df_terminal = run_monte_carlo_advanced(
                n_sims=n_sims,
                use_vectorized=use_vectorized,
                use_conditional_probs=use_conditional_probs,
                use_param_uncertainty=use_param_uncertainty,
                use_shocks=use_shocks,
                ipca=data['ipca_12m'],
            )

        # Gráfico de trajetórias
        st.subheader("Trajetórias de Selic (10k simulações)")

        fig_mc = go.Figure()

        # Adicionar algumas trajetórias com baixa opacidade
        for sim in range(min(100, len(trajectories))):  # Mostrar apenas 100 pra não ficar pesado
            fig_mc.add_trace(go.Scatter(
                y=trajectories[sim],
                mode='lines',
                line=dict(color='rgba(31, 119, 180, 0.1)', width=1),
                showlegend=False,
                hoverinfo='skip',
            ))

        # Percentis
        percentil_10 = np.percentile(trajectories, 10, axis=0)
        percentil_50 = np.percentile(trajectories, 50, axis=0)
        percentil_90 = np.percentile(trajectories, 90, axis=0)

        fig_mc.add_trace(go.Scatter(
            y=percentil_50,
            mode='lines',
            name='P50 (Mediana)',
            line=dict(color='green', width=3),
        ))

        fig_mc.add_trace(go.Scatter(
            y=percentil_10,
            mode='lines',
            name='P10',
            line=dict(color='red', width=2, dash='dash'),
        ))

        fig_mc.add_trace(go.Scatter(
            y=percentil_90,
            mode='lines',
            name='P90',
            line=dict(color='blue', width=2, dash='dash'),
        ))

        fig_mc.update_layout(
            title="Distribuição de Trajetórias Possíveis de Selic até dez/2026",
            xaxis_title="Reunião COPOM",
            yaxis_title="Selic (%)",
            height=500,
            hovermode='x unified',
        )

        st.plotly_chart(fig_mc, use_container_width=True)

        # Distribuição terminal do MC
        st.subheader("Distribuição Terminal (resultado das 10k simulações)")

        fig_terminal = px.histogram(
            df_terminal,
            x='selic_terminal',
            nbins=30,
            labels={'selic_terminal': 'Selic em dez/2026 (%)', 'count': 'Frequência'},
            title='Distribuição de Selic Terminal (Monte Carlo)',
        )

        st.plotly_chart(fig_terminal, use_container_width=True)

        # Estatísticas do MC
        st.subheader("📊 Resumo da Simulação")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Mínimo (MC)", f"{df_terminal['selic_terminal'].min():.2f}%")
        with col2:
            st.metric("P25", f"{df_terminal['selic_terminal'].quantile(0.25):.2f}%")
        with col3:
            st.metric("Mediana (MC)", f"{df_terminal['selic_terminal'].median():.2f}%")
        with col4:
            st.metric("P75", f"{df_terminal['selic_terminal'].quantile(0.75):.2f}%")
        with col5:
            st.metric("Máximo (MC)", f"{df_terminal['selic_terminal'].max():.2f}%")
    else:
        st.info("Clique no botão acima para executar a simulação.")

# ============================================================================
# PÁGINA 5: AVALIAÇÃO (BACKTEST & BENCHMARK)
# ============================================================================

elif page == "📋 Histórico":
    st.title("📋 Histórico & Avaliação")

    tab1, tab2, tab3 = st.tabs(
        ["📊 Histórico Selic", "🔍 Backtest", "⚡ Benchmark"]
    )

    with tab1:
        st.subheader("Série Histórica de Selic")
        st.info("Dados históricos seriam carregados de `data/selic_history.csv` (não fornecido neste momento)")
        st.markdown("""
        Para adicionar histórico:
        1. Crie arquivo `data/selic_history.csv` com colunas:
           - `date`: data em YYYY-MM-DD
           - `selic`: Selic corrente (%)
           - `ipca_12m`: IPCA 12 meses (%)
           - `selic_realized`: Selic realizada (ex-post)
        2. Recarregue a página
        """)

    with tab2:
        st.subheader("Backtest do Modelo")
        st.markdown("""
        O backtest avalia como o modelo teria performado em pontos históricos.
        
        **Métricas**:
        - **Brier Score**: Média de (prob - outcome)² para eventos binários (0-1)
        - **CRPS**: Continuous Ranked Probability Score para distribuições contínuas
        
        Menores valores indicam melhor calibração.
        """)

        if st.button("🔄 Executar backtest", key="run_backtest"):
            st.warning("⏳ Backtest requer `data/selic_history.csv` com histórico de realizações.")
            st.info("""
            Para rodar backtest:
            1. Forneça arquivo `data/selic_history.csv` com colunas: date, selic, ipca_12m, selic_realized
            2. Clique novamente
            
            O backtest calcula Brier Score e CRPS em cada ponto histórico.
            """)

    with tab3:
        st.subheader("Benchmark: Vetorizado vs Loop")
        st.markdown("""
        Compara performance entre:
        - **Vetorizado**: Versão NumPy pura (operações vetorizadas)
        - **Loop**: Versão original com loops Python
        """)

        if st.button("🔄 Rodar benchmark", key="run_benchmark"):
            with st.spinner("Benchmarking (testando 1k, 5k, 10k simulações)..."):
                df_bench = benchmark_vectorized_vs_loop(n_sims_list=[1000, 5000, 10000])

            st.dataframe(df_bench, use_container_width=True)

            # Gráfico speedup
            fig_speedup = px.bar(
                df_bench,
                x='n_simulations',
                y='speedup',
                labels={'n_simulations': 'Número de Simulações', 'speedup': 'Speedup (Loop / Vetorizado)'},
                title='Speedup: Versão Loop vs Vetorizada',
                color='speedup',
                color_continuous_scale='Reds',
            )

            fig_speedup.axhline(y=1.0, line_dash='dash', line_color='black', annotation_text='Paridade')
            st.plotly_chart(fig_speedup, use_container_width=True)

            st.markdown("""
            **Observações**:
            - Speedup > 1.0: Loop é mais rápido
            - Speedup < 1.0: Vetorizado é mais rápido
            - Trade-off: Vetorizado é mais legível e escalável; loop é mais otimizado atualmente
            """)

# ============================================================================
# PÁGINA 5: HISTÓRICO
# ============================================================================

elif page == "📋 Histórico":
    st.title("📋 Histórico de Mudanças")

    st.markdown("""
    Esta página mostraria o histórico de mudanças em:
    - Selic corrente
    - Focus
    - Probabilidades
    
    (Implementar integração com banco de dados ou arquivo CSV)
    """)

    st.info("Funcionalidade em desenvolvimento. Volte em breve!")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    <p>Selic Probability Dashboard | v1.0 | Atualizado em 18/dez/2025</p>
    <p>Dados: BC Brasil, IBGE, Focus Bulletin</p>
    <p>⚠️ Este é um modelo educacional. Não é aconselhamento financeiro.</p>
</div>
""", unsafe_allow_html=True)
