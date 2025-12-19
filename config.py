"""
config.py
=========
Configurações globais e credenciais para o dashboard de Selic.

Este módulo centraliza todas as constantes, endpoints de API, e credenciais
necessárias para rodar o projeto. Sensível a variáveis de ambiente.

Autor: Finance Dashboard Team
Data: 2025-12-18
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONFIGURAÇÕES GERAIS
# ============================================================================

PROJECT_NAME = "Selic Probability Dashboard"
VERSION = "1.0.0"
TIMEZONE = "America/Sao_Paulo"

# ============================================================================
# ENDPOINTS DE API E SCRAPING
# ============================================================================

# Focus Bulletin (Banco Central do Brasil)
FOCUS_API_URL = "https://www.bcb.gov.br/api/bcdata/datastructure/BM12/data"
FOCUS_PARAMS_SELIC = {
    "lastNObservations": 1,  # Última observação (mais recente)
    "frequency": "D"  # Daily
}

# IPCA (Inflation) - IBGE
IBGE_IPCA_URL = "https://www.ibge.gov.br/api/json/dataset/1737/data"

# Curva DI (B3 - Brazilian Exchange)
# Nota: B3 não expõe API pública fácil; vamos usar scraping ou arquivo local
DI_CURVE_URL = "https://www2.bmf.com.br/pages/portal/bmfbovespa/lumis/lum-precos-futuros-en.asp"
DI_CURVE_FILE = "./data/di_curve.csv"  # caminho local preferencial para curva DI (maturity, rate)

# ============================================================================
# PARÂMETROS DO MODELO DE SELIC
# ============================================================================

# Valores atuais (pode ser atualizado dinamicamente via API)
SELIC_CURRENT = 15.00  # % a.a.
IPCA_12M_CURRENT = 4.46  # % a.a.
FOCUS_MEDIAN_DEC_2026 = 12.13  # % a.a.

# Distribuição de Selic terminal em dez/2026
SELIC_STATES = {
    10.5: 0.04,
    11.0: 0.08,
    11.5: 0.14,
    12.0: 0.22,
    12.5: 0.23,
    13.0: 0.16,
    13.5: 0.08,
    14.0: 0.05,
}

# Limites de taxa
SELIC_MIN = 9.5
SELIC_MAX = 16.0

# Desvio padrão da distribuição (usado em simulações)
SELIC_STD_DEV = 0.60

# ---------------------------------------------------------------------------
# Risco extremo / choques
# Probabilidade de ocorrer um choque exógeno durante o horizonte (por trajetória)
SHOCK_PROB = 0.01
# Magnitude típica do choque em pontos percentuais
SHOCK_MAG = 1.50

# Incerteza de parâmetros: alfa para Dirichlet ao perturbar COPOM_PROBS
PARAM_UNCERTAINTY_ALPHA = 50.0

# Arquivo histórico para backtests (serie mensal/diária com selic realizadA)
HISTORY_FILE = "./data/selic_history.csv"

# ============================================================================
# COPOM MEETING DATES (2026)
# ============================================================================

COPOM_DATES_2026 = [
    "2026-01-27",
    "2026-03-18",
    "2026-05-06",
    "2026-06-17",
    "2026-07-29",
    "2026-08-19",
    "2026-09-16",
    "2026-10-21",
    "2026-12-09",
]

# ============================================================================
# WHATSAPP / NOTIFICAÇÕES
# ============================================================================

# Twilio (para envio de WhatsApp)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")  # sandbox
TWILIO_WHATSAPP_TO = os.getenv("TWILIO_WHATSAPP_TO", "whatsapp:+5511999999999")   # SEU número

# Enable/disable notificações
ENABLE_WHATSAPP_ALERTS = os.getenv("ENABLE_WHATSAPP_ALERTS", "false").lower() == "true"

# ============================================================================
# CACHE E STORAGE
# ============================================================================

CACHE_DIR = "./cache"
DATA_DIR = "./data"
HISTORY_FILE = "./data/selic_history.csv"
PROJECTIONS_FILE = "./data/selic_projections.json"

# TTL para cache (em segundos)
CACHE_TTL_MINUTES = 60

# ============================================================================
# THRESHOLDS DE MUDANÇA (para dispara alertas)
# ============================================================================

# Se mediana de Focus mudar mais que isso, envia WhatsApp
FOCUS_CHANGE_THRESHOLD = 0.0001 # 25 bps

# Se Selic corrente mudar mais que isso
SELIC_CHANGE_THRESHOLD = 0.0001 # 50 bps (tipo, novo corte ou hold para cut)

# Se probabilidade de um estado específico mudar mais que isso
PROB_CHANGE_THRESHOLD = 0.0001 # 5 pontos percentuais

# ============================================================================
# SIMULAÇÃO MONTE CARLO
# ============================================================================

MC_SIMULATIONS = 10_000  # Número de trajetórias a simular
MC_STEPS = 8  # Número de reuniões COPOM

# Probabilidades de ação do COPOM por cenário
COPOM_PROBS = {
    "dovish": {
        "cut_50": 0.30,
        "cut_25": 0.50,
        "hold": 0.20,
    },
    "central": {
        "cut_50": 0.10,
        "cut_25": 0.70,
        "hold": 0.20,
    },
    "hawkish": {
        "cut_50": 0.05,
        "cut_25": 0.35,
        "hold": 0.60,
    },
}

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = "./logs/selic_dashboard.log"
