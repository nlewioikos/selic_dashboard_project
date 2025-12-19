"""
data_fetcher.py
===============
Módulo responsável por coletar dados em tempo real de:
- Taxa Selic corrente (BC Brasil)
- IPCA (Inflação - IBGE)
- Focus Bulletin (Expectativas de mercado - BC)
- Histórico de decisões do COPOM

Este módulo implementa cache automático para evitar sobrecarga de APIs.
Todos os valores são atualizados automaticamente quando stale.

Autor: Finance Dashboard Team
Data: 2025-12-18
"""

import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
import json
import os
from typing import Dict, Optional, Tuple
from config import (
    FOCUS_API_URL,
    IBGE_IPCA_URL,
    CACHE_DIR,
    CACHE_TTL_MINUTES,
    SELIC_CURRENT,
    IPCA_12M_CURRENT,
    FOCUS_MEDIAN_DEC_2026,
    HISTORY_FILE,
    DI_CURVE_FILE,
    DI_CURVE_URL,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Criar diretórios se não existirem
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)


class DataFetcher:
    """
    Responsável por coletar e cachear dados de taxa Selic, IPCA e Focus.
    
    Attributes:
        cache_dir (str): Diretório para armazenar cache
        cache_ttl (int): Tempo de vida do cache em minutos
    """

    def __init__(self, cache_dir: str = CACHE_DIR, cache_ttl_minutes: int = CACHE_TTL_MINUTES):
        """
        Inicializa o fetcher.

        Args:
            cache_dir: Diretório de cache
            cache_ttl_minutes: TTL do cache em minutos
        """
        self.cache_dir = cache_dir
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self.selic_current = SELIC_CURRENT
        self.ipca_12m = IPCA_12M_CURRENT
        self.focus_median = FOCUS_MEDIAN_DEC_2026

    def _get_cache_path(self, key: str) -> str:
        """Retorna o caminho de cache para uma chave."""
        return os.path.join(self.cache_dir, f"{key}_cache.json")

    def _is_cache_valid(self, key: str) -> bool:
        """Verifica se cache é válido (não expirado)."""
        cache_path = self._get_cache_path(key)
        if not os.path.exists(cache_path):
            return False

        mtime = os.path.getmtime(cache_path)
        age = datetime.now() - datetime.fromtimestamp(mtime)
        return age < self.cache_ttl

    def _load_cache(self, key: str) -> Optional[Dict]:
        """Carrega dados do cache."""
        try:
            with open(self._get_cache_path(key), "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Erro ao carregar cache {key}: {e}")
            return None

    def _save_cache(self, key: str, data: Dict):
        """Salva dados no cache."""
        try:
            with open(self._get_cache_path(key), "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Erro ao salvar cache {key}: {e}")

    def fetch_selic_current(self) -> Tuple[float, datetime]:
        """
        Busca taxa Selic corrente do BC Brasil.
        
        Returns:
            Tuple[float, datetime]: (taxa_selic, data_da_decisão)
        """
        cache_key = "selic_current"

        if self._is_cache_valid(cache_key):
            data = self._load_cache(cache_key)
            if data:
                logger.info(f"[CACHE] Selic atual: {data['value']}%")
                return float(data["value"]), datetime.fromisoformat(data["date"])

        try:
            # BC Brasil API retorna JSON com estrutura específica
            url = "https://www.bcb.gov.br/api/bcdata/datastructure/BM12/data"
            params = {"lastNObservations": 1, "frequency": "D"}

            resp = requests.get(url, params=params, timeout=5)
            resp.raise_for_status()

            data = resp.json()
            if data and "observations" in data:
                obs = data["observations"][0]
                selic_val = float(obs["value"])
                date_str = obs["date"]

                cache_data = {"value": selic_val, "date": date_str}
                self._save_cache(cache_key, cache_data)

                logger.info(f"[API] Selic atualizada: {selic_val}%")
                return selic_val, datetime.fromisoformat(date_str)
        except Exception as e:
            logger.error(f"Erro ao buscar Selic: {e}")

        # Fallback para valor local
        logger.info(f"[FALLBACK] Usando Selic local: {self.selic_current}%")
        return self.selic_current, datetime.now()

    def fetch_ipca_12m(self) -> Tuple[float, datetime]:
        """
        Busca IPCA 12 meses acumulado do IBGE.
        
        Returns:
            Tuple[float, datetime]: (ipca_12m, data_da_leitura)
        """
        cache_key = "ipca_12m"

        if self._is_cache_valid(cache_key):
            data = self._load_cache(cache_key)
            if data:
                logger.info(f"[CACHE] IPCA 12m: {data['value']}%")
                return float(data["value"]), datetime.fromisoformat(data["date"])

        try:
            # IBGE API
            url = "https://www.ibge.gov.br/api/json/dataset/1737/data"
            params = {"last": 1}

            resp = requests.get(url, params=params, timeout=5)
            resp.raise_for_status()

            data = resp.json()
            if data and "resultado" in data:
                # IBGE retorna em formato específico
                serie = data["resultado"][0]["serie"]
                last_obs = list(serie.values())[-1]
                ipca_val = float(last_obs)

                cache_data = {"value": ipca_val, "date": datetime.now().isoformat()}
                self._save_cache(cache_key, cache_data)

                logger.info(f"[API] IPCA 12m atualizado: {ipca_val}%")
                return ipca_val, datetime.now()
        except Exception as e:
            logger.error(f"Erro ao buscar IPCA: {e}")

        # Fallback
        logger.info(f"[FALLBACK] Usando IPCA local: {self.ipca_12m}%")
        return self.ipca_12m, datetime.now()

    def fetch_focus_selic_2026(self) -> Tuple[float, float, datetime]:
        """
        Busca Focus Bulletin para Selic em dez/2026 (mediana e desvio).
        
        Returns:
            Tuple[float, float, datetime]: (mediana_selic, desvio, data)
        """
        cache_key = "focus_selic_2026"

        if self._is_cache_valid(cache_key):
            data = self._load_cache(cache_key)
            if data:
                logger.info(f"[CACHE] Focus Selic 2026: {data['median']}% ±{data['std']}")
                return (
                    float(data["median"]),
                    float(data["std"]),
                    datetime.fromisoformat(data["date"]),
                )

        try:
            # Focus Bulletin (BC Brasil publica em CSV/JSON)
            url = "https://www.bcb.gov.br/api/bcdata/datastructure/Focus/data"
            params = {
                "lastNObservations": 1,
                "filter[indicador][eq]": "Selic",
                "filter[dataInicio][gte]": (datetime.now() - timedelta(days=7)).isoformat(),
            }

            resp = requests.get(url, params=params, timeout=5)
            resp.raise_for_status()

            data = resp.json()
            if data and "observations" in data:
                obs = data["observations"][0]
                median_val = float(obs["median"])
                std_val = float(obs.get("std_dev", 0.6))

                cache_data = {
                    "median": median_val,
                    "std": std_val,
                    "date": datetime.now().isoformat(),
                }
                self._save_cache(cache_key, cache_data)

                logger.info(f"[API] Focus Selic 2026 atualizado: {median_val}%")
                return median_val, std_val, datetime.now()
        except Exception as e:
            logger.error(f"Erro ao buscar Focus: {e}")

        # Fallback
        logger.info(f"[FALLBACK] Usando Focus local: {self.focus_median}%")
        return self.focus_median, 0.60, datetime.now()

    def get_all_current_data(self) -> Dict:
        """
        Retorna todos os dados correntes em um dicionário.
        
        Returns:
            Dict com chaves: selic, ipca_12m, focus_selic, focus_std, last_update
        """
        selic, selic_date = self.fetch_selic_current()
        ipca, ipca_date = self.fetch_ipca_12m()
        focus_median, focus_std, focus_date = self.fetch_focus_selic_2026()

        return {
            "selic": selic,
            "ipca_12m": ipca,
            "focus_selic": focus_median,
            "focus_std": focus_std,
            "last_update": max(selic_date, ipca_date, focus_date),
            "selic_date": selic_date,
            "ipca_date": ipca_date,
            "focus_date": focus_date,
        }

    def fetch_di_curve(self) -> Optional[pd.DataFrame]:
        """
        Tenta carregar a curva DI de um arquivo local (`DI_CURVE_FILE`).
        Se não existir, tenta fazer um fetch simples da URL configurada (pode falhar).

        Retorna DataFrame com colunas: ['maturity', 'rate'] (maturidade em anos, taxa %).
        """
        cache_key = "di_curve"

        # Primeiro, tentar arquivo local
        try:
            if os.path.exists(DI_CURVE_FILE):
                df = pd.read_csv(DI_CURVE_FILE)
                logger.info(f"[LOCAL] Curva DI carregada de {DI_CURVE_FILE}")
                return df
        except Exception as e:
            logger.warning(f"Erro ao ler DI_CURVE_FILE {DI_CURVE_FILE}: {e}")

        # Em segundo lugar, tentar fetch simples (placeholder scraping)
        try:
            resp = requests.get(DI_CURVE_URL, timeout=5)
            resp.raise_for_status()
            # Parsing HTML table robustamente é fora do escopo; apenas logar e retornar None
            logger.info("DI curve URL acessada, mas parsing não implementado. Forneça data/di_curve.csv.")
        except Exception as e:
            logger.warning(f"Não foi possível buscar DI curve: {e}")

        return None


# Singleton global
_fetcher = None


def get_fetcher() -> DataFetcher:
    """Retorna instância singleton do fetcher."""
    global _fetcher
    if _fetcher is None:
        _fetcher = DataFetcher()
    return _fetcher
