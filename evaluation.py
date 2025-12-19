"""
evaluation.py
==============
Módulo de avaliação e backtest do modelo de Selic.

Inclui:
- Funções para decisões condicionais do COPOM (baseadas em gaps macroeconômicos)
- Métricas de calibração (Brier score, CRPS)
- Backtest histórico
- Benchmark de performance

Autor: Finance Dashboard Team
Data: 2025-12-19
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional, Callable
from config import COPOM_PROBS

logger = logging.getLogger(__name__)


def get_conditional_copom_probs(
    selic: float,
    ipca_12m: float,
    selic_target: float = 11.0,
    ipca_target: float = 4.5,
) -> Dict[str, Dict[str, float]]:
    """
    Retorna probabilidades de ação do COPOM condicionadas aos gaps macroeconômicos.

    Lógica simples:
    - ipca_gap = ipca_12m - ipca_target
    - selic_gap = selic - selic_target
    
    Se ipca_gap > 1.0: COPOM mais conservador (mais hold)
    Se ipca_gap < -0.5: COPOM mais agressivo (mais cut)
    
    Args:
        selic: Taxa Selic corrente (%)
        ipca_12m: IPCA 12 meses (%)
        selic_target: Meta de Selic do BC (%)
        ipca_target: Meta de IPCA do BC (%)

    Returns:
        Dict com estrutura {scenario: {action: prob}} customizado por gaps
    """
    ipca_gap = ipca_12m - ipca_target
    selic_gap = selic - selic_target

    # Começar com base
    probs = {k: v.copy() for k, v in COPOM_PROBS.items()}

    # Ajustar dovish (reduz hold, aumenta cut)
    if ipca_gap < -0.5:
        probs["dovish"]["cut_50"] = min(0.40, probs["dovish"]["cut_50"] + 0.1)
        probs["dovish"]["cut_25"] = max(0.40, probs["dovish"]["cut_25"] - 0.05)
        probs["dovish"]["hold"] = max(0.05, probs["dovish"]["hold"] - 0.05)

    # Ajustar hawkish (aumenta hold, reduz cut) se inflação pressionar
    if ipca_gap > 1.5:
        probs["hawkish"]["hold"] = min(0.80, probs["hawkish"]["hold"] + 0.15)
        probs["hawkish"]["cut_25"] = max(0.15, probs["hawkish"]["cut_25"] - 0.1)
        probs["hawkish"]["cut_50"] = max(0.0, probs["hawkish"]["cut_50"] - 0.05)

    # Normalizar
    for scenario in probs:
        s = sum(probs[scenario].values())
        if s > 0:
            probs[scenario] = {k: v / s for k, v in probs[scenario].items()}

    logger.info(f"Probs condicionais: ipca_gap={ipca_gap:.2f}, selic_gap={selic_gap:.2f}")
    return probs


def brier_score(probabilities: np.ndarray, observations: np.ndarray) -> float:
    """
    Calcula Brier Score para avaliação de probabilidades.
    
    BS = mean((p - o)^2) onde p é probabilidade prevista, o é observação (0 ou 1).

    Args:
        probabilities: Array de probabilidades [0, 1]
        observations: Array de observações binárias (0 ou 1)

    Returns:
        Brier Score (menor é melhor; mínimo 0)
    """
    return float(np.mean((probabilities - observations) ** 2))


def crps(quantile_probs: np.ndarray, quantiles: np.ndarray, observed: float) -> float:
    """
    Calcula CRPS (Continuous Ranked Probability Score) para distribuição discreta.

    Aproximação: CRPS ≈ sum_i |F(q_i) - H(q_i - observed)| * Δq_i
    onde F é CDF teórica (quantile_probs) e H é Heaviside.

    Args:
        quantile_probs: Probabilidades cumulativas nos quantis
        quantiles: Valores dos quantis (em ordem)
        observed: Valor observado

    Returns:
        CRPS (menor é melhor)
    """
    if len(quantiles) != len(quantile_probs):
        raise ValueError("quantiles e quantile_probs devem ter mesmo tamanho")

    cdf_theor = quantile_probs
    heaviside = (quantiles >= observed).astype(float)

    # Diferenças entre quantis consecutivos
    dq = np.diff(np.concatenate([[-np.inf], quantiles, [np.inf]]))

    score = np.sum(np.abs(cdf_theor - heaviside) * dq[:-1])
    return float(score)


def backtest_simple(
    df_history: pd.DataFrame,
    model_factory: Callable,
    date_col: str = "date",
    selic_col: str = "selic",
    ipca_col: str = "ipca_12m",
) -> Dict:
    """
    Realiza backtest simples aplicando o modelo a pontos históricos.

    Args:
        df_history: DataFrame com coluna date, selic, ipca_12m, selic_realized
        model_factory: Função que retorna modelo calibrado (ex: create_calibrated_model)
        date_col: Nome da coluna de data
        selic_col: Nome da coluna de Selic corrente
        ipca_col: Nome da coluna de IPCA 12m

    Returns:
        Dict com métricas de backtest
    """
    if df_history.empty or 'selic_realized' not in df_history.columns:
        logger.warning("Histórico vazio ou sem coluna selic_realized; backtest retorna None")
        return {}

    brier_scores = []
    crps_scores = []
    predictions = []

    for idx, row in df_history.iterrows():
        try:
            # Calibrar modelo com dados naquele momento
            focus_median = row.get(selic_col, 12.0)  # usar selic como proxy para focus
            model = model_factory(focus_median=focus_median)

            stats = model.get_stats()
            probs_df = model.get_probabilities()

            realized = row.get('selic_realized', np.nan)
            if np.isnan(realized):
                continue

            # Brier score: distância até 0/1 (evento ocorreu ou não no bucket)
            realized_level = round(realized * 2) / 2  # arredondar para nível mais próximo
            bs = brier_score(
                np.array([1.0 if level == realized_level else 0.0
                          for level in probs_df['selic_level'].values]),
                np.array([1.0 if level == realized_level else 0.0
                          for level in probs_df['selic_level'].values]),
            )
            brier_scores.append(bs)

            # CRPS
            cr = crps(
                probs_df['cumulative'].values,
                probs_df['selic_level'].values,
                realized,
            )
            crps_scores.append(cr)

            predictions.append({
                'date': row.get(date_col),
                'brier': bs,
                'crps': cr,
                'realized': realized,
                'mean': stats['mean'],
                'p50': stats['p50'],
            })
        except Exception as e:
            logger.warning(f"Erro no backtest para idx {idx}: {e}")
            continue

    result = {
        'n_points': len(predictions),
        'mean_brier': float(np.mean(brier_scores)) if brier_scores else np.nan,
        'mean_crps': float(np.mean(crps_scores)) if crps_scores else np.nan,
        'std_brier': float(np.std(brier_scores)) if brier_scores else np.nan,
        'std_crps': float(np.std(crps_scores)) if crps_scores else np.nan,
        'predictions': predictions,
    }

    logger.info(f"Backtest completo: {result['n_points']} pontos, "
                f"Brier={result['mean_brier']:.4f}, CRPS={result['mean_crps']:.4f}")

    return result


def benchmark_vectorized_vs_loop(
    n_sims_list: list = [1000, 10000, 100000],
) -> pd.DataFrame:
    """
    Benchmark comparando simulador vetorizado vs loop (se disponível).

    Args:
        n_sims_list: Lista de tamanhos de simulação para testar

    Returns:
        DataFrame com tempos (ms) para cada método e tamanho
    """
    import time
    from model import MonteCarloSimulator

    results = []

    for n_sims in n_sims_list:
        sim = MonteCarloSimulator(n_simulations=n_sims, n_steps=8)

        # Teste versão vetorizada
        t0 = time.time()
        traj_vec, _ = sim.simulate_vectorized()
        t_vec = (time.time() - t0) * 1000  # ms

        # Teste versão loop (original)
        t0 = time.time()
        traj_loop, _ = sim.simulate()
        t_loop = (time.time() - t0) * 1000  # ms

        results.append({
            'n_simulations': n_sims,
            'time_vectorized_ms': t_vec,
            'time_loop_ms': t_loop,
            'speedup': t_loop / t_vec if t_vec > 0 else np.inf,
        })

        logger.info(f"n_sims={n_sims}: vec={t_vec:.2f}ms, loop={t_loop:.2f}ms, "
                    f"speedup={t_loop/t_vec if t_vec > 0 else 'inf':.2f}x")

    return pd.DataFrame(results)
