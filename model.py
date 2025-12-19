"""
model.py
========
Modelo estatístico de probabilidade de trajetória da Selic em 2026.

Implementa:
- Distribuição discreta de Selic terminal (dez/2026)
- Simulação Monte Carlo de trajetórias
- Calibração automática baseada em Focus/DI

Toda a lógica é documentada com referências à metodologia
(ver selic_model_methodology.md).

Autor: Finance Dashboard Team
Data: 2025-12-18
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List

# Try to import numba for JIT compilation
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(func):
        """Fallback: no-op decorator if numba not available."""
        return func

from config import (
    SELIC_STATES,
    SELIC_CURRENT,
    FOCUS_MEDIAN_DEC_2026,
    SELIC_STD_DEV,
    MC_SIMULATIONS,
    MC_STEPS,
    COPOM_PROBS,
    COPOM_DATES_2026,
    SELIC_MIN,
    SELIC_MAX,
    SHOCK_PROB,
    SHOCK_MAG,
    PARAM_UNCERTAINTY_ALPHA,
    DI_CURVE_FILE,
)
from data_fetcher import get_fetcher

logger = logging.getLogger(__name__)


class SelicProbabilityModel:
    """
    Modelo de distribuição de probabilidade para Selic terminal (dez/2026).
    
    Calibrado em:
    - Focus Bulletin (mediana de mercado)
    - Curva DI (expectativas implícitas de juros)
    - Restrições de dinâmica do COPOM (corte gradual, não violento)
    
    Attributes:
        states (Dict): Mapa de {selic_level: probability}
        focus_median (float): Mediana do Focus para Selic em dez/2026
        selic_std (float): Desvio padrão calibrado
    """

    def __init__(
        self,
        focus_median: float = FOCUS_MEDIAN_DEC_2026,
        selic_std: float = SELIC_STD_DEV,
        states: Dict[float, float] = None,
    ):
        """
        Inicializa o modelo.

        Args:
            focus_median: Mediana do Focus para Selic terminal
            selic_std: Desvio padrão da distribuição
            states: Dict {selic_level: probability} ou None para usar default
        """
        self.focus_median = focus_median
        self.selic_std = selic_std
        self.states = states or SELIC_STATES

        # Validar que soma a 1
        total_prob = sum(self.states.values())
        if abs(total_prob - 1.0) > 0.01:
            logger.warning(f"Probabilidades não somam 1: {total_prob}")
            # Normalizar
            self.states = {k: v / total_prob for k, v in self.states.items()}

    def get_probabilities(self) -> pd.DataFrame:
        """
        Retorna DataFrame com distribuição de probabilidades.

        Returns:
            DataFrame com colunas: selic_level, probability, cumulative
        """
        df = pd.DataFrame(
            list(self.states.items()),
            columns=["selic_level", "probability"]
        ).sort_values("selic_level")

        df["cumulative"] = df["probability"].cumsum()
        df["pct_display"] = (df["probability"] * 100).round(2)

        return df

    def expected_value(self) -> float:
        """
        Calcula esperança (média) da distribuição.

        Returns:
            float: E[Selic] em %
        """
        return sum(level * prob for level, prob in self.states.items())

    def probability_range(self, lower: float, upper: float) -> float:
        """
        Calcula probabilidade de Selic estar em intervalo [lower, upper].

        Args:
            lower: Limite inferior (%)
            upper: Limite superior (%)

        Returns:
            float: P(lower <= Selic <= upper)
        """
        return sum(
            prob for level, prob in self.states.items()
            if lower <= level <= upper
        )

    def percentile(self, p: float) -> float:
        """
        Retorna p-ésimo percentil da distribuição.

        Args:
            p: Percentil (0-100)

        Returns:
            float: Valor de Selic no percentil p
        """
        df = self.get_probabilities()
        idx = np.searchsorted(df["cumulative"].values, p / 100.0)
        return df.iloc[min(idx, len(df) - 1)]["selic_level"]

    def get_stats(self) -> Dict:
        """
        Retorna estatísticas resumidas da distribuição.

        Returns:
            Dict com mean, std, p10, p25, p50, p75, p90
        """
        df = self.get_probabilities()

        mean = self.expected_value()
        variance = sum(
            (level - mean) ** 2 * prob
            for level, prob in self.states.items()
        )
        std = np.sqrt(variance)

        return {
            "mean": round(mean, 3),
            "std": round(std, 3),
            "min": df["selic_level"].min(),
            "max": df["selic_level"].max(),
            "p10": round(self.percentile(10), 3),
            "p25": round(self.percentile(25), 3),
            "p50": round(self.percentile(50), 3),  # Mediana
            "p75": round(self.percentile(75), 3),
            "p90": round(self.percentile(90), 3),
        }

    def interpret_distribution(self) -> Dict[str, Tuple[float, str]]:
        """
        Interpreta a distribuição em cenários nomeados.

        Returns:
            Dict com cenários: {nome: (probabilidade, descrição)}
        """
        # Use interval partitioning without overlap to ensure probabilities sum to ~1.0
        # dovish: [10.5, 11.5) ; central: [11.5, 12.5) ; hawkish: [12.5, 14.0]
        dovish_prob = sum(
            prob for level, prob in self.states.items() if 10.5 <= level < 11.5
        )
        central_prob = sum(
            prob for level, prob in self.states.items() if 11.5 <= level < 12.5
        )
        hawkish_prob = sum(
            prob for level, prob in self.states.items() if 12.5 <= level <= 14.0
        )

        return {
            "Dovish (corte agressivo)": (
                dovish_prob,
                "BC corta forte: IPCA desinflaciona, atividade fraca"
            ),
            "Central (corte gradual)": (
                central_prob,
                "Cenário base: cortes suaves até ~12% conforme Focus"
            ),
            "Hawkish (hold conservador)": (
                hawkish_prob,
                "Fiscal/câmbio pressionam: BC corta menos, termina acima de 12,5%"
            ),
        }


def tilt_distribution(states_dict: Dict[float, float], target_mean: float, tol: float = 1e-4, max_iter: int = 100) -> Tuple[Dict[float, float], float]:
    """
    Exponential tilting (Esscher) to shift discrete distribution mean to target_mean.

    Returns (new_states_dict, achieved_mean).
    """
    levels = np.array(sorted(states_dict.keys()))
    prior = np.array([states_dict[l] for l in levels], dtype=float)

    def safe_exp(x):
        return np.exp(np.clip(x, -700, 700))

    def compute_mean(lmbda: float):
        ex = safe_exp(lmbda * levels)
        weights = prior * ex
        s = weights.sum()
        if s == 0 or not np.isfinite(s):
            return float('nan'), np.zeros_like(prior)
        probs = weights / s
        return (probs * levels).sum(), probs

    low, high = -20.0, 20.0
    mean_low, _ = compute_mean(low)
    mean_high, _ = compute_mean(high)

    expand_limit = 200.0
    step = 20.0
    while (np.isnan(mean_low) or np.isnan(mean_high) or not (min(mean_low, mean_high) <= target_mean <= max(mean_low, mean_high))) and (abs(low) < expand_limit and abs(high) < expand_limit):
        low -= step
        high += step
        mean_low, _ = compute_mean(low)
        mean_high, _ = compute_mean(high)

    if np.isnan(mean_low) or np.isnan(mean_high):
        final_probs = prior / prior.sum()
        return dict(zip(levels.tolist(), final_probs.tolist())), (final_probs * levels).sum()

    if not (min(mean_low, mean_high) <= target_mean <= max(mean_low, mean_high)):
        candidates = [low, 0.0, high]
        best = min(candidates, key=lambda x: abs(compute_mean(x)[0] - target_mean))
        final_mean, final_probs = compute_mean(best)
        final_probs = final_probs / final_probs.sum()
        return dict(zip(levels.tolist(), final_probs.tolist())), final_mean

    mid = 0.0
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        mean_mid, probs_mid = compute_mean(mid)
        if np.isnan(mean_mid):
            low = (low + mid) / 2.0
            high = (high + mid) / 2.0
            continue
        if abs(mean_mid - target_mean) < tol:
            final_probs = probs_mid / probs_mid.sum()
            return dict(zip(levels.tolist(), final_probs.tolist())), mean_mid
        if mean_mid < target_mean:
            low = mid
        else:
            high = mid

    final_probs = probs_mid / probs_mid.sum()
    return dict(zip(levels.tolist(), final_probs.tolist())), mean_mid


class MonteCarloSimulator:
    """
    Simulador Monte Carlo de trajetória de Selic de jan a dez/2026.
    
    Simula múltiplas trajetórias de decisões do COPOM,
    cada uma aleatória mas consistente com restrições macroeconômicas.
    
    Attributes:
        n_simulations: Número de trajetórias a simular
        n_steps: Número de reuniões COPOM
        copom_dates: Datas das reuniões
    """

    def __init__(
        self,
        n_simulations: int = MC_SIMULATIONS,
        n_steps: int = MC_STEPS,
        copom_dates: List[str] = None,
    ):
        """
        Inicializa o simulador.

        Args:
            n_simulations: Quantas trajetórias
            n_steps: Quantas reuniões COPOM
            copom_dates: Datas das reuniões (ou None para usar padrão)
        """
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.copom_dates = copom_dates or COPOM_DATES_2026

        if len(self.copom_dates) < n_steps:
            logger.warning(f"Poucos COPOM dates: {len(self.copom_dates)} < {n_steps}")

    def _choose_action(self, scenario: str) -> float:
        """
        Sorteia ação do COPOM (cut_50, cut_25, hold) baseado em cenário.

        Args:
            scenario: 'dovish', 'central', ou 'hawkish'

        Returns:
            float: Mudança de taxa em pontos percentuais
        """
        probs = COPOM_PROBS[scenario]
        rand = np.random.random()

        cumsum = 0
        for action, prob in probs.items():
            cumsum += prob
            if rand < cumsum:
                if action == "cut_50":
                    return -0.50
                elif action == "cut_25":
                    return -0.25
                else:  # hold
                    return 0.0

        return 0.0  # Fallback

    def simulate(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Executa simulação Monte Carlo completa.

        Returns:
            Tuple[array_trajetórias, df_resumo_terminal]
            - array: shape (n_simulations, n_steps) com trajetória de Selic
            - df: resumo com distribuição terminal
        """
        # Array para armazenar trajetórias
        trajectories = np.zeros((self.n_simulations, self.n_steps + 1))
        trajectories[:, 0] = SELIC_CURRENT  # Condição inicial

        for sim in range(self.n_simulations):
            # Sorteia cenário para essa trajetória (baseado em probs de COPOM)
            scenario_rand = np.random.random()
            if scenario_rand < 0.30:
                scenario = "dovish"
            elif scenario_rand < 0.80:
                scenario = "central"
            else:
                scenario = "hawkish"

            # Simula cada reunião COPOM
            for step in range(1, self.n_steps + 1):
                action = self._choose_action(scenario)

                # Aplica mudança
                new_rate = trajectories[sim, step - 1] + action

                # Restrições: não sai dos limites
                new_rate = np.clip(new_rate, SELIC_MIN, SELIC_MAX)

                trajectories[sim, step] = new_rate

        # Distribução terminal
        terminal_rates = trajectories[:, -1]

        df_terminal = pd.DataFrame({
            "selic_terminal": terminal_rates
        })

        # Agregar por níveis de 0,25%
        bins = np.arange(SELIC_MIN - 0.25, SELIC_MAX + 0.5, 0.25)
        df_terminal["bin"] = pd.cut(terminal_rates, bins=bins)

        dist = df_terminal.groupby("bin", observed=True).size() / self.n_simulations

        logger.info(f"MC: {self.n_simulations} simulações completadas")

        return trajectories, df_terminal

    def simulate_vectorized(
        self,
        scenario_probs: Tuple[float, float, float] = (0.30, 0.50, 0.20),
        copom_probs_base: Dict = COPOM_PROBS,
        param_uncertainty_alpha: float = PARAM_UNCERTAINTY_ALPHA,
        shock_prob: float = SHOCK_PROB,
        shock_mag: float = SHOCK_MAG,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Versão vetorizada do simulador Monte Carlo. Suporta incerteza de parâmetros
        via Dirichlet e choques exógenos.

        Returns:
            (trajectories, df_terminal)
        """
        n = self.n_simulations
        steps = self.n_steps

        trajectories = np.zeros((n, steps + 1))
        trajectories[:, 0] = SELIC_CURRENT

        # Cenários por simulação
        scenario_choices = np.random.choice(
            ['dovish', 'central', 'hawkish'], size=n, p=list(scenario_probs)
        )
        scenario_map = {'dovish': 0, 'central': 1, 'hawkish': 2}
        scenario_idx = np.array([scenario_map[s] for s in scenario_choices], dtype=int)

        # Preparar probabilidades base em ordem [cut_50, cut_25, hold]
        actions_order = ['cut_50', 'cut_25', 'hold']
        base_matrix = np.zeros((3, 3), dtype=float)  # scenarios x actions
        sc_names = ['dovish', 'central', 'hawkish']
        for i, sc in enumerate(sc_names):
            probs = copom_probs_base.get(sc, {})
            base_matrix[i, :] = [probs.get(a, 0.0) for a in actions_order]

        # Perturbar probabilidades por simulação (Dirichlet around base)
        if param_uncertainty_alpha and param_uncertainty_alpha > 0:
            # construct alpha vectors for each scenario scaled
            alpha_matrix = base_matrix * param_uncertainty_alpha
            # ensure no zero rows
            alpha_matrix = np.where(alpha_matrix <= 0, 1e-3, alpha_matrix)
            # For each simulation, sample a perturbed probs corresponding to its scenario
            perturbed_probs = np.zeros((n, 3))
            for i in range(n):
                sidx = scenario_idx[i]
                a = alpha_matrix[sidx]
                sample = np.random.default_rng().random(len(a))
                # use Dirichlet via gamma sampling
                gam = np.random.gamma(a, 1.0)
                gam = np.where(gam <= 0, 1e-8, gam)
                perturbed_probs[i, :] = gam / gam.sum()
        else:
            # Use base prob corresponding to scenario
            perturbed_probs = base_matrix[scenario_idx]

        # Pre-generate shocks (n x steps)
        shock_draws = np.random.random((n, steps))
        shock_flags = shock_draws < shock_prob
        shock_signs = np.where(np.random.random((n, steps)) < 0.5, -1.0, 1.0)
        shock_values = shock_flags * shock_signs * shock_mag

        # Actions mapping
        action_values = np.array([-0.50, -0.25, 0.0], dtype=float)

        # Simular todos os steps vetorizadamente
        for step in range(1, steps + 1):
            u = np.random.random(n)
            # cumulative sum along actions
            cums = np.cumsum(perturbed_probs, axis=1)
            # determine action index where u < cumsum
            # broadcasting comparison
            less = u[:, None] < cums
            # argmax of first True gives action idx
            acts = less.argmax(axis=1)

            # aplicar ação e choque
            actions_applied = action_values[acts]
            trajectories[:, step] = trajectories[:, step - 1] + actions_applied + shock_values[:, step - 1]
            # limites
            trajectories[:, step] = np.clip(trajectories[:, step], SELIC_MIN, SELIC_MAX)

        terminal_rates = trajectories[:, -1]
        df_terminal = pd.DataFrame({
            "selic_terminal": terminal_rates
        })

        logger.info(f"MC vetorizado: {n} simulações completadas (steps={steps})")
        return trajectories, df_terminal

    def get_distribution_from_simulations(
        self, trajectories: np.ndarray
    ) -> pd.DataFrame:
        """
        Extrai distribuição de probabilidade a partir de simulações.

        Args:
            trajectories: Array de trajetórias

        Returns:
            DataFrame com probabilidades em cada nível de Selic
        """
        terminal_rates = trajectories[:, -1]

        # Agrupar em bins de 0,5%
        bins = np.arange(SELIC_MIN - 0.5, SELIC_MAX + 1, 0.5)
        counts, bin_edges = np.histogram(terminal_rates, bins=bins)

        probs = counts / self.n_simulations
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        df = pd.DataFrame({
            "selic_level": bin_centers,
            "probability_mc": probs,
        })

        return df.sort_values("selic_level")


def create_calibrated_model(
    focus_median: float = FOCUS_MEDIAN_DEC_2026,
    selic_current: float = SELIC_CURRENT,
) -> SelicProbabilityModel:
    """
    Factory function para criar modelo calibrado automaticamente.

    Args:
        focus_median: Mediana do Focus
        selic_current: Selic corrente

    Returns:
        SelicProbabilityModel instanciado e pronto
    """
    # Tentar calibrar à curva DI local (se disponível) e depois ao Focus
    di_df = None
    try:
        fetcher = get_fetcher()
        di_df = fetcher.fetch_di_curve()
    except Exception:
        di_df = None

    target_mean = focus_median
    if di_df is not None and {'maturity', 'rate'}.issubset(set(di_df.columns)):
        try:
            from datetime import datetime
            target_date = datetime(2026, 12, 31)
            years = (target_date - datetime.now()).days / 365.25
            implied = np.interp(years, di_df['maturity'].astype(float).values, di_df['rate'].astype(float).values)
            target_mean = float(implied)
            logger.info(f"DI curve presente: implicita para dez/2026 = {target_mean:.3f}% (usada na calibração)")
        except Exception as e:
            logger.warning(f"Erro ao interpolar DI curve: {e}; usando Focus para calibração.")

    tilted_states, achieved_mean = tilt_distribution(SELIC_STATES, target_mean)

    levels = np.array(sorted(tilted_states.keys()))
    probs = np.array([tilted_states[l] for l in levels], dtype=float)
    variance = ((levels - achieved_mean) ** 2 * probs).sum()
    calibrated_std = float(np.sqrt(variance))

    logger.info(f"Calibração completa: target={target_mean:.3f}%, achieved_mean={achieved_mean:.3f}% , std={calibrated_std:.3f}%")

    model = SelicProbabilityModel(
        focus_median=focus_median,
        selic_std=calibrated_std,
        states=tilted_states,
    )

    return model
