"""
test_benchmark.py
=================
Script de testes unitários simples e benchmark de performance.

Executa:
1. Testes de distribuição (média, std, percentis)
2. Testes de simulador (constraints, terminal distribution)
3. Benchmark vectorizado vs loop
4. Validação de calibração

Autor: Finance Dashboard Team
Data: 2025-12-19
"""

import sys
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from model import create_calibrated_model, MonteCarloSimulator, tilt_distribution
from config import SELIC_STATES, SELIC_MIN, SELIC_MAX, FOCUS_MEDIAN_DEC_2026
from evaluation import brier_score, crps, benchmark_vectorized_vs_loop


def test_tilting():
    """Testa se tilting consegue igualar a média ao target."""
    logger.info("\n=== Test: Tilting Distribution ===")
    
    target = 11.5
    tilted, mean = tilt_distribution(SELIC_STATES, target, tol=1e-3)
    
    assert abs(mean - target) < 0.01, f"Média {mean:.4f} ≠ target {target:.4f}"
    assert sum(tilted.values()) > 0.99, f"Probs não somam 1: {sum(tilted.values())}"
    logger.info(f"✓ Tilting: target={target}, achieved={mean:.4f}")


def test_calibrated_model():
    """Testa modelo calibrado."""
    logger.info("\n=== Test: Calibrated Model ===")
    
    model = create_calibrated_model(focus_median=FOCUS_MEDIAN_DEC_2026)
    stats = model.get_stats()
    
    # Verificar média bate com Focus
    assert abs(stats['mean'] - FOCUS_MEDIAN_DEC_2026) < 0.01, \
        f"Média {stats['mean']} ≠ Focus {FOCUS_MEDIAN_DEC_2026}"
    
    # Verificar percentis em ordem
    assert stats['p10'] <= stats['p25'] <= stats['p50'] <= stats['p75'] <= stats['p90'], \
        "Percentis fora de ordem"
    
    # Verificar limites
    assert stats['min'] >= SELIC_MIN and stats['max'] <= SELIC_MAX, \
        "Stats fora dos limites"
    
    logger.info(f"✓ Model: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
                f"p50={stats['p50']:.3f}")


def test_monte_carlo_vectorized():
    """Testa simulador vetorizado."""
    logger.info("\n=== Test: Monte Carlo Vectorized ===")
    
    n_sims = 5000
    sim = MonteCarloSimulator(n_simulations=n_sims, n_steps=8)
    
    # Simular
    traj, df_term = sim.simulate_vectorized()
    
    # Verificar forma
    assert traj.shape == (n_sims, 9), f"Shape {traj.shape} ≠ (n_sims, 9)"
    
    # Verificar condição inicial
    assert np.allclose(traj[:, 0], 15.0), "Condição inicial não está correta"
    
    # Verificar limites
    assert traj.min() >= SELIC_MIN, f"Min {traj.min()} < SELIC_MIN {SELIC_MIN}"
    assert traj.max() <= SELIC_MAX, f"Max {traj.max()} > SELIC_MAX {SELIC_MAX}"
    
    # Verificar terminal distribution
    terminal = df_term['selic_terminal'].values
    assert len(terminal) == n_sims, f"Terminal size {len(terminal)} ≠ {n_sims}"
    
    logger.info(f"✓ MC Vectorized: mean={terminal.mean():.3f}, std={terminal.std():.3f}, "
                f"min={terminal.min():.3f}, max={terminal.max():.3f}")


def test_monte_carlo_loop():
    """Testa simulador loop (original)."""
    logger.info("\n=== Test: Monte Carlo Loop ===")
    
    n_sims = 5000
    sim = MonteCarloSimulator(n_simulations=n_sims, n_steps=8)
    
    traj, df_term = sim.simulate()
    
    assert traj.shape == (n_sims, 9), f"Shape inválido"
    assert traj.min() >= SELIC_MIN and traj.max() <= SELIC_MAX, "Fora dos limites"
    
    terminal = df_term['selic_terminal'].values
    logger.info(f"✓ MC Loop: mean={terminal.mean():.3f}, std={terminal.std():.3f}")


def test_metrics():
    """Testa métricas Brier e CRPS."""
    logger.info("\n=== Test: Metrics (Brier, CRPS) ===")
    
    # Perfeito
    probs_perfect = np.array([0, 0, 1, 0, 0])
    obs_perfect = np.array([0, 0, 1, 0, 0])
    bs = brier_score(probs_perfect, obs_perfect)
    assert bs == 0.0, f"Brier score perfeito deve ser 0, got {bs}"
    
    # CRPS simples
    quantiles = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    cdf = np.array([0.1, 0.3, 0.6, 0.85, 1.0])
    observed = 12.5
    cr = crps(cdf, quantiles, observed)
    assert cr >= 0, f"CRPS deve ser ≥ 0, got {cr}"
    
    logger.info(f"✓ Metrics: Brier(perfect)={bs:.4f}, CRPS={cr:.4f}")


def main():
    """Executa todos os testes e benchmark."""
    logger.info("=" * 60)
    logger.info("SELIC MODEL - TEST SUITE & BENCHMARK")
    logger.info("=" * 60)
    
    try:
        test_tilting()
        test_calibrated_model()
        test_monte_carlo_loop()
        test_monte_carlo_vectorized()
        test_metrics()
        
        logger.info("\n" + "=" * 60)
        logger.info("Running Benchmark: Vectorized vs Loop")
        logger.info("=" * 60)
        
        bench_df = benchmark_vectorized_vs_loop(n_sims_list=[1000, 5000, 10000])
        print("\n" + bench_df.to_string(index=False))
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ ALL TESTS PASSED")
        logger.info("=" * 60)
        return 0
    except AssertionError as e:
        logger.error(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        logger.error(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
