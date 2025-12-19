"""
quickstart.py
=============
Script de inicialização rápida para testar o modelo.

Executa:
1. Carregamento de dados
2. Calibração do modelo
3. Simulação MC simples
4. Exibição de resultados

Uso: python quickstart.py
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from model import create_calibrated_model, MonteCarloSimulator
from data_fetcher import get_fetcher
from evaluation import get_conditional_copom_probs
import pandas as pd

print("\n" + "=" * 70)
print("🎯 SELIC MODEL - QUICKSTART")
print("=" * 70)

# 1. Carregar dados
print("\n📥 Carregando dados...")
fetcher = get_fetcher()
data = fetcher.get_all_current_data()
print(f"  ✓ Selic: {data['selic']:.2f}%")
print(f"  ✓ Focus 2026: {data['focus_selic']:.2f}%")
print(f"  ✓ IPCA 12m: {data['ipca_12m']:.2f}%")

# 2. Calibrar modelo
print("\n🔧 Calibrando modelo...")
model = create_calibrated_model(focus_median=data['focus_selic'])
stats = model.get_stats()
print(f"  ✓ Média: {stats['mean']:.2f}%")
print(f"  ✓ Desvio Padrão: {stats['std']:.2f}%")
print(f"  ✓ P50 (mediana): {stats['p50']:.2f}%")
print(f"  ✓ P10-P90: {stats['p10']:.2f}% - {stats['p90']:.2f}%")

# 3. Distribuição
print("\n📊 Distribuição de Probabilidade:")
df_probs = model.get_probabilities()
for _, row in df_probs.iterrows():
    pct = row['pct_display']
    bar = "█" * max(1, int(pct / 2))
    print(f"  {row['selic_level']:>5.1f}%: {bar} {pct:>5.1f}%")

# 4. Cenários
print("\n🎭 Cenários Macroeconômicos:")
scenarios = model.interpret_distribution()
for name, (prob, desc) in scenarios.items():
    print(f"  {name}: {prob*100:>5.1f}%")

# 5. Simulação MC básica
print("\n🎲 Executando Monte Carlo (5000 simulações)...")
sim = MonteCarloSimulator(n_simulations=5000, n_steps=8)
traj, df_term = sim.simulate_vectorized()
print(f"  ✓ Trajetórias completas: {traj.shape}")
print(f"  ✓ Média terminal: {df_term['selic_terminal'].mean():.2f}%")
print(f"  ✓ Std terminal: {df_term['selic_terminal'].std():.2f}%")

# 6. Com decisões condicionais
print("\n⚙️  MC com Decisões Condicionais (IPCA ajuste)...")
probs_cond = get_conditional_copom_probs(selic=data['selic'], ipca_12m=data['ipca_12m'])
traj_cond, df_cond = sim.simulate_vectorized(copom_probs_base=probs_cond)
print(f"  ✓ Média terminal: {df_cond['selic_terminal'].mean():.2f}%")
print(f"  ✓ Intervalo 80% (P10-P90): {df_cond['selic_terminal'].quantile(0.1):.2f}% - {df_cond['selic_terminal'].quantile(0.9):.2f}%")

# 7. Com todos os recursos
print("\n⚡ MC com Choques + Incerteza Parâmetros...")
traj_adv, df_adv = sim.simulate_vectorized(
    param_uncertainty_alpha=50.0,
    shock_prob=0.01,
    shock_mag=1.5,
)
print(f"  ✓ Média terminal: {df_adv['selic_terminal'].mean():.2f}%")
print(f"  ✓ Min: {df_adv['selic_terminal'].min():.2f}%, Max: {df_adv['selic_terminal'].max():.2f}%")

print("\n" + "=" * 70)
print("✅ QUICKSTART CONCLUÍDO")
print("=" * 70)
print("\n💡 Próximos passos:")
print("  1. Ver documentação: 'selic_model_methodology.md'")
print("  2. Rodar testes: 'python test_benchmark.py'")
print("  3. Rodar dashboard: 'streamlit run app.py'")
print("  4. Ver melhorias: 'IMPROVEMENTS_V2.md'")
print()
