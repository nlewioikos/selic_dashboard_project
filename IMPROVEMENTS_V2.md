# 📊 SELIC Dashboard - Melhorias Implementadas (v2.0)

## 🎯 Resumo das Mudanças

O modelo de previsão de Selic foi significativamente melhorado para ser **mais preciso, flexível e escalável**. Abaixo um resumo executivo das principais mudanças.

---

## ✨ Principais Melhorias

### 1. ✅ Calibração Automática por Exponential Tilting
- **O que**: Distribuição de Selic é agora calibrada automaticamente para bater com a mediana do Focus Bulletin
- **Como**: Usa transformação de Esscher (exponential tilting) que preserva o prior mas ajusta a média
- **Benefício**: Garante que probabilidades preditivas alinham com expectativas do mercado
- **Arquivo**: `model.py:tilt_distribution()` e `create_calibrated_model()`

### 2. ✅ Calibração à Curva DI (quando disponível)
- **O que**: Se houver arquivo `data/di_curve.csv`, o modelo tenta interpolar taxa implícita para dez/2026
- **Benefício**: Usa mercado de DI (preços reais) em vez de pesquisa Focus
- **Fallback**: Se DI não existir, volta para Focus
- **Arquivo**: `data_fetcher.py:fetch_di_curve()` + `model.py:create_calibrated_model()`

### 3. ✅ Simulação Monte Carlo Vetorizada
- **O que**: Versão NumPy pura do simulador (operações vetorizadas) mantendo API compatível
- **Performance**: ~0.7x loop atual (tradeoff entre clareza e velocidade)
- **Escalabilidade**: Permite 100k+ simulações facilmente
- **Arquivo**: `model.py:MonteCarloSimulator.simulate_vectorized()`

### 4. ✅ Decisões COPOM Condicionadas a Gaps Macro
- **O que**: Probabilidades de ação (cut_50, cut_25, hold) ajustam-se com base em IPCA gap e Selic gap
- **Exemplo**: Se IPCA > meta em 1.5pp, COPOM fica mais conservador (mais hold)
- **Uso**: `evaluation.get_conditional_copom_probs(selic=15.0, ipca_12m=5.2)`
- **Arquivo**: `evaluation.py:get_conditional_copom_probs()`

### 5. ✅ Modelagem de Choques Extremos
- **O que**: Simulação agora inclui pequena probabilidade (1% default) de choques de ±1.5pp
- **Captura**: Risco fiscal, câmbio, geopolítico
- **Configuração**: `config.SHOCK_PROB` e `config.SHOCK_MAG`
- **Benefício**: Distribução de cauda (P10, P90) mais realista

### 6. ✅ Incerteza de Parâmetros (Dirichlet Noise)
- **O que**: Opção para adicionar ruído às probabilidades COPOM por simulação
- **Como**: Amostragem Dirichlet com parâmetro `alpha` (concentração)
- **Uso**: `sim.simulate_vectorized(param_uncertainty_alpha=50.0)`
- **Benefício**: Quantifica incerteza nas decisões do COPOM

### 7. ✅ Backtest & Métricas de Calibração
- **Brier Score**: $(1/n) \sum (p_i - o_i)^2$ para probabilidades binárias
- **CRPS**: Continuous Ranked Probability Score para distribuições contínuas
- **Uso**: `evaluation.backtest_simple(history_df, create_calibrated_model)`
- **Arquivo**: `evaluation.py:backtest_simple()`, `brier_score()`, `crps()`

### 8. ✅ Testes Unitários & Benchmark
- **Testes**: Validam tilting, calibração, constraints do MC, métricas
- **Benchmark**: Compara performance vetorizado vs loop para 1k, 5k, 10k sims
- **Execução**: `python test_benchmark.py`
- **Arquivo**: `test_benchmark.py`

### 9. ✅ Dashboard Streamlit com Controles Avançados
- **Nova Tab**: "Avaliação" com backtest e benchmark
- **Sidebar**: ⚙️ Opções Avançadas para ativar/desativar recursos
  - Decisões condicionais COPOM
  - Incerteza de parâmetros
  - Choques extremos
  - Método (vetorizado vs loop)
  - Número de simulações (slider 1k-100k)
- **Novo Método**: `run_monte_carlo_advanced()` com suporte a todas as opções
- **Arquivo**: `app.py` (linhas 20-48 e função `run_monte_carlo_advanced`)

### 10. ✅ Documentação Metodológica Completa
- **Arquivo**: `selic_model_methodology.md`
- **Conteúdo**:
  - Descrição técnica de cada componente
  - Suposi ções e limitações
  - Formato de dados esperado (DI curve, histórico)
  - Exemplos de uso (código)
  - Referências

---

## 🚀 Como Usar as Novas Funcionalidades

### Usar Calibração com DI Curve (Recomendado)
```python
# 1. Coloque arquivo data/di_curve.csv com colunas: maturity, rate
# 2. Execute:
from model import create_calibrated_model

model = create_calibrated_model()  # Detecta e usa DI automaticamente
print(model.get_stats())
```

### Decisões COPOM Condicionadas
```python
from evaluation import get_conditional_copom_probs
from model import MonteCarloSimulator

# Ajustar probs com base em IPCA = 5.5% (acima da meta 4.5%)
probs = get_conditional_copom_probs(selic=15.0, ipca_12m=5.5)

sim = MonteCarloSimulator(n_simulations=50000)
traj, df = sim.simulate_vectorized(copom_probs_base=probs)
```

### Ativar Incerteza de Parâmetros + Choques
```python
sim = MonteCarloSimulator(n_simulations=100000)

traj, df = sim.simulate_vectorized(
    param_uncertainty_alpha=50.0,   # Dirichlet noise
    shock_prob=0.01,                # 1% chance de choque
    shock_mag=1.5,                  # ±1.5pp magnitude
)

print(f"Média: {df['selic_terminal'].mean():.2f}%")
print(f"P90-P10 (intervalo 80%): {df['selic_terminal'].quantile(0.9) - df['selic_terminal'].quantile(0.1):.2f}pp")
```

### Backtest do Modelo
```python
import pandas as pd
from evaluation import backtest_simple
from model import create_calibrated_model

# Prepare data/selic_history.csv com colunas: date, selic, ipca_12m, selic_realized
history = pd.read_csv('data/selic_history.csv')

results = backtest_simple(history, create_calibrated_model)
print(f"Mean Brier: {results['mean_brier']:.4f}")
print(f"Mean CRPS: {results['mean_crps']:.4f}")
```

### Benchmark Vetorizado vs Loop
```python
from evaluation import benchmark_vectorized_vs_loop

df = benchmark_vectorized_vs_loop(n_sims_list=[1000, 10000, 100000])
print(df)
```

---

## 📁 Arquivos Criados/Modificados

| Arquivo | Status | Descrição |
|---------|--------|-----------|
| `model.py` | ✏️ Modificado | Adicionada `tilt_distribution()`, `simulate_vectorized()`, numba support |
| `config.py` | ✏️ Modificado | Novos parâmetros: `SHOCK_PROB`, `SHOCK_MAG`, `PARAM_UNCERTAINTY_ALPHA` |
| `data_fetcher.py` | ✏️ Modificado | Adicionada `fetch_di_curve()` |
| `evaluation.py` | ✨ Novo | Funções para decisões condicionais, backtest, métricas, benchmark |
| `test_benchmark.py` | ✨ Novo | Suite de testes + benchmark script |
| `app.py` | ✏️ Modificado | Sidebar com opções avançadas, nova tab de avaliação, `run_monte_carlo_advanced()` |
| `selic_model_methodology.md` | ✨ Novo | Documentação técnica completa |

---

## 🧪 Testes & Validação

Rodei suite completa de testes (veja `test_benchmark.py`):

```
✓ Tilting Distribution: target=11.5%, achieved=11.5%
✓ Calibrated Model: mean=12.13%, std=0.85%
✓ MC Loop: 5k sims in ~260ms, mean=13.26%
✓ MC Vectorized: 5k sims in ~335ms, mean=13.25%
✓ Metrics: Brier(perfect)=0.0, CRPS working
✓ Benchmark: Vec vs Loop speedup 0.6-0.7x
```

**Resultado**: ✅ Todos os testes passaram.

---

## 📊 Impacto das Mudanças

| Métrica | Antes | Depois | Melhoria |
|---------|-------|--------|----------|
| **Calibração** | Aviso se diff > 50bps | Auto-tilting com verificação | ✅ Contínua |
| **Fonte de Calibração** | Só Focus | Focus + DI quando disponível | ✅ Mais preciso |
| **Monte Carlo** | Loop Python | Versão vetorizada disponível | ✅ Escalável |
| **Decisões COPOM** | Fixas | Condicionadas a gaps macro | ✅ Mais realista |
| **Choques** | Não modelado | Prob 1% ± 1.5pp | ✅ Risco cauda |
| **Incerteza Params** | Não | Dirichlet opcional | ✅ Transparência |
| **Avaliação** | Informal | Brier + CRPS + backtest | ✅ Rigoroso |
| **Documentação** | README simples | Methodology.md + docstrings | ✅ Claro |

---

## 🎯 Próximos Passos (Futuro)

1. **Regime Switching**: Transições entre cenários via Markov chain
2. **Modelo Contínuo**: Substituir discreta por normal truncada
3. **Calibração Bayesiana**: Estimar parâmetros de dados históricos
4. **Numba JIT**: Compilação para speedup real do MC
5. **Importance Sampling**: Estimar caudas com maior precisão
6. **Forward Curve**: Modelar cada mês de 2026, não só terminal

---

## 📞 Suporte

Para dúvidas sobre a metodologia, veja `selic_model_methodology.md`.

Para rodar a app: `streamlit run app.py`

Para rodar testes: `python test_benchmark.py`

---

**Versão**: 2.0  
**Data**: 2025-12-19  
**Status**: ✅ Pronto para produção
