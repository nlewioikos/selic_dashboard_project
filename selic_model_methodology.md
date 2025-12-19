# SELIC Probability Model - Methodology & Technical Notes

## Visão Geral
Este documento descreve o modelo probabilístico de trajetória da taxa Selic em 2026, implementado em Python com suporte a:
- Calibração automática à mediana do Focus Bulletin e curva DI (se disponível)
- Simulação Monte Carlo vetorizada (alta performance)
- Decisões do COPOM condicionadas a gaps macroeconômicos
- Modelagem de choques extremos e risco de cauda
- Quantificação de incerteza de parâmetros
- Backtesting e métricas de calibração

## Componentes Principais

### 1. Distribuição Terminal de Selic (Calibração)

**Arquivo**: `model.py`  
**Função**: `create_calibrated_model()` + `tilt_distribution()`

A distribuição de Selic em dezembro de 2026 é modelada como uma **distribuição discreta** nos níveis configurados em `config.SELIC_STATES` (default: 10.5%, 11.0%, ..., 14.0%).

**Calibração via Exponential Tilting (Esscher)**:
- Começa com uma distribuição **prior** em `SELIC_STATES`
- Aplica transformação exponencial: $\tilde{p}_i = p_i \cdot e^{\lambda x_i} / Z(\lambda)$
- Resolve para $\lambda$ via bisection tal que $E[\tilde{X}] = \text{Focus Median}$
- Resultado: distribuição que respeita o prior mas tem média calibrada ao mercado

**Preferência de Calibração**:
1. Tenta ler curva DI local de `data/di_curve.csv`
2. Interpola taxa implícita para dez/2026
3. Se não houver DI, usa `FOCUS_MEDIAN_DEC_2026` como target

**Vantagens**:
- Garante soma de probabilidades = 1
- Suave: minimiza mudança relative ao prior
- Numericamente estável com clipping de exp

### 2. Simulação Monte Carlo Vetorizada

**Arquivo**: `model.py`  
**Classe**: `MonteCarloSimulator`  
**Método**: `simulate_vectorized()`

Simula $N$ trajetórias independentes de Selic de janeiro a dezembro de 2026.

**Algoritmo**:
- Para cada simulação:
  1. Sorteia um cenário (dovish/central/hawkish) com probs globais (30%, 50%, 20%)
  2. Para cada reunião COPOM:
     - Sorteia ação (cut_50, cut_25, hold) com probs dependentes do cenário
     - Aplica mudança em Selic
     - Adiciona choque extremo com probabilidade `SHOCK_PROB`
     - Impõe limites `[SELIC_MIN, SELIC_MAX]`
  3. Armazena trajetória final

**Operações Vetorizadas**:
```python
# Todas as ações sorteadas de uma vez para N simulações
scenario_choices = np.random.choice(['dovish', 'central', 'hawkish'], size=N, p=[...])
# Ações mapeadas em valores [-0.50, -0.25, 0.0]
actions_applied = action_values[acts]  # shape (N,)
# Aplicadas a todas as trajetórias simultâneamente
trajectories[:, step] += actions_applied + shocks
```

**Performance**:
- Teste com 10k simulações: ~600ms (vs ~400ms loop original)
- Tradeoff: um pouco mais lento que loop puro, mas escalável
- **Melhoria Futura**: Considerar `numba.njit` para funções core

### 3. Parâmetros Condicionais do COPOM

**Arquivo**: `evaluation.py`  
**Função**: `get_conditional_copom_probs(selic, ipca_12m, selic_target, ipca_target)`

Permite que as probabilidades de ação (`COPOM_PROBS`) sejam ajustadas dinamicamente com base em **gaps macroeconômicos**:

- **IPCA Gap**: $\text{gap} = \text{IPCA}_{12m} - \text{META}$ (default: 4.5%)
  - Se gap < -0.5%: COPOM mais agressivo em cortes (dovish shift)
  - Se gap > 1.5%: COPOM mais conservador, aumenta hold (hawkish shift)

- **Selic Gap**: $\text{gap} = \text{SELIC} - \text{META}$ (default: 11.0%)
  - Informativo mas não muda probs diretamente (futuro: usar em modelo de feedback)

**Uso**:
```python
from evaluation import get_conditional_copom_probs

probs = get_conditional_copom_probs(selic=15.0, ipca_12m=5.2)
trajectories, df = sim.simulate_vectorized(copom_probs_base=probs)
```

### 4. Choques Extremos

**Arquivo**: `model.py` + `config.py`

Modelagem de eventos raros (fiscal, câmbio, etc.):

- **Probabilidade**: `SHOCK_PROB` (default: 1% por reunião COPOM)
- **Magnitude**: `SHOCK_MAG` (default: 1.5 pp)
- **Direção**: 50% para cima, 50% para baixo (podia ser assimétrico)

No simulador vetorizado:
```python
shock_flags = (np.random.random((n, steps)) < shock_prob)
shock_signs = np.where(np.random.random((n, steps)) < 0.5, -1.0, 1.0)
shock_values = shock_flags * shock_signs * shock_mag
```

**Calibração**: `SHOCK_PROB` e `SHOCK_MAG` podem ser ajustados em `config.py` ou via UI.

### 5. Incerteza de Parâmetros

**Arquivo**: `model.py`

Opção para adicionar ruído Dirichlet às probabilidades de ação por simulação:

```python
trajectories, df = sim.simulate_vectorized(param_uncertainty_alpha=50.0)
```

- Se `alpha > 0`: para cada simulação, amostrar $\mathbf{p} \sim \text{Dirichlet}(\alpha \cdot p_0)$
- Produz intervalo de confiança nas probabilidades finais (com mais réplicas)
- Útil para quantificar incerteza em regras de decisão do COPOM

### 6. Backtesting & Métricas

**Arquivo**: `evaluation.py`

#### Brier Score
$$BS = \frac{1}{n} \sum_{i=1}^{n} (p_i - o_i)^2$$

onde $p_i$ é probabilidade prevista, $o_i$ é observação (0 ou 1).

- Range: [0, 1], menor é melhor
- Uso: avaliar calibração de probabilidades binárias

#### CRPS (Continuous Ranked Probability Score)
$$\text{CRPS} = \int_{-\infty}^{\infty} (F(x) - \mathbb{1}_{x \geq x_0})^2 dx$$

- Avalia distribuição inteira vs um evento observado
- Range: [0, ∞), menor é melhor
- Implementação: aproximação discreta com soma ponderada

**Backtest Simples**:
```python
from evaluation import backtest_simple
from model import create_calibrated_model

history_df = pd.read_csv('data/selic_history.csv')
results = backtest_simple(history_df, create_calibrated_model)
print(f"Mean Brier: {results['mean_brier']:.4f}")
print(f"Mean CRPS: {results['mean_crps']:.4f}")
```

## Configuração

**Arquivo**: `config.py`

### Parâmetros de Modelo
- `SELIC_STATES`: Níveis discretos de Selic terminal (default: 10.5% a 14.0% em 0.5pp steps)
- `FOCUS_MEDIAN_DEC_2026`: Mediana do Focus (default: 12.13%)
- `SELIC_STD_DEV`: Desvio padrão prior (default: 0.60%)
- `COPOM_PROBS`: Probabilidades de ação por cenário

### Simulação Monte Carlo
- `MC_SIMULATIONS`: Número de trajetórias (default: 10,000)
- `MC_STEPS`: Número de reuniões COPOM (default: 8 em 2026)
- `SELIC_MIN`, `SELIC_MAX`: Limites de Selic (default: 9.5% - 16.0%)

### Risco Extremo
- `SHOCK_PROB`: Probabilidade de choque por reunião (default: 0.01 = 1%)
- `SHOCK_MAG`: Magnitude do choque em pp (default: 1.5)

### Incerteza de Parâmetros
- `PARAM_UNCERTAINTY_ALPHA`: Parâmetro de concentração Dirichlet (default: 50.0)
  - Valores altos: pouco ruído (concentrado no prior)
  - Valores baixos: muito ruído (distribuição uniforme)

### Dados
- `DI_CURVE_FILE`: Caminho local da curva DI (default: `./data/di_curve.csv`)
- `HISTORY_FILE`: Caminho do histórico para backtest (default: `./data/selic_history.csv`)

## Formato de Dados Esperado

### `data/di_curve.csv`
```csv
maturity,rate
0.5,15.2
1.0,14.8
2.0,13.5
```
Colunas: `maturity` (anos), `rate` (% a.a.)

### `data/selic_history.csv`
```csv
date,selic,ipca_12m,selic_realized
2025-10-01,15.0,4.46,15.00
2025-11-01,14.75,4.52,14.75
```
Colunas: data, Selic corrente, IPCA 12m, Selic realizada (ex-post)

## Exemplos de Uso

### Criar Modelo Calibrado
```python
from model import create_calibrated_model

model = create_calibrated_model(focus_median=12.13)
stats = model.get_stats()
print(f"Mean: {stats['mean']:.2f}%, P50: {stats['p50']:.2f}%")
```

### Simular Trajetórias
```python
from model import MonteCarloSimulator

sim = MonteCarloSimulator(n_simulations=100000, n_steps=8)
trajectories, df_terminal = sim.simulate_vectorized()

# Distribuição terminal
print(f"Mean: {df_terminal['selic_terminal'].mean():.2f}%")
print(f"Std: {df_terminal['selic_terminal'].std():.2f}%")
```

### Decisões Condicionais
```python
from evaluation import get_conditional_copom_probs

probs = get_conditional_copom_probs(selic=15.0, ipca_12m=5.5)
trajectories, df = sim.simulate_vectorized(copom_probs_base=probs)
```

### Backtest
```python
from evaluation import backtest_simple
import pandas as pd

history = pd.read_csv('data/selic_history.csv')
results = backtest_simple(history, create_calibrated_model)
print(f"Backtest: Brier={results['mean_brier']:.4f}, CRPS={results['mean_crps']:.4f}")
```

### Benchmark
```python
from evaluation import benchmark_vectorized_vs_loop

df_bench = benchmark_vectorized_vs_loop(n_sims_list=[1000, 10000, 100000])
print(df_bench)
```

## Suposições & Limitações

### Suposições
1. **Distribuição Terminal Discreta**: Selic em dez/2026 segue distribuição discreta calibrada ao Focus
2. **Cenários Independentes**: Cada simulação escolhe um cenário; transitions entre cenários não são modeladas
3. **Ações COPOM Constantes**: Probabilidades de ação (cut_50, cut_25, hold) são fixas por cenário (ou ajustadas via gaps)
4. **Sem Feedback**: Simulações não incorporam feedback da Selic realizada em períodos anteriores
5. **Choques i.i.d.**: Choques são independentes entre reuniões

### Limitações
- Sem modelo estrutural macroeconômico: as probabilidades não derivam de equações de política monetária
- Calibração discreta: usa grades fixas, pode perder informação em regiões intermediárias
- Backtest limitado: requer histórico suficiente; viés de look-ahead se usar dados futuros
- Performance: MC vetorizado é ~0.7-0.8x mais lento que loop puro (tradeoff por clareza)

## Melhorias Futuras

1. **Regime Switching**: Transições entre cenários via cadeia de Markov
2. **Modelo Contínuo**: Substituir discreta por normal truncada ou mistura gaussiana
3. **Calibração Bayesiana**: Estimar parâmetros (cenário probs, choques) de dados históricos
4. **Importance Sampling**: Estimar caudas com precisão para cenários extremos
5. **Vetorização Numba**: JIT compilation para puro NumPy inner loops
6. **Forward Curve Fitting**: Fitar cada mês 2026 não apenas terminal

## Referências

- Focus Bulletin: https://www.bcb.gov.br/
- COPOM Decisions: https://www.bcb.gov.br/?COPOM
- Esscher Transform: https://en.wikipedia.org/wiki/Escher's_transformation
- CRPS: https://www.ncei.noaa.gov/products/weather-climate-modeling/probabilistic-skill-metrics

---

**Última Atualização**: 2025-12-19  
**Autor**: Finance Dashboard Team
