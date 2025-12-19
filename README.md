# 💰 Selic Probability Dashboard 2026

Dashboard em Streamlit que modela a **distribuição de probabilidade da Selic em dez/2026**, usando:

- Dados em tempo real (Selic, IPCA, Focus)
- Modelo probabilístico calibrado em Focus
- Simulação Monte Carlo (10.000 trajetórias)
- Alertas opcionais via WhatsApp (Twilio)

## 🚀 Features

- 📈 Dashboard em tempo real (Selic, Focus, IPCA)
- 📊 Distribuição discreta de Selic terminal (10,5% a 14,0%)
- 🎲 Monte Carlo: 10k cenários possíveis de decisões do COPOM
- 🕊️/⚖️/🦅 Cenários dovish, central, hawkish com probabilidades
- 🎓 Tutorial explicando tudo para leigos (sem economês)
- 📱 Alertas via WhatsApp quando projeções mudam (opcional)

## 📂 Estrutura

Arquivos principais:

- `app.py` – App Streamlit multipage
- `config.py` – Configurações globais
- `data_fetcher.py` – Coleta de dados (BC, IBGE, Focus)
- `model.py` – Modelo probabilístico + Monte Carlo
- `notifier.py` – Notificações via WhatsApp
- `requirements.txt` – Dependências

## 🛠️ Instalação (resumo)

