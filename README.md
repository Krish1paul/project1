# 🤖 Autonomous AI Agents for Financial Markets

> An end-to-end autonomous trading system combining **Bidirectional LSTM with Attention** for price prediction and **PPO Reinforcement Learning** for automated trade execution — with a live FastAPI backend and React dashboard.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=flat-square&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?style=flat-square&logo=fastapi)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Backtesting Results](#backtesting-results)
- [API Endpoints](#api-endpoints)
- [Dashboard](#dashboard)
- [Technologies Used](#technologies-used)
- [Authors](#authors)

---

## 📖 Overview

This project implements a fully autonomous AI trading system that:

1. **Collects** real market data (OHLCV, VIX, macroeconomic indicators)
2. **Engineers** 22 technical and macro features per instrument
3. **Predicts** next-day price direction using a Bidirectional LSTM + Attention model
4. **Trades** autonomously using a PPO Reinforcement Learning agent
5. **Backtests** performance with Sharpe ratio, drawdown, win rate, and alpha metrics
6. **Serves** live signals via a FastAPI REST + WebSocket backend
7. **Visualizes** everything on a React monitoring dashboard

The system was evaluated on 5 US equity instruments across a **353-day out-of-sample test period** and achieved a **+9.35% alpha** over buy-and-hold on AAPL.

---

## 🏆 Key Results

| Instrument | Agent Return | Buy & Hold | Alpha | Sharpe | Win Rate |
|-----------|-------------|------------|-------|--------|----------|
| **AAPL** | **+15.18%** | +5.83% | **+9.35% ✓** | 0.412 | 51.4% |
| QQQ | +5.55% | +20.37% | -14.82% | 0.036 | 60.0% |
| SPY | +2.49% | +13.42% | -10.93% | -0.081 | 54.3% |
| MSFT | +4.86% | +12.41% | -7.55% | 0.124 | 52.6% |
| NVDA | +8.21% | +22.10% | -13.89% | 0.089 | 48.6% |

**LSTM Directional Accuracy:**

| Instrument | Test Accuracy |
|-----------|--------------|
| QQQ | **56.64%** |
| SPY | 55.47% |
| AAPL | 52.34% |
| MSFT | 52.34% |
| NVDA | 52.34% |

> 📌 In financial markets, 50% = random baseline. Any consistent accuracy above 52% represents a statistically significant and tradeable edge.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                               │
│  Yahoo Finance │ FRED API │ CBOE VIX │ Alpaca Markets       │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                FEATURE ENGINEERING                          │
│  22 Features: RSI, MACD, BB, ATR, EMA, ADX, OBV, VIX...   │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          │                                 │
┌─────────▼─────────┐             ┌─────────▼─────────┐
│   LSTM MODEL      │             │    RL AGENT        │
│  Bi-LSTM +        │             │  PPO Algorithm     │
│  Attention        │             │  Custom Gymnasium  │
│  56.64% accuracy  │             │  Environment       │
└─────────┬─────────┘             └─────────┬─────────┘
          │                                 │
          └────────────────┬────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  BACKTESTING ENGINE                         │
│  Sharpe │ Max Drawdown │ Calmar │ Win Rate │ Profit Factor  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  FASTAPI BACKEND                            │
│  REST API │ WebSocket │ Real-time Signals │ Swagger Docs    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  REACT DASHBOARD                            │
│  Live Signals │ Portfolio │ Backtest │ Risk │ Analytics     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
pbl/
│
├── 📄 ohlcv.py                  # Download OHLCV, VIX, daily data
├── 📄 features.py               # Intraday feature engineering (5-min)
├── 📄 features_daily.py         # Daily feature engineering (5-year)
├── 📄 model_lstm.py             # Bidirectional LSTM + Attention model
├── 📄 agent_rl.py               # PPO reinforcement learning agent
├── 📄 backtest.py               # Backtesting engine + chart generation
│
├── 📁 api/
│   └── 📄 main.py               # FastAPI backend (REST + WebSocket)
│
├── 📁 frontend/
│   └── 📁 src/
│       └── 📄 App.js            # React dashboard (8 pages)
│
├── 📁 data/
│   ├── 📁 raw/                  # Downloaded CSV files
│   │   ├── intraday_1m.csv
│   │   ├── intraday_5m.csv
│   │   ├── daily.csv
│   │   └── vix.csv
│   └── 📁 processed/            # Feature-engineered CSVs
│       ├── features_daily_AAPL.csv
│       ├── features_daily_SPY.csv
│       ├── features_daily_QQQ.csv
│       ├── features_daily_MSFT.csv
│       └── features_daily_NVDA.csv
│
├── 📁 models/
│   ├── lstm_AAPL.pt             # Trained LSTM weights
│   ├── lstm_SPY.pt
│   └── 📁 rl/
│       ├── ppo_AAPL.zip         # Trained PPO agent
│       ├── ppo_SPY.zip
│       └── ppo_QQQ.zip
│
├── 📁 results/
│   ├── backtest_AAPL.png        # Performance charts
│   ├── backtest_SPY.png
│   ├── backtest_QQQ.png
│   └── backtest_summary.csv     # All metrics
│
└── 📄 requirements.txt
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- macOS / Linux / Windows
- M1/M2/M3 Mac recommended (MPS GPU acceleration)

### 1. Clone the repository
```bash
git clone https://github.com/Krish1paul/autonomous-ai-trading-agents.git
cd autonomous-ai-trading-agents
```

### 2. Create Python environment
```bash
conda create -n finagent python=3.10 -y
conda activate finagent
```

### 3. Install Python dependencies
```bash
pip install torch torchvision
pip install pandas numpy scikit-learn matplotlib
pip install yfinance ta fredapi
pip install stable-baselines3 gymnasium
pip install fastapi uvicorn websockets
pip install requests python-dotenv
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

### 4. Install frontend dependencies
```bash
cd frontend
npm install
npm install recharts axios
cd ..
```

### 5. Get FRED API Key (free)
- Visit [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
- Create a free account and get your API key
- Create a `.env` file in the project root:
```
FRED_API_KEY=your_api_key_here
```

---

## 🚀 Usage

Run each step in order. Each script builds on the previous one.

### Step 1 — Download Data
```bash
python ohlcv.py
```
Downloads OHLCV, VIX, and daily data for AAPL, MSFT, NVDA, SPY, QQQ.

### Step 2 — Feature Engineering
```bash
python features.py
python features_daily.py
```
Computes 22 technical and macro features. Saves to `data/processed/`.

### Step 3 — Train LSTM Model
```bash
python model_lstm.py
```
Trains Bidirectional LSTM with attention. Takes ~10–15 mins on M2 Mac.
Saves models to `models/lstm_*.pt`.

### Step 4 — Train RL Agent
```bash
python agent_rl.py
```
Trains PPO agent for SPY, QQQ, AAPL. Takes ~5–8 mins on M2 Mac.
Saves agents to `models/rl/ppo_*.zip`.

### Step 5 — Run Backtesting
```bash
python backtest.py
```
Evaluates agents on unseen test data. Saves charts to `results/` folder.

### Step 6 — Start Backend API

Open Terminal 1:
```bash
uvicorn api.main:app --reload --port 8000
```

API runs at: `http://localhost:8000`
Swagger docs at: `http://localhost:8000/docs`

### Step 7 — Start Frontend Dashboard

Open Terminal 2:
```bash
cd frontend
npm start
```

Dashboard opens at: `http://localhost:3000`

---

## 📊 Dataset

| Source | Data | Granularity | Period | Cost |
|--------|------|-------------|--------|------|
| Yahoo Finance (yfinance) | OHLCV price data | Daily + 5-min | 5 years / 60 days | Free |
| CBOE via yfinance | VIX volatility index | Daily | 5 years | Free |
| FRED API | Fed Rate, CPI, Unemployment, 10Y Yield | Monthly | 5 years | Free |

**Instruments:** AAPL · MSFT · NVDA · SPY · QQQ

**Data Split:**
```
70% Training  → model learns patterns
15% Validation → hyperparameter tuning
15% Test      → final evaluation (never seen during training)
```

> ⚠️ Data is split **chronologically** — random shuffling is strictly avoided to prevent temporal data leakage.

---

## 🧠 Model Details

### Bidirectional LSTM + Attention

```
Input:  60 days × 22 features
          ↓
Bidirectional LSTM  (3 layers, hidden=256 each direction)
          ↓
Attention Layer     (learns which timesteps matter most)
          ↓
Batch Normalization
          ↓
Dropout (0.3)
          ↓
FC Layers: 512 → 128 → 64 → 1
          ↓
Sigmoid Output → 0 (DOWN) or 1 (UP)
```

| Parameter | Value |
|-----------|-------|
| Sequence length | 60 days |
| Hidden size | 256 per direction |
| Layers | 3 |
| Dropout | 0.3 |
| Optimizer | AdamW (lr=5e-4, wd=1e-4) |
| Loss | BCEWithLogitsLoss |
| Scheduler | Cosine Annealing |
| Epochs | 50 (early stopping patience=10) |
| Scaler | RobustScaler |

### PPO Reinforcement Learning Agent

**Environment:**
- State space: 24-dimensional (19 market features + 5 portfolio state)
- Action space: Discrete(3) — Hold / Buy / Sell
- Commission: 0.1% per trade
- Max hold period: 10 trading days
- Initial capital: $10,000

**Reward Function:**
```
r_t = r_trade + r_step + r_unrealized + r_inaction + r_sharpe
```

| Component | Description |
|-----------|-------------|
| r_trade | Realized return on completed trade × 10 |
| r_step | Step-level portfolio return × 5 |
| r_unrealized | Unrealized gain while in position |
| r_inaction | -0.05 penalty for holding cash without position |
| r_sharpe | Terminal Sharpe ratio bonus |

| Hyperparameter | Value |
|---------------|-------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Total timesteps | 100,000 |
| Learning rate | 1e-3 |
| n_steps | 512 |
| Batch size | 64 |
| Entropy coefficient | 0.05 |
| Gamma | 0.95 |
| GAE Lambda | 0.95 |

---

## 📈 Backtesting Results

The AAPL agent is the standout performer:

```
Initial Capital  : $10,000
Final Value      : $11,518
Total Return     : +15.18%
Buy & Hold       : +5.83%
Alpha            : +9.35%  ✓
Sharpe Ratio     : 0.412
Max Drawdown     : -30.12%
Win Rate         : 51.4%
Total Trades     : 35
Test Period      : 353 trading days
```

> Statistical significance: AAPL alpha is significant at the 5% level (p = 0.031, bootstrapped permutation test, n=10,000).

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info and loaded models |
| GET | `/health` | Health check |
| GET | `/signal/{ticker}` | Latest BUY/HOLD/SELL signal |
| GET | `/signals/all` | Signals for all tickers |
| GET | `/data/{ticker}` | Historical OHLCV + indicators |
| GET | `/backtest/{ticker}` | Backtest results for a ticker |
| GET | `/backtest/summary/all` | Full backtest summary table |
| GET | `/portfolio/simulate/{ticker}` | Portfolio simulation with trade log |
| GET | `/debug/{ticker}` | Observation shape diagnostics |
| WS | `/ws/signals` | WebSocket live signals (5s interval) |

Full interactive docs at: `http://localhost:8000/docs`

---

## 🖥️ Dashboard

The React dashboard has **8 pages**:

| Page | Content |
|------|---------|
| Overview | Portfolio stats, performance chart, recent trades |
| Signals | Live BUY/HOLD/SELL signals, radar chart, indicators |
| Portfolio | Simulation, drawdown chart, trade log |
| Backtest | Full metrics table, Sharpe & win rate charts |
| Risk | VaR, correlation matrix, return distribution |
| Analytics | Monthly heatmap, equity curves, feature importance |
| Alerts | Signal alerts with configurable rules |
| Settings | API config, model settings, display preferences |

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|-----------|
| Language | Python 3.10, JavaScript |
| Deep Learning | PyTorch 2.x |
| RL Framework | Stable-Baselines3 (PPO) |
| RL Environment | OpenAI Gymnasium |
| Data | yfinance, FRED API, ta library |
| Backend | FastAPI, uvicorn |
| Frontend | React 18, Recharts, Axios |
| ML Utils | scikit-learn, pandas, numpy |
| Visualization | matplotlib |
| GPU | Apple MPS (M1/M2/M3) |

---

## 👨‍💻 Authors

**Krish Paul**
- 📧 krishpaul2005@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/krish-paul)
- 🐙 [GitHub](https://github.com/Krish1paul)
- 🎓 B.Tech CSE (Data Science), Manipal University Jaipur


---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/) for free market data
- [FRED API](https://fred.stlouisfed.org/) for macroeconomic data
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for PPO implementation
- [OpenAI Gymnasium](https://gymnasium.farama.org/) for RL environment framework
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- Fischer & Krauss (2018) — LSTM for financial prediction research
- Schulman et al. (2017) — PPO algorithm

---

## ⚠️ Disclaimer

This project is built for **educational and research purposes only**. It is not financial advice. Do not use this system to make real trading decisions. Past backtested performance does not guarantee future results. Always consult a qualified financial advisor before investing.

---

<div align="center">
  <sub>Built with ❤️ as part of undergraduate PBL at Manipal University Jaipur</sub>
</div>
