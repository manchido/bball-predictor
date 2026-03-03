# BballPredictor

Production-grade international basketball game-total predictor (2023–2026).
Covers **EuroLeague**, **EuroCup**, **ACB** (Spain), **BSL** (Turkey), **BBL** (Germany).

---

## Bookmaker Source

| Priority | Source | Note |
|---|---|---|
| **Primary** | [The Odds API](https://the-odds-api.com) | Legitimate REST API — aggregates **bet365**, Pinnacle, William Hill, and others. Free tier: 500 req/month. |
| **Fallback** | OddsPortal (HTML scrape) | Used when API key absent or quota exceeded. Source stamped as `oddsportal` in tracker. |

> **Why not bet365/stake.com directly?** Both sites use Cloudflare + JS rendering and their ToS prohibit automated access. The Odds API provides bet365 odds legally via their aggregation service.

Every tracker row includes an `odds_source` column identifying the data origin.

---

## Architecture

```
RealGM (schedule, box scores, team stats)
        │
        ▼
/data/bronze/     Raw HTML/JSON + timestamps
        │
        ▼  [bronze_to_silver.py]
/data/silver/     Parsed parquet tables (schedules, box_scores, odds)
        │
        ▼  [silver_to_gold.py + features.py]
/data/gold/       Feature-engineered table (one row per game)
        │
        ▼  [trainer.py]
/models/          Stacked ensemble (XGB + LGBM + RF → Ridge)
        │
        ▼
FastAPI API  +  Typer CLI  +  APScheduler (09:00 America/St_Johns)
        │
        ▼
/tracking/daily_tracker.csv
```

### ML Stack

| Layer | Models |
|---|---|
| Base | XGBoost, LightGBM (mean), RandomForest |
| Intervals | LightGBM quantile (α=0.10, α=0.90) |
| Meta | Ridge regression on OOF predictions |

**Temporal splits:**
- Train: 2023–2024
- Validate: 2024–2025 (meta-model + calibration)
- Test: 2025–2026 (final metrics)

**Possessions formula (Oliver):**
```
POSS = FGA - ORB + TOV + 0.44 × FTA
```
Pace normalised to per-40 possessions; OT games scaled to 40-min equivalent.

---

## Setup

### 1. Clone & install

```bash
git clone <repo>
cd Bball-predictor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure `.env`

```bash
cp .env.example .env
```

Edit `.env` and add your Odds API key:

```
ODDS_API_KEY=your_key_here   # Get free key at https://the-odds-api.com
```

### 3. Build gold table & train

```bash
python cli.py build_gold
python cli.py train
```

---

## CLI Usage

```bash
# Train model
python cli.py train

# Rebuild gold table then train
python cli.py train --rebuild-gold

# Today's predictions (console)
python cli.py predict_today

# Today's predictions (JSON)
python cli.py predict_today --json

# Predict specific leagues only
python cli.py predict_today --league euroleague --league eurocup

# Backtest
python cli.py backtest --start 2025-10-01 --end 2026-02-28

# Smoke-test scrapers
python cli.py scrape_test --league euroleague

# Fill post-game actuals into tracker
python cli.py update_actuals
```

---

## API Server

```bash
uvicorn src.api.main:app --reload --port 8000
```

Interactive docs: `http://localhost:8000/docs`

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check + model status |
| `POST` | `/train` | Start training job |
| `GET` | `/train/{job_id}` | Training job status |
| `GET` | `/predict/today` | Today's predictions |
| `GET` | `/predictions/{game_id}` | Single game prediction |
| `POST` | `/backtest` | Historical backtest |
| `GET` | `/tracking` | Query daily tracker |
| `POST` | `/tracking/update_actuals` | Fill post-game results |

### Example: predict today

```bash
curl http://localhost:8000/predict/today?leagues=euroleague&leagues=eurocup
```

### Example: backtest

```bash
curl -X POST http://localhost:8000/backtest \
  -H "Content-Type: application/json" \
  -d '{"start_date": "2025-10-01", "end_date": "2026-02-28"}'
```

---

## Daily Tracker

`/tracking/daily_tracker.csv` columns:

| Column | Description |
|---|---|
| `date` | Game date |
| `league` | League ID |
| `match` | `Away @ Home` |
| `book_total` | Bookmaker over/under line |
| `book_spread` | Bookmaker spread |
| `model_total_mean` | Model prediction (mean) |
| `model_total_p10` | 10th percentile |
| `model_total_p90` | 90th percentile (calibrated to 80% coverage) |
| `confidence` | Width-based confidence score |
| `edge` | `model_mean − book_total` |
| `actual_total` | Final score total (filled post-game) |
| `error` | `actual − model_mean` |
| `odds_source` | `the-odds-api` or `oddsportal` |
| `timestamp` | Prediction timestamp (`America/St_Johns`) |

---

## Project Structure

```
Bball-predictor/
├── data/
│   ├── bronze/         # Raw HTML/JSON
│   ├── silver/         # Parsed parquet tables
│   └── gold/           # Feature-engineered modeling table
├── models/             # Serialised ensemble + metrics.json
├── tracking/           # daily_tracker.csv
├── logs/               # Rotating log files
├── src/
│   ├── api/
│   │   ├── main.py         # FastAPI app
│   │   ├── routers/        # train, predict, backtest, tracking
│   │   └── schemas.py      # Pydantic models
│   ├── scrapers/
│   │   ├── realgm.py       # RealGM scraper (throttled, cached)
│   │   ├── odds.py         # The Odds API + OddsPortal fallback
│   │   └── injuries.py     # Injury reports (graceful fallback)
│   ├── pipeline/
│   │   ├── bronze_to_silver.py
│   │   ├── features.py     # Rolling features, possessions, pace
│   │   └── silver_to_gold.py
│   ├── models/
│   │   ├── ensemble.py     # XGB + LGBM + RF → Ridge
│   │   ├── calibration.py  # 80% interval calibration
│   │   └── trainer.py      # Walk-forward CV + temporal splits
│   ├── injury/
│   │   └── adjuster.py     # Top-3 usage proxy, PPP adjustment
│   └── utils/
│       ├── config.py       # Settings (pydantic-settings + .env)
│       ├── logging.py      # loguru setup
│       ├── ids.py          # Team name normalisation + game_id
│       └── cache.py        # Disk cache with TTL
├── scheduler.py        # APScheduler daily job (09:00 St. John's)
├── cli.py              # Typer CLI
├── requirements.txt
└── .env.example
```

---

## Success Metrics

| Metric | Target |
|---|---|
| Game total MAE (test season) | < 12 pts |
| 80% interval calibration | 75–85% |
| Edge hit rate | > 50% |

Metrics are reported after training and saved to `models/metrics.json`.

---

## Injury Adjustment (Assumptions)

The injury module is a **proxy model**, not a precise substitution analysis:

- Top-3 players identified by `usage_pct × avg_minutes` (season-to-date)
- Each missing top-3 player → **3% reduction in team offensive rating proxy**
- Assumptions: conservative, deterministic, documented
- Fallback: if no injury data available, no adjustment is made
- Actual impact varies significantly by player role and team system

---

## Data Flow & Leakage Prevention

All rolling features use `.shift(1)` before computing rolling means, ensuring only games **before** the current one contribute to the feature. Season-to-date means use `.expanding()` with the same shift. OT games are normalised to 40-minute equivalents before rolling computation.
