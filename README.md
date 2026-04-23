# Hourly Electricity Demand Forecasting

### Predictive Paradox — IITG.ai Recruitment Task

---

## Objective

Predict Bangladesh's next hour electricity demand (`demand_mw` at time T+1) using historical grid data, weather observations, and annual economic indicators. Only classical machine learning is used — no deep learning, no ARIMA, no Prophet.

---

## File Structure

```
├── main.ipynb                            # Full end-to-end pipeline
├── README.md
├── PGCB_date_power_demand.xlsx           # Hourly grid demand & generation data
├── weather_data.xlsx                     # Hourly weather observations
└── economic_full_1.csv                   # Annual macroeconomic indicators (World Bank)
```

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/PrinceK-Git/Electricity-Demand-Forecasting.git
cd Electricity-Demand-Forecasting
```

2. Install dependencies
```bash
pip install numpy pandas matplotlib scikit-learn xgboost openpyxl
```

3. Place the three raw data files in the same directory as `main.ipynb`:
   - `PGCB_date_power_demand.xlsx`
   - `weather_data.xlsx`
   - `economic_full_1.csv`

4. Open and run the notebook
```bash
jupyter notebook main.ipynb
```
Run all cells from top to bottom. The notebook will clean the data, engineer features, train the model, and print MAPE for 2024 and 2025.



---

## Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn xgboost openpyxl
```

---

## Data Sources

### 1. PGCB Demand Data (`PGCB_date_power_demand.xlsx`)
Hourly power grid records from the Power Grid Company of Bangladesh.

| Column | Description |
|---|---|
| `datetime` | Hourly timestamp |
| `demand_mw` | Total electricity demand (MW) |
| `generation_mw` | Total power generated (MW) |
| `load_shedding` | Amount of load shed (MW) |
| `gas`, `liquid_fuel`, `coal`, `hydro`, `solar`, `wind` | Generation by fuel type (MW) |
| `india_bheramara_hvdc`, `india_tripura`, `india_adani`, `nepal` | Cross-border imports (MW) |

### 2. Weather Data (`weather_data.xlsx`)
Hourly weather readings with 9 indicators: temperature, apparent temperature, dew point, relative humidity, precipitation, soil temperature, wind direction, cloud cover, and sunshine duration.

### 3. Economic Data (`economic_full_1.csv`)
Annual World Bank indicators for Bangladesh — 112+ indicators including GDP, population, urban growth, trade (exports/imports), and infrastructure metrics. Stored in wide format (years as columns), reshaped to long format and joined to the hourly series by calendar year.

---

## Data Preparation

### Step 1 — Half-hourly timestamp handling
The raw data contains entries with timestamps ending in `:30` (half-hourly readings). These are combined with their corresponding hour using a weighted average:
- Hour mark (`:00`) → weight **0.6**
- Half-hour mark (`:30`) → weight **0.4**

**Reasoning behind the weights:** The `:00` reading represents demand at the start of the hour — the most representative snapshot for that hour. The `:30` reading is a mid-hour observation, by which point demand has already partially transitioned toward the next hour's level. It therefore carries less weight (0.4) since it is less representative of the hour as a whole compared to the `:00` reading (0.6).

This collapses all rows into a clean hourly grid.

### Step 1b — NaN and missing value handling in PGCB data
Different columns are handled differently based on their physical meaning:

- **`india_adani` and `nepal`** — filled with **0**. These are cross-border import connections that did not exist in the early years of the dataset. A missing value means the connection was not yet operational, so 0 MW is the physically correct value, not an unknown.
- **`solar` and `wind`** — filled with **0**. Same reasoning — utility-scale solar and wind capacity was not present in Bangladesh's grid in earlier years. Missing values mean zero generation, not a data gap.
- **Missing hourly timestamps** — after collapsing to an hourly grid, the datetime index is reindexed to a complete continuous hourly range. Any hours missing entirely from the raw data are filled using forward fill — the missing hour inherits the value of the most recent known hour before it. This ensures lag features always reference the correct point in time rather than silently skipping rows.

- **Rows after feature engineering** — after creating lag features (`lag_168h`, `demand_8760h_ago`) and the target column (`target_next_hour_demand`), rows that cannot have valid values are removed using `dropna()`. This removes the first ~8760 rows (which lack a full year of lag history) and the last row (which has no next-hour target). These rows are genuinely unusable and are correctly discarded rather than imputed.

### Step 2 — Clipping known physical limits
Before outlier removal, columns with known physical capacity limits are hard-clipped:

| Column | Cap (MW) |
|---|---|
| `india_tripura` | 160 |
| `india_bheramara_hvdc` | 1000 |
| `india_adani` | 1500 |
| `nepal` | 50 |
| `solar` | 500 |
| `wind` | 100 |
| `gas` | 8000 |
| `liquid_fuel` | 6500 |
| `coal` | 5500 |
| `hydro` | 350 |

### Step 3 — Outlier removal on `demand_mw`
Global IQR method applied on the full demand series:
- Lower bound = Q1 − 1.5 × IQR
- Upper bound = Q3 + 1.5 × IQR

This removed 87 rows out of ~88,000 — only extreme, physically impossible spikes.

**Why IQR:** It is a non-parametric, distribution-free method that does not assume normality. It is widely accepted in power systems data cleaning and is mathematically sound for detecting undocumented spikes.

### Step 4 — Merging weather and economic data
- Weather data is merged on `datetime` (hourly join).
- Economic data is joined by calendar `year` in two stages:

**Stage 1 — Core indicators:** GDP, Population, and Industry value added are directly extracted and merged.

**Stage 2 — Extended indicators (112 additional features):** The 1,500+ available World Bank indicators are searched using keywords like "Urban population", "Transport", "exports", and "imports" to find relevant socioeconomic data. The matched indicators are processed as follows:
1. **Keyword search** — finds 112 matching indicators from the full list
2. **Filtering and melting** — pulls those indicators and converts them from wide format (years as columns) into long format (one row per year)
3. **Pivoting** — reshapes into a clean table where each row is a year and each column is an economic indicator
4. **Column name cleaning** — removes spaces and special characters (`%`, `(`, `,`) so column names are Python-friendly (e.g. `Urban_population`)
5. **Left merge** — attaches the new indicators to the hourly dataset by matching on year
6. **Interpolation** — since economic data is recorded once a year, linear interpolation creates smooth transitions across the 8,760 hours of each year, giving the model continuous values to learn from

The result is a dataset that grows from a few columns to **152 columns** in total.

---

## Feature Engineering

All features use only information available at time T or earlier. The target is demand at T+1.

### Calendar Features
| Feature | Description |
|---|---|
| `hour` | Hour of day (0–23) |
| `day` | Day of month |
| `month` | Month of year (1–12) |
| `day_of_week` | Day of week (0 = Monday) |
| `is_weekend` | 1 if Saturday or Sunday |

### Lag Features
Past demand values given directly as features so the model can see recent history:

| Feature | Shift | Purpose |
|---|---|---|
| `demand_now` (lag_0h) | shift(0) | Current hour demand — valid input when predicting T+1 |
| `lag_24h` | shift(24) | Same hour yesterday |
| `lag_168h` | shift(168) | Same hour last week |
| `demand_8760h_ago` | shift(8760) | Same hour one year ago |

**Why these lags:** Electricity demand follows strong daily (24h) and weekly (168h) cycles. The year-ago lag captures long-term seasonal growth trends.

### Rolling Features
| Feature | Window | Purpose |
|---|---|---|
| `rolling_mean_24h` | 24 hours | Average demand over the last 24 hours |
| `rolling_mean_168h` | 168 hours | Average demand over the last week |
| `rolling_mean_8760h_window` | 337 hours centered on 1-year-ago point | Smoothed year-ago baseline to capture long-run trends |

### Grid State Features
Generation source columns (`gas`, `liquid_fuel`, `coal`, `hydro`, `solar`, `wind`, `india_bheramara_hvdc`) are included directly as features. They reflect the current state of the power system at time T.

### Target Variable
```python
demand['target_next_hour_demand'] = demand['demand_mw'].shift(-1)
```
Each row's target is the demand one hour into the future.

---

## Train / Test Split

| Set | Period | Rows |
|---|---|---|
| Training | 2016 – 2023 | ~66,000 |
| Validation | Last 20% of training (shuffle=False) | ~13,000 |
| Test (primary) | 2024 | ~8,700 |
| Test (extended) | 2025 (partial) | ~4,000 |

Data is **never shuffled**. The 80/20 split is internal to the 2016–2023 range:

- **X_train (80%)** — roughly 2016 to mid-2022 — the model actually learns from this
- **X_val (20%)** — roughly mid-2022 to end of 2023 — the model never trains on this

The validation set serves two specific purposes:

1. **Early stopping** — XGBoost checks its performance on X_val after every tree it builds. If the error stops improving for 50 consecutive rounds, training stops. This prevents overfitting on X_train.
2. **Performance check** — after training finishes, MAPE is calculated on X_val to confirm the model generalises well before touching 2024/2025.

2024 and 2025 are never used during training in any way — not even for early stopping. They are the true unseen test.

---

## Model

**XGBoost Regressor** — chosen because it handles tabular data efficiently, is robust to feature scale differences, naturally handles mixed feature types (lags + calendar + weather + economic), and supports early stopping to prevent overfitting.

```
n_estimators        = 1000
learning_rate       = 0.05
max_depth           = 6
subsample           = 0.8
colsample_bytree    = 0.8
early_stopping_rounds = 50
```

Features excluded from training: `datetime`, `demand_mw`, `generation_mw`, `load_shedding`, `remarks`, `year`, `target_next_hour_demand`.

---

## Results

| Period | MAPE |
|---|---|
| Test — 2024 | **4.87%** |

The test MAPE of 4.87% is a genuine out-of-sample result on a full unseen year. This is within the industry benchmark of under 5% for short-term demand forecasting, and reflects a true T+1 forecast with no data leakage.

---

## Feature Importance

Feature importance extracted from the trained XGBoost model shows that the strongest predictors are:

- **Lag features** (`demand_now`, `lag_24h`, `lag_168h`, `demand_8760h_ago`) — dominate importance, confirming that recent and periodic historical demand is the best predictor of future demand.
- **Calendar features** (`hour`, `month`, `day_of_week`) — capture daily and seasonal cycles.
- **Grid state features** (`gas`, `coal`, `liquid_fuel`) — reflect the real-time operational state of the grid.
- **Weather features** (`temp_2m`, `apparent_temp`) — temperature drives cooling and heating load.
- **Economic features** — contribute lower individual importance but provide long-run demand-level context.

---

## Key Design Decisions

**Why XGBoost over Random Forest:**
XGBoost uses gradient boosting which sequentially corrects errors, generally outperforming Random Forest on structured tabular data. It also supports early stopping natively, making overfitting control straightforward.

**Why IQR for outliers:**
Non-parametric, simple, and requires no distributional assumptions. The raw PGCB data contained spikes exceeding 120,000 MW — physically impossible for Bangladesh's grid — making a threshold-based removal necessary and justified.

**Why year-based join for economic data:**
Economic indicators like GDP and population are annual figures. Broadcasting them to every hourly row in the corresponding year is the most logical and leak-free way to incorporate macro-level context into the hourly feature set.

**Why shift(-1) as the target:**
The task requires predicting the *next* hour's demand. Shifting demand_mw by -1 means each row's label is the demand one hour ahead, which is the correct supervised learning formulation.
