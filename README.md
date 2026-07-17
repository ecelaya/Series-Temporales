# Time Series Analysis and Forecasting — US Retail Sales (RSXFSN)

Final project for the **Data Mining and Predictive Modelling** module, MSc in Big Data, Data Science & Artificial Intelligence (UCM).

## Objective

Analyse and forecast the monthly retail sales series in the United States (RSXFSN, U.S. Census Bureau / FRED), covering January 2012 to December 2025 (168 observations). Two forecasting families are identified, fitted and compared on a held-out 12-month test set:

- **Exponential smoothing** (Holt-Winters)
- **SARIMA**, following the Box-Jenkins methodology

## Project structure

    .
    ├── Proyecto_Final_RetailSales.ipynb   # Full analysis notebook
    ├── Memoria_ProyectoFinal_RetailSales.pdf   # Full report (Spanish, PDF)
    ├── retail_sales.csv                   # RSXFSN monthly series
    └── requirements.txt                   # Python dependencies

## Methodology

### Decomposition

Seasonal decomposition (`seasonal_decompose`, period = 12) confirms a **multiplicative** structure: seasonal amplitude grows with the level of the series. Trend is increasing across the whole period, with a sharp drop in 2020 (COVID-19) followed by a level shift and accelerated growth in 2021–2025. December shows the highest seasonal coefficient (1.132, +13.2% vs. annual mean); February the lowest (0.887, −11.3%).

### Train / test split

| Set | Observations | Period |
|---|---|---|
| Train | 156 | 2012-01 to 2024-12 |
| Test | 12 | 2025-01 to 2025-12 |

Chronological split (no random shuffling), reserving the full last year as an out-of-sample forecasting horizon.

### Holt-Winters

Three trend/seasonality combinations (additive/multiplicative) are compared by AIC on train and RMSE on test:

| Model | AIC | RMSE (test) | MAE (test) |
|---|---|---|---|
| add-add | 2982.9 | 20.164 | 17.650 |
| add-mul | 2966.8 | 9.000 | 7.749 |
| **mul-mul** | **2965.4** | 11.320 | 9.381 |

Selected model: fully multiplicative Holt-Winters (lowest AIC, consistent with the multiplicative structure observed in the decomposition). Estimated parameters: α = 0.507, β = 0.000, γ = 0.000 — trend and seasonal pattern are essentially stable over the sample.

### SARIMA

Box-Jenkins identification on the log-transformed series (variance-stabilising, λ = 0 Box-Cox):

- ADF/KPSS tests → **d = 1** (regular differencing removes the trend)
- Seasonal differencing → **D = 1** (ADF p = 0.0008 after ∇₁₂)
- ACF/PACF correlograms → MA(1) regular, MA(1) seasonal signature

`auto_arima` (stepwise, AIC) explores 41 candidates and selects:

**SARIMA(0,1,2)(2,1,1)[12]**, AIC = -584.9

| Parameter | Estimate | p-value |
|---|---|---|
| ma.L1 | -0.343 | 0.000 |
| ma.L2 | -0.240 | 0.000 |
| ar.S.L12 | -0.137 | 0.356 |
| ar.S.L24 | -0.184 | 0.226 |
| ma.S.L12 | -0.724 | 0.000 |

Ljung-Box test on residuals: p = 0.76 → no significant autocorrelation, residuals behave as white noise.

## Results

| Method | RMSE | MAE | MAPE | 95% CI valid |
|---|---|---|---|---|
| **Holt-Winters (mul-mul)** | **11.320** | **9.381** | **1.50%** | N/A |
| SARIMA(0,1,2)(2,1,1)[12] | 16.745 | 14.341 | 2.27% | Yes (all 12 obs. inside) |

*(Figures in millions of USD)*

## Key findings

- Holt-Winters mul-mul is the most accurate model on the test set (MAPE 1.50%), with β = γ ≈ 0 reflecting a very stable trend and seasonal pattern over the sample.
- SARIMA(0,1,2)(2,1,1)[12] provides a more rigorous statistical framework — validated residual diagnostics and 95% confidence intervals that contain all 12 observed values in 2025 — at the cost of higher point-forecast error (MAPE 2.27%).
- Both models confirm annual seasonality (period 12) as the dominant structural component, a stable increasing trend, and the 2020 COVID shock as an isolated outlier in the irregular component that neither model could have anticipated from historical data alone.

## Tech stack

Python — pandas, numpy, statsmodels, pmdarima, matplotlib

## Setup

```bash
pip install -r requirements.txt
jupyter notebook Proyecto_Final_RetailSales.ipynb
```

## Author

**Eloy Celaya López**
MSc in Big Data, Data Science & Artificial Intelligence — Universidad Complutense de Madrid

Academic project — Data Mining and Predictive Modelling, MSc NTIC (UCM), 2026–2027
*(Full report available in Spanish — `Memoria_ProyectoFinal_RetailSales.pdf`)*
