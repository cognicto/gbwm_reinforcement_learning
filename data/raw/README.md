# Raw Data Directory

This directory contains original, unprocessed data files used for training and evaluation.

## Structure:

### market_data/
- `sp500_historical.csv`: Historical S&P 500 price and volume data
- `bond_yields.csv`: Treasury bond yield time series
- `international_indices.csv`: International market index data
- `economic_indicators.csv`: Macroeconomic indicators

### portfolio_benchmarks/
- `efficient_frontier_params.json`: Asset class parameters from research literature
- `asset_correlations.csv`: Historical asset correlation matrices

## Data Sources:
- Academic papers (Das et al. 2024)
- Financial databases (Yahoo Finance, FRED, etc.)
- Regulatory filings and reports

## Note:
Raw data should never be modified. All preprocessing is done in separate scripts
that output to the `processed/` directory.
