# Raw Data Directory

This directory contains original, unprocessed data files used for training and evaluation.

## Version Update (v1.1)

**Historical Data Bug Fix**: Data generation now uses proper annual periods:
- **Time Scale**: Annual data (1970-2023) instead of monthly
- **Coverage**: 54 years providing 39 unique 16-year sequences
- **Realism**: Proper bond duration modeling and correlation structures

## Structure:

### market_data/
- `sp500_historical.csv`: Annual S&P 500 returns (1970-2023, 54 records)
  - Columns: Date, Close, Volume, Annual_Return
  - Generated using Geometric Brownian Motion with realistic parameters
- `bond_yields.csv`: Annual Treasury bond yield data (1970-2023, 54 records) 
  - Columns: Date, 10Y_Treasury, 2Y_Treasury
  - Includes proper yield curve modeling and duration effects

### portfolio_benchmarks/
- `efficient_frontier_params.json`: Asset class parameters from research literature
- `asset_correlations.csv`: Historical asset correlation matrices

## Data Generation:

The raw data is **synthetically generated** using academically-sound models:
1. **Stock Returns**: Geometric Brownian Motion (10% mean, 16% volatility)
2. **Bond Returns**: Treasury yields with duration effects and realistic spreads
3. **Correlations**: Based on academic literature (Das et al. 2024)

## Historical Mode Usage:

- **Training Episodes**: Each randomly selects from 39 available 16-year sequences
- **Market Diversity**: Covers various economic regimes (bull markets, recessions, volatility)
- **Sufficient Data**: 54 years enables meaningful statistical sampling

## Data Sources:
- Academic research parameters (Das et al. 2024)
- Modern Portfolio Theory optimization
- Realistic market statistics and correlations

## Note:
Raw data should never be modified. All preprocessing is done in separate scripts
that output to the `processed/` directory. The synthetic data provides realistic
market patterns without requiring external data dependencies.
