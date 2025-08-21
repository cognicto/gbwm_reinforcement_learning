# Processed Data Directory

This directory contains cleaned, preprocessed data ready for model training and evaluation.

## Structure:

### portfolio_parameters/
- `mean_returns.npy`: Expected returns for 15 efficient frontier portfolios
- `portfolio_stds.npy`: Standard deviations for portfolios
- `correlation_matrix.npy`: Asset correlation matrix
- `efficient_frontier.json`: Complete portfolio configuration

### simulation_data/
- `gbm_calibration.json`: Geometric Brownian Motion parameters
- `goal_cost_scenarios.csv`: Different goal cost assumptions

### training_datasets/
- `synthetic_trajectories.pkl`: Pre-generated training trajectories
- `evaluation_scenarios.json`: Standard test scenarios

## Processing Scripts:
Data is processed using scripts in the `scripts/` directory:
- `process_market_data.py`: Clean and normalize market data
- `calibrate_models.py`: Estimate model parameters
- `generate_scenarios.py`: Create evaluation scenarios
