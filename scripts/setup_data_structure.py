"""
Setup proper data structure with sample files

This script creates the expected directory structure and sample data files.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from pathlib import Path
from config.base_config import DATA_DIR


def create_data_structure():
    """Create proper data directory structure with sample files"""

    print("🏗️  Setting up data directory structure...")

    # Create subdirectories
    subdirs = [
        "raw/market_data",
        "raw/portfolio_benchmarks",
        "processed/portfolio_parameters",
        "processed/simulation_data",
        "processed/training_datasets",
        "models/production",
        "models/experiments",
        "models/baselines"
    ]

    for subdir in subdirs:
        path = DATA_DIR / subdir
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {subdir}")

    # Create sample data files
    create_sample_raw_data()
    create_sample_processed_data()
    create_readme_files()

    print("\n✅ Data structure setup complete!")


def create_sample_raw_data():
    """Create sample raw data files"""

    raw_dir = DATA_DIR / "raw"

    # 1. Market data (sample historical data)
    print("\n📊 Creating sample market data...")

    # Sample S&P 500 data - ANNUAL frequency for 50 years (1970-2023)
    # This provides enough data for meaningful 16-year sequences
    dates = pd.date_range('1970-12-31', '2023-12-31', freq='Y')  # Annual data
    print(f"  📅 Generating {len(dates)} years of annual market data (1970-2023)")
    
    # More realistic annual return parameters
    annual_mean_return = 0.10  # 10% average annual return
    annual_volatility = 0.16   # 16% annual volatility
    
    # Generate annual returns using GBM
    annual_returns = np.random.normal(annual_mean_return, annual_volatility, len(dates))
    
    # Convert to cumulative prices starting from 100
    initial_price = 100
    prices = [initial_price]
    for i in range(len(annual_returns)):
        prices.append(prices[-1] * (1 + annual_returns[i]))
    
    sp500_data = pd.DataFrame({
        'Date': dates,
        'Close': prices[1:],  # Skip initial price
        'Volume': np.random.randint(1000000, 5000000, len(dates)),
        'Annual_Return': annual_returns  # Store the annual returns
    })
    sp500_data.to_csv(raw_dir / "market_data/sp500_historical.csv", index=False)

    # Sample bond yields - ANNUAL data
    # Generate more realistic bond yield time series
    bond_yields_10y = []
    bond_yields_2y = []
    base_yield_10y = 0.06  # Start at 6%
    base_yield_2y = 0.04   # Start at 4%
    
    for i in range(len(dates)):
        # Add mean reversion and some volatility
        yield_10y = base_yield_10y + np.random.normal(0, 0.01)  # 1% volatility
        yield_2y = base_yield_2y + np.random.normal(0, 0.008)   # 0.8% volatility
        
        # Ensure 10Y > 2Y (normal yield curve)
        if yield_10y < yield_2y:
            yield_10y = yield_2y + 0.005  # At least 0.5% spread
            
        bond_yields_10y.append(yield_10y)
        bond_yields_2y.append(yield_2y)
        
        # Update base rates with some drift
        base_yield_10y += np.random.normal(0, 0.002)  # Small drift
        base_yield_2y += np.random.normal(0, 0.002)
        
        # Keep yields reasonable (between 0% and 15%)
        base_yield_10y = np.clip(base_yield_10y, 0.01, 0.15)
        base_yield_2y = np.clip(base_yield_2y, 0.01, 0.12)

    bond_data = pd.DataFrame({
        'Date': dates,
        '10Y_Treasury': bond_yields_10y,
        '2Y_Treasury': bond_yields_2y
    })
    bond_data.to_csv(raw_dir / "market_data/bond_yields.csv", index=False)

    # 2. Portfolio benchmarks (from the paper)
    portfolio_config = {
        "asset_classes": ["US_Bonds", "US_Stocks", "International_Stocks"],
        "expected_returns": [0.0493, 0.0770, 0.0886],
        "volatilities": [0.0412, 0.1990, 0.1978],
        "correlation_matrix": [
            [1.0, -0.2077, -0.2685],
            [-0.2077, 1.0, 0.7866],
            [-0.2685, 0.7866, 1.0]
        ],
        "source": "Das et al. 2024 - Reinforcement Learning for GBWM"
    }

    with open(raw_dir / "portfolio_benchmarks/efficient_frontier_params.json", 'w') as f:
        json.dump(portfolio_config, f, indent=2)


def create_sample_processed_data():
    """Create sample processed data files"""

    processed_dir = DATA_DIR / "processed"

    print("🔄 Creating processed data files...")

    # 1. Portfolio parameters (from paper)
    portfolio_returns = np.linspace(0.052632, 0.088636, 15)
    portfolio_stds = np.linspace(0.037351, 0.195437, 15)

    np.save(processed_dir / "portfolio_parameters/mean_returns.npy", portfolio_returns)
    np.save(processed_dir / "portfolio_parameters/portfolio_stds.npy", portfolio_stds)

    # Covariance matrix
    correlation_matrix = np.array([
        [1.0, -0.2077, -0.2685],
        [-0.2077, 1.0, 0.7866],
        [-0.2685, 0.7866, 1.0]
    ])
    np.save(processed_dir / "portfolio_parameters/correlation_matrix.npy", correlation_matrix)

    # 2. Efficient frontier configuration
    efficient_frontier = {
        "num_portfolios": 15,
        "portfolios": []
    }

    for i in range(15):
        efficient_frontier["portfolios"].append({
            "id": i,
            "expected_return": float(portfolio_returns[i]),
            "volatility": float(portfolio_stds[i]),
            "risk_level": "Conservative" if i < 5 else "Moderate" if i < 10 else "Aggressive"
        })

    with open(processed_dir / "portfolio_parameters/efficient_frontier.json", 'w') as f:
        json.dump(efficient_frontier, f, indent=2)

    # 3. GBM calibration parameters
    gbm_params = {
        "description": "Geometric Brownian Motion parameters for portfolio simulation",
        "time_step": 1.0,  # Annual
        "portfolios": {
            f"portfolio_{i}": {
                "drift": float(portfolio_returns[i] - 0.5 * portfolio_stds[i] ** 2),
                "volatility": float(portfolio_stds[i])
            }
            for i in range(15)
        }
    }

    with open(processed_dir / "simulation_data/gbm_calibration.json", 'w') as f:
        json.dump(gbm_params, f, indent=2)

    # 4. Standard evaluation scenarios
    evaluation_scenarios = {
        "description": "Standard scenarios for model evaluation",
        "scenarios": [
            {"name": "Young & Poor", "age": 25, "wealth": 50000},
            {"name": "Young & Average", "age": 27, "wealth": 120000},
            {"name": "Young & Wealthy", "age": 28, "wealth": 250000},
            {"name": "Mid-career Struggling", "age": 33, "wealth": 100000},
            {"name": "Mid-career Average", "age": 35, "wealth": 200000},
            {"name": "Mid-career Successful", "age": 36, "wealth": 400000},
            {"name": "Late-career Conservative", "age": 38, "wealth": 300000},
            {"name": "Late-career Wealthy", "age": 40, "wealth": 600000}
        ]
    }

    with open(processed_dir / "training_datasets/evaluation_scenarios.json", 'w') as f:
        json.dump(evaluation_scenarios, f, indent=2)


def create_readme_files():
    """Create README files explaining each directory"""

    # Raw data README
    raw_readme = """# Raw Data Directory

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
"""
    with open(DATA_DIR / "raw/README.md", 'w') as f:
        f.write(raw_readme)

    # Processed data README
    processed_readme = """# Processed Data Directory

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
"""

    with open(DATA_DIR / "processed/README.md", 'w') as f:
        f.write(processed_readme)

    # Models README
    models_readme = """# Models Directory

This directory contains trained model artifacts organized by purpose and version.

## Structure:

### production/
Production-ready models for deployment:
- `gbwm_4goals_production.pth`: 4-goal model (recommended)
- `gbwm_8goals_production.pth`: 8-goal model
- `model_metadata.json`: Model performance metrics and versioning

### experiments/
Research and development models:
- `ablation_studies/`: Models testing different components
- `hyperparameter_sweeps/`: Models from parameter optimization
- `architecture_comparisons/`: Different network architectures

### baselines/
Simple baseline models for comparison:
- `random_policy.pth`: Random action selection
- `greedy_policy.pth`: Greedy goal-taking strategy
- `buy_and_hold_policy.pth`: Buy-and-hold investment strategy

## Model Naming Convention:
`gbwm_{num_goals}goals_{configuration}_{version}.pth`

Example: `gbwm_4goals_ppo_v2.pth`

## Usage:
Load models using the GBWMEvaluator class:
```python
from src.evaluation.evaluator import GBWMEvaluator
evaluator = GBWMEvaluator("data/models/production/gbwm_4goals_production.pth")
    """

    with open(DATA_DIR / "processed/README.md", 'w') as f:
        f.write(processed_readme)


def move_existing_models():
    """Move existing models to proper location"""

    results_dir = DATA_DIR / "results"
    models_dir = DATA_DIR / "models"

    if results_dir.exists():
        print("\n📦 Moving existing models to proper location...")

        # Find model files in results
        for experiment_dir in results_dir.iterdir():
            if experiment_dir.is_dir():
                model_files = list(experiment_dir.glob("*.pth"))

                for model_file in model_files:
                    if "final_model" in model_file.name:
                        # Move to production if it's a final model
                        dest = models_dir / "production" / f"{experiment_dir.name}.pth"
                        print(f"  📦 Moving {model_file.name} → {dest}")
                        # Copy instead of move to preserve original
                        import shutil

                        shutil.copy2(model_file, dest)





if __name__ == "__main__":
    create_data_structure()
    move_existing_models()
    print("\n📋 Summary:")
    print("✅ Created proper data directory structure")
    print("✅ Added sample data files")
    print("✅ Created README documentation")
    print("✅ Moved existing models to proper locations")
    print("\n🎯 Next steps:")
    print("  - Run 'python scripts/setup_data_structure.py' to organize your data")
    print("  - Add real market data to data/raw/ if available")
    print("  - Use data/models/production/ for your best models")
