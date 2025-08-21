"""
Test script for HistoricalDataLoader
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.historical_data_loader import create_historical_loader
import numpy as np

def test_historical_loader():
    """Test the historical data loader"""
    print("Testing HistoricalDataLoader...")
    
    try:
        # Create loader
        loader = create_historical_loader()
        
        # Validate data
        stats = loader.validate_data_quality()
        print("\nData Quality Stats:")
        print(f"Total periods: {stats['total_periods']}")
        print(f"Available 16-year sequences: {stats['available_16y_sequences']}")
        print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        print(f"Missing values: {stats['missing_values']}")
        
        # Test random sequence
        print("\nTesting random sequence generation...")
        sequence = loader.get_random_sequence(length=16)
        print(f"Sequence shape: {sequence.shape}")
        print(f"Portfolio 0 returns (first 5): {sequence[0][:5]}")
        
        # Test portfolio weights
        weights = loader.get_portfolio_weights()
        print(f"\nPortfolio weights shape: {weights.shape}")
        print(f"Portfolio 0 weights: {weights[0]}")
        print(f"Portfolio 14 weights: {weights[14]}")
        
        # Show some portfolio statistics
        print("\nFirst 3 portfolio statistics:")
        for i in range(min(3, len(stats['portfolio_statistics']))):
            p_stats = stats['portfolio_statistics'][i]
            print(f"Portfolio {i}: mean={p_stats['mean_return']:.4f}, std={p_stats['std_return']:.4f}")
        
        print("\n✅ HistoricalDataLoader test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_historical_loader()