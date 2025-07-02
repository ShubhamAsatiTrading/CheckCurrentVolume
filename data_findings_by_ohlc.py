import pandas as pd
import os
import glob
from typing import Tuple, Dict, Optional

def read_config() -> Dict[str, str]:
    """Read configuration from common.txt"""
    config = {}
    try:
        with open('common.txt', 'r') as file:
            for line in file:
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
        print(f"Config loaded: OHLC={config.get('ohlc_value')}, Stop Loss={config.get('stop_loss')}%, Target={config.get('target')}%")
    except FileNotFoundError:
        config = {'ohlc_value': 'close', 'stop_loss': '2.5', 'target': '5.0'}
        print("common.txt not found, using defaults")
    return config


def get_file_pairs() -> list:
    """Get matching aggregated and historical file pairs"""
    agg_files = glob.glob("aggregated_data/*_historical_*_aggregated.csv")
    pairs = []
    missing = []
    
    for agg_file in agg_files:
        symbol = os.path.basename(agg_file).split('_')[0]
        hist_file = f"stocks_historical_data/{symbol}_historical.csv"
        
        if os.path.exists(hist_file):
            pairs.append((symbol, agg_file, hist_file))
            print(f" {symbol}: {os.path.basename(agg_file)} ↔ {os.path.basename(hist_file)}")
        else:
            missing.append(symbol)
    
    print(f"\nFound {len(pairs)} pairs, Missing: {missing}")
    return pairs

def display_data(symbol: str, agg_file: str, hist_file: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Read and display first 10 rows of both files"""
    print(f"\n=== {symbol} ===")
    
    try:
        agg_df = pd.read_csv(agg_file)
     
    except Exception as e:
        print(f"Error reading {agg_file}: {e}")
        agg_df = None
    
    try:
        hist_df = pd.read_csv(hist_file)
     
    except Exception as e:
        print(f"Error reading {hist_file}: {e}")
        hist_df = None
    
    return agg_df, hist_df

def process_all_stocks() -> Dict[str, Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]]:
    """Main function to process all stock data"""
    config = read_config()
    pairs = get_file_pairs()
    
    results = {}
    for symbol, agg_file, hist_file in pairs:
        results[symbol] = display_data(symbol, agg_file, hist_file)
    
    print(f"\nProcessed {len(results)} stocks with OHLC: {config.get('ohlc_value')}")
    return results

def get_stock_data(symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Get data for specific symbol"""
    agg_files = glob.glob(f"aggregated_data/{symbol}_historical_*_aggregated.csv")
    if not agg_files:
        print(f"No aggregated file for {symbol}")
        return None, None
    
    hist_file = f"stocks_historical_data/{symbol}_historical.csv"
    if not os.path.exists(hist_file):
        print(f"No historical file for {symbol}")
        return None, None
    
    return display_data(symbol, agg_files[0], hist_file)

if __name__ == "__main__":
    process_all_stocks()

