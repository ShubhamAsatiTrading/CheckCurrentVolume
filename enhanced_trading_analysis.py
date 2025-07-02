"""
Real Enhanced Trading Analysis - Production Ready
File: enhanced_trading_analysis.py

REAL DATA ONLY:
- Uses actual backtester results
- No simulations or predictions
- Vectorized calculations for performance
- Multi-threaded processing
- Clean Excel export only
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

class RealEnhancedAnalysis:
    """Production-ready analysis using real backtest results only"""
    
    def __init__(self, config_file: str = "common.txt"):
        self.config = self.load_config(config_file)
        self.execution_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("results", exist_ok=True)
        
    def load_config(self, config_file: str) -> Dict:
        """Load configuration efficiently"""
        config = {
            'stop_loss_pct': 2.0,
            'target_pct': 3.0,
            'max_position_size': 10.0,
            'portfolio_value': 100000
        }
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        if '#' in value:
                            value = value.split('#')[0].strip()
                        if key in config:
                            config[key] = float(value)
        
        return config
    
    def run_real_backtest_for_signals(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        REAL DATA: Run actual backtest using real price data
        """
        if signals_df.empty:
            return pd.DataFrame()
        
        from volume_boost_backtester import run_real_backtest
        
        all_results = []
        symbols = signals_df['symbol'].unique()
        
        for symbol in symbols:
            symbol_signals = signals_df[signals_df['symbol'] == symbol]
            
            # Load historical data
            hist_file = f"stocks_historical_data/{symbol}_historical.csv"
            if not os.path.exists(hist_file):
                continue
            
            hist_df = pd.read_csv(hist_file)
            hist_df['date'] = pd.to_datetime(hist_df['date'])
            
            # Prepare signals for backtester
            agg_df = symbol_signals.copy()
            if 'close' not in agg_df.columns and 'price' in agg_df.columns:
                agg_df['close'] = agg_df['price']
            
            agg_df['interval_minutes'] = 0  # Immediate execution
            
            # Run real backtest
            backtester = run_real_backtest(
                agg_df, hist_df,
                self.config['stop_loss_pct'],
                self.config['target_pct'],
                'close', symbol
            )
            
            # Convert results to analysis format
            if backtester.trades:
                results_df = self.convert_backtest_to_analysis(backtester.trades, symbol_signals)
                all_results.append(results_df)
        
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        
        return pd.DataFrame()
    
    def convert_backtest_to_analysis(self, trades: List[Dict], original_signals: pd.DataFrame) -> pd.DataFrame:
        """
        Convert real backtest results to analysis format
        """
        analysis_records = []
        
        for i, trade in enumerate(trades):
            # Calculate position sizing
            execution_price = trade['execution_price']
            position_size_pct = min(
                self.config['max_position_size'],
                (self.config['portfolio_value'] * 0.02) / (execution_price * 100)  # 2% risk per trade
            )
            
            # Calculate portfolio impact
            pnl_pct = trade['pnl_pct']
            portfolio_impact = pnl_pct * (position_size_pct / 100)
            
            # Get signal data
            signal_match = original_signals[
                original_signals['date'] <= trade['execution_time']
            ].iloc[-1] if not original_signals.empty else {}
            
            record = {
                # Core identification
                'trade_id': f"T{i+1:06d}",
                'symbol': trade['symbol'],
                'strategy': 'Real_Conservative_VWAP_RSI',
                
                # Real execution data
                'execution_time': trade['execution_time'],
                'execution_price': trade['execution_price'],
                'direction': 'LONG',  # Assuming LONG trades
                'position_size_pct': position_size_pct,
                
                # Real risk management
                'stop_loss_price': trade['stop_loss_price'],
                'target_price': trade['target_price'],
                'risk_reward_ratio': abs(trade['target_price'] - trade['execution_price']) / 
                                   abs(trade['stop_loss_price'] - trade['execution_price']),
                
                # Real trade outcomes
                'outcome': trade['outcome'],
                'exit_time': trade['exit_time'],
                'exit_price': trade['exit_price'],  # ACTUAL exit price from real data
                'holding_time_minutes': trade['holding_minutes'],
                'holding_time_hours': trade['holding_minutes'] / 60,
                
                # Real performance metrics
                'pnl_pct': pnl_pct,
                'net_pnl_pct': pnl_pct - 0.1,  # 0.1% costs
                'portfolio_impact_pct': portfolio_impact,
                'net_portfolio_impact_pct': portfolio_impact - (0.1 * position_size_pct / 100),
                'is_winner': trade['is_winner'],
                'win_loss_status': 'WIN' if trade['is_winner'] else 'LOSS',
                
                # Signal data
                'signal_strength': signal_match.get('signal_strength', 5),
                'vwap_value': signal_match.get('vwap', execution_price),
                'rsi_value': signal_match.get('rsi', 50),
                'breakout_pct': signal_match.get('breakout_pct', 0),
                
                # Trading context
                'trading_session': self.determine_trading_session(trade['execution_time']),
                'total_cost_pct': 0.1,
                'data_source': 'REAL_BACKTEST'
            }
            
            analysis_records.append(record)
        
        return pd.DataFrame(analysis_records)
    
    def add_portfolio_metrics_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        OPTIMIZED: Add portfolio metrics using vectorized operations
        """
        if df.empty:
            return df
        
        df = df.sort_values('execution_time').copy()
        
        # Vectorized calculations
        df['cumulative_portfolio_pnl'] = df['net_portfolio_impact_pct'].cumsum()
        df['rolling_win_rate'] = df['is_winner'].rolling(window=10, min_periods=1).mean() * 100
        
        # Consecutive wins/losses (vectorized)
        df['win_flag'] = df['is_winner'].astype(int)
        df['loss_flag'] = (~df['is_winner']).astype(int)
        
        # Calculate consecutive streaks efficiently
        df['win_group'] = (df['win_flag'] != df['win_flag'].shift()).cumsum()
        df['loss_group'] = (df['loss_flag'] != df['loss_flag'].shift()).cumsum()
        
        df['consecutive_wins'] = df.groupby('win_group')['win_flag'].cumsum() * df['win_flag']
        df['consecutive_losses'] = df.groupby('loss_group')['loss_flag'].cumsum() * df['loss_flag']
        
        # Portfolio-level statistics
        total_trades = len(df)
        win_rate = df['is_winner'].mean() * 100
        avg_win = df[df['is_winner']]['net_pnl_pct'].mean()
        avg_loss = df[~df['is_winner']]['net_pnl_pct'].mean()
        
        df['portfolio_win_rate_pct'] = win_rate
        df['portfolio_avg_win_pct'] = avg_win
        df['portfolio_avg_loss_pct'] = avg_loss
        
        # Profit factor
        total_wins = df[df['is_winner']]['net_pnl_pct'].sum()
        total_losses = abs(df[~df['is_winner']]['net_pnl_pct'].sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
        df['profit_factor'] = profit_factor
        
        return df
    
    def determine_trading_session(self, timestamp: datetime) -> str:
        """Determine trading session efficiently"""
        hour = timestamp.hour
        if 9 <= hour < 11:
            return "OPENING"
        elif 11 <= hour < 14:
            return "MID_DAY"
        elif 14 <= hour < 16:
            return "CLOSING"
        else:
            return "AFTER_HOURS"
    
    def generate_comprehensive_analysis(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        PRODUCTION: Generate analysis using real backtest results
        """
        if signals_df.empty:
            return pd.DataFrame()
        
        # Run real backtest
        analysis_df = self.run_real_backtest_for_signals(signals_df)
        
        if analysis_df.empty:
            return pd.DataFrame()
        
        # Add portfolio metrics
        analysis_df = self.add_portfolio_metrics_vectorized(analysis_df)
        
        return analysis_df
    
    def export_to_excel(self, analysis_df: pd.DataFrame, filename: str = None) -> str:
        """
        PRODUCTION: Export comprehensive results to Excel
        """
        if analysis_df.empty:
            return ""
        
        if not filename:
            filename = f"results/enhanced_analysis_{self.execution_timestamp}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: Complete Analysis
            analysis_df.to_excel(writer, sheet_name='Complete_Analysis', index=False)
            
            # Sheet 2: Performance Summary
            summary = self.generate_performance_summary(analysis_df)
            summary_df = pd.DataFrame([summary])
            summary_df.to_excel(writer, sheet_name='Performance_Summary', index=False)
            
            # Sheet 3: Symbol Performance
            symbol_perf = analysis_df.groupby('symbol').agg({
                'pnl_pct': ['count', 'sum', 'mean'],
                'is_winner': ['sum', 'mean'],
                'holding_time_hours': 'mean',
                'net_portfolio_impact_pct': 'sum'
            }).round(3)
            symbol_perf.columns = ['trade_count', 'total_pnl', 'avg_pnl', 'wins', 'win_rate', 'avg_hold_hours', 'portfolio_impact']
            symbol_perf.to_excel(writer, sheet_name='Symbol_Performance')
            
            # Sheet 4: Risk Analysis
            risk_analysis = self.generate_risk_analysis(analysis_df)
            risk_df = pd.DataFrame(risk_analysis)
            risk_df.to_excel(writer, sheet_name='Risk_Analysis', index=False)
            
            # Sheet 5: Trade Distribution
            hourly_dist = analysis_df.groupby(analysis_df['execution_time'].dt.hour).agg({
                'pnl_pct': ['count', 'mean'],
                'is_winner': 'mean'
            }).round(3)
            hourly_dist.columns = ['trade_count', 'avg_pnl', 'win_rate']
            hourly_dist.to_excel(writer, sheet_name='Hourly_Distribution')
        
        return filename
    
    def generate_performance_summary(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive performance summary"""
        return {
            'total_trades': len(df),
            'win_rate_pct': df['is_winner'].mean() * 100,
            'total_pnl_pct': df['cumulative_portfolio_pnl'].iloc[-1] if not df.empty else 0,
            'avg_win_pct': df[df['is_winner']]['net_pnl_pct'].mean(),
            'avg_loss_pct': df[~df['is_winner']]['net_pnl_pct'].mean(),
            'profit_factor': df['profit_factor'].iloc[-1] if not df.empty else 0,
            'max_consecutive_wins': df['consecutive_wins'].max(),
            'max_consecutive_losses': df['consecutive_losses'].max(),
            'avg_holding_hours': df['holding_time_hours'].mean(),
            'max_drawdown_pct': (df['cumulative_portfolio_pnl'] - df['cumulative_portfolio_pnl'].cummax()).min(),
            'sharpe_ratio': self.calculate_sharpe_ratio(df),
            'symbols_traded': df['symbol'].nunique(),
            'data_quality': 'REAL_BACKTEST_DATA'
        }
    
    def generate_risk_analysis(self, df: pd.DataFrame) -> List[Dict]:
        """Generate risk analysis metrics"""
        risk_metrics = []
        
        # Daily risk metrics
        df['date'] = pd.to_datetime(df['execution_time']).dt.date
        daily_pnl = df.groupby('date')['net_portfolio_impact_pct'].sum()
        
        risk_metrics.append({
            'metric': 'Daily VaR (95%)',
            'value': np.percentile(daily_pnl, 5),
            'description': '5th percentile of daily returns'
        })
        
        risk_metrics.append({
            'metric': 'Daily Standard Deviation',
            'value': daily_pnl.std(),
            'description': 'Standard deviation of daily returns'
        })
        
        risk_metrics.append({
            'metric': 'Maximum Daily Loss',
            'value': daily_pnl.min(),
            'description': 'Worst single day performance'
        })
        
        risk_metrics.append({
            'metric': 'Average Position Size',
            'value': df['position_size_pct'].mean(),
            'description': 'Average position size as % of portfolio'
        })
        
        return risk_metrics
    
    def calculate_sharpe_ratio(self, df: pd.DataFrame) -> float:
        """Calculate Sharpe ratio"""
        if df.empty:
            return 0
        
        returns = df['net_portfolio_impact_pct'] / 100
        return (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

def run_enhanced_analysis(symbol: str = None) -> str:
    """
    PRODUCTION: Main function for real enhanced analysis
    """
    analyzer = RealEnhancedAnalysis()
    
    # Load signals
    signal_files = []
    if os.path.exists("strategy_signals"):
        signal_files = [f for f in os.listdir("strategy_signals") 
                      if f.endswith('.csv') and ('conservative' in f.lower() or 'FIXED' in f)]
    
    if not signal_files:
        return ""
    
    # Use latest signals file
    latest_file = f"strategy_signals/{max(signal_files)}"
    signals_df = pd.read_csv(latest_file)
    signals_df['date'] = pd.to_datetime(signals_df['date'])
    
    # Filter by symbol if specified
    if symbol:
        signals_df = signals_df[signals_df['symbol'] == symbol]
    
    if signals_df.empty:
        return ""
    
    # Generate real analysis
    analysis_df = analyzer.generate_comprehensive_analysis(signals_df)
    
    if analysis_df.empty:
        return ""
    
    # Export to Excel
    output_file = analyzer.export_to_excel(analysis_df)
    
    return output_file

def run_multi_symbol_analysis(symbols: List[str], max_workers: int = 4) -> str:
    """
    OPTIMIZED: Multi-threaded analysis for multiple symbols
    """
    def analyze_symbol(symbol):
        return run_enhanced_analysis(symbol)
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(analyze_symbol, symbol): symbol for symbol in symbols}
        
        for future in as_completed(future_to_symbol):
            result = future.result()
            if result:
                results.append(result)
    
    return results

if __name__ == "__main__":
    # Production execution
    output_file = run_enhanced_analysis()
    if output_file:
        print(f"Analysis completed: {output_file}")

