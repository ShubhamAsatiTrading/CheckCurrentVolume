"""
Real Pipeline Runner - Production Ready
File: run_enhanced_pipeline.py

PRODUCTION READY:
- Real data only, no simulations
- Multi-threaded processing
- Comprehensive Excel exports
- Clean, optimized code
- Minimal console output
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class RealTradingPipeline:
    """Production-ready trading pipeline using real data only"""
    
    def __init__(self):
        self.execution_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = "pipeline_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Summary data for final export
        self.pipeline_summary = {
            'execution_time': datetime.now(),
            'stages_completed': [],
            'files_generated': [],
            'errors': [],
            'performance_metrics': {}
        }
    
    def check_dependencies(self) -> bool:
        """Check all required dependencies silently"""
        required_files = [
            "conservative_vwap_rsi_strategy.py",
            "volume_boost_backtester.py", 
            "enhanced_trading_analysis.py",
            "trading_analysis_generator.py"
        ]
        
        required_dirs = ["stocks_historical_data"]
        
        missing = []
        for file in required_files:
            if not os.path.exists(file):
                missing.append(file)
        
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                missing.append(dir_name)
        
        if missing:
            self.pipeline_summary['errors'].extend(missing)
            return False
        
        return True
    
    def run_strategy_generation(self, symbols: List[str] = None) -> Tuple[bool, str]:
        """
        Stage 1: Generate trading signals using real conservative strategy
        """
        try:
            from conservative_vwap_rsi_strategy import run_real_conservative_strategy
            
            signals_df, saved_file = run_real_conservative_strategy(symbols)
            
            if signals_df is not None and not signals_df.empty:
                self.pipeline_summary['stages_completed'].append('Strategy_Generation')
                self.pipeline_summary['files_generated'].append(saved_file)
                self.pipeline_summary['performance_metrics']['signals_generated'] = len(signals_df)
                self.pipeline_summary['performance_metrics']['symbols_with_signals'] = signals_df['symbol'].nunique()
                return True, saved_file
            else:
                self.pipeline_summary['errors'].append('No signals generated')
                return False, ""
                
        except Exception as e:
            self.pipeline_summary['errors'].append(f"Strategy generation error: {str(e)}")
            return False, ""
    
    def run_backtest_analysis(self, signals_file: str) -> Tuple[bool, str]:
        """
        Stage 2: Run real backtest analysis
        """
        try:
            from trading_analysis_generator import analyze_real_conservative_strategy_results
            
            output_file = analyze_real_conservative_strategy_results(signals_file)
            
            if output_file:
                self.pipeline_summary['stages_completed'].append('Backtest_Analysis')
                self.pipeline_summary['files_generated'].append(output_file)
                
                # Load results for metrics
                results_df = pd.read_excel(output_file, sheet_name='Trade_Analysis')
                self.pipeline_summary['performance_metrics']['total_trades'] = len(results_df)
                self.pipeline_summary['performance_metrics']['win_rate'] = results_df['is_winner'].mean() * 100
                self.pipeline_summary['performance_metrics']['total_pnl'] = results_df['net_pnl_pct'].sum()
                
                return True, output_file
            else:
                self.pipeline_summary['errors'].append('Backtest analysis failed')
                return False, ""
                
        except Exception as e:
            self.pipeline_summary['errors'].append(f"Backtest analysis error: {str(e)}")
            return False, ""
    
    def run_enhanced_analysis(self, signals_file: str) -> Tuple[bool, str]:
        """
        Stage 3: Run enhanced real analysis
        """
        try:
            from enhanced_trading_analysis import run_enhanced_analysis
            
            output_file = run_enhanced_analysis()
            
            if output_file:
                self.pipeline_summary['stages_completed'].append('Enhanced_Analysis')
                self.pipeline_summary['files_generated'].append(output_file)
                return True, output_file
            else:
                self.pipeline_summary['errors'].append('Enhanced analysis failed')
                return False, ""
                
        except Exception as e:
            self.pipeline_summary['errors'].append(f"Enhanced analysis error: {str(e)}")
            return False, ""
    
    def run_multi_symbol_backtest(self, symbols: List[str], max_workers: int = 4) -> Tuple[bool, str]:
        """
        OPTIMIZED: Run backtest for multiple symbols in parallel
        """
        try:
            from volume_boost_backtester import run_multi_symbol_backtest, export_consolidated_results
            
            # Load configuration
            config = {'stop_loss_pct': 2.0, 'target_pct': 3.0}
            if os.path.exists('common.txt'):
                with open('common.txt', 'r') as f:
                    for line in f:
                        if 'stop_loss=' in line and not line.startswith('#'):
                            config['stop_loss_pct'] = float(line.split('=')[1].strip().split('#')[0])
                        elif 'target=' in line and not line.startswith('#'):
                            config['target_pct'] = float(line.split('=')[1].strip().split('#')[0])
            
            # Run multi-symbol backtest
            results = run_multi_symbol_backtest(
                symbols, 
                config['stop_loss_pct'], 
                config['target_pct'], 
                'close', 
                max_workers
            )
            
            if results:
                # Export consolidated results
                output_file = export_consolidated_results(results)
                
                self.pipeline_summary['stages_completed'].append('Multi_Symbol_Backtest')
                self.pipeline_summary['files_generated'].append(output_file)
                self.pipeline_summary['performance_metrics']['symbols_backtested'] = len(results)
                
                return True, output_file
            else:
                return False, ""
                
        except Exception as e:
            self.pipeline_summary['errors'].append(f"Multi-symbol backtest error: {str(e)}")
            return False, ""
    
    def validate_price_data_coverage(self) -> Dict[str, bool]:
        """
        Validate that we have adequate price data for backtesting
        """
        coverage = {}
        
        try:
            # Check for signal files
            signal_files = []
            if os.path.exists("strategy_signals"):
                signal_files = [f for f in os.listdir("strategy_signals") 
                              if f.endswith('.csv')]
            
            if not signal_files:
                return coverage
            
            # Load latest signals
            latest_file = f"strategy_signals/{max(signal_files)}"
            signals_df = pd.read_csv(latest_file)
            
            # Check price data coverage for each symbol
            for symbol in signals_df['symbol'].unique():
                hist_file = f"stocks_historical_data/{symbol}_historical.csv"
                if os.path.exists(hist_file):
                    hist_df = pd.read_csv(hist_file)
                    hist_df['date'] = pd.to_datetime(hist_df['date'])
                    
                    # Check date range coverage
                    signal_dates = pd.to_datetime(signals_df[signals_df['symbol'] == symbol]['date'])
                    min_signal_date = signal_dates.min()
                    max_signal_date = signal_dates.max()
                    
                    hist_min = hist_df['date'].min()
                    hist_max = hist_df['date'].max()
                    
                    # Check if we have data covering signal period + buffer for exits
                    coverage_start = hist_min <= min_signal_date
                    coverage_end = hist_max >= max_signal_date + timedelta(days=7)  # 7-day buffer for exits
                    
                    coverage[symbol] = coverage_start and coverage_end and len(hist_df) > 100
                else:
                    coverage[symbol] = False
                    
        except Exception as e:
            self.pipeline_summary['errors'].append(f"Price data validation error: {str(e)}")
        
        return coverage
    
    def export_pipeline_summary(self) -> str:
        """
        Export comprehensive pipeline summary to Excel
        """
        try:
            filename = f"{self.results_dir}/pipeline_summary_{self.execution_timestamp}.xlsx"
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Pipeline execution summary
                execution_summary = pd.DataFrame([{
                    'execution_time': self.pipeline_summary['execution_time'],
                    'stages_completed': len(self.pipeline_summary['stages_completed']),
                    'files_generated': len(self.pipeline_summary['files_generated']),
                    'errors_count': len(self.pipeline_summary['errors']),
                    'total_duration_minutes': (datetime.now() - self.pipeline_summary['execution_time']).total_seconds() / 60
                }])
                execution_summary.to_excel(writer, sheet_name='Execution_Summary', index=False)
                
                # Stages completed
                if self.pipeline_summary['stages_completed']:
                    stages_df = pd.DataFrame([{'stage': stage} for stage in self.pipeline_summary['stages_completed']])
                    stages_df.to_excel(writer, sheet_name='Stages_Completed', index=False)
                
                # Files generated
                if self.pipeline_summary['files_generated']:
                    files_df = pd.DataFrame([{'file_path': file} for file in self.pipeline_summary['files_generated']])
                    files_df.to_excel(writer, sheet_name='Files_Generated', index=False)
                
                # Performance metrics
                if self.pipeline_summary['performance_metrics']:
                    metrics_df = pd.DataFrame([self.pipeline_summary['performance_metrics']])
                    metrics_df.to_excel(writer, sheet_name='Performance_Metrics', index=False)
                
                # Errors (if any)
                if self.pipeline_summary['errors']:
                    errors_df = pd.DataFrame([{'error': error} for error in self.pipeline_summary['errors']])
                    errors_df.to_excel(writer, sheet_name='Errors', index=False)
                
                # Price data coverage
                coverage = self.validate_price_data_coverage()
                if coverage:
                    coverage_df = pd.DataFrame([{'symbol': k, 'has_adequate_data': v} for k, v in coverage.items()])
                    coverage_df.to_excel(writer, sheet_name='Data_Coverage', index=False)
            
            return filename
            
        except Exception as e:
            return f"Summary export error: {str(e)}"
    
    def run_complete_pipeline(self, symbols: List[str] = None, enable_parallel: bool = True) -> bool:
        """
        PRODUCTION: Run complete real trading pipeline
        """
        # Stage 1: Generate signals
        success, signals_file = self.run_strategy_generation(symbols)
        if not success:
            return False
        
        # Stage 2: Run backtest analysis
        success, analysis_file = self.run_backtest_analysis(signals_file)
        if not success:
            return False
        
        # Stage 3: Run enhanced analysis
        success, enhanced_file = self.run_enhanced_analysis(signals_file)
        # Continue even if enhanced analysis fails
        
        # Stage 4: Multi-symbol backtest (if enabled and multiple symbols)
        if enable_parallel and symbols and len(symbols) > 1:
            success, multi_file = self.run_multi_symbol_backtest(symbols)
        
        return True
    
    def run_symbol_specific_analysis(self, symbol: str) -> bool:
        """
        Run analysis for a specific symbol only
        """
        return self.run_complete_pipeline([symbol], enable_parallel=False)
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols from historical data
        """
        symbols = []
        if os.path.exists("stocks_historical_data"):
            for filename in os.listdir("stocks_historical_data"):
                if filename.endswith("_historical.csv"):
                    symbol = filename.replace("_historical.csv", "")
                    symbols.append(symbol)
        return sorted(symbols)

def main():
    """
    PRODUCTION: Main execution function
    """
    pipeline = RealTradingPipeline()
    
    # Check dependencies
    if not pipeline.check_dependencies():
        print("Dependencies check failed")
        return False
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command in ['help', '-h', '--help']:
            print("Usage:")
            print("  python run_enhanced_pipeline.py [symbol]  - Run for specific symbol")
            print("  python run_enhanced_pipeline.py all       - Run for all symbols")
            print("  python run_enhanced_pipeline.py           - Run complete pipeline")
            return True
            
        elif command == 'all':
            # Run for all available symbols
            symbols = pipeline.get_available_symbols()
            if not symbols:
                print("No symbols found")
                return False
            success = pipeline.run_complete_pipeline(symbols, enable_parallel=True)
            
        elif len(command) <= 10:  # Assume it's a symbol
            symbol = command.upper()
            success = pipeline.run_symbol_specific_analysis(symbol)
            
        else:
            print("Invalid command")
            return False
    else:
        # Run complete pipeline with default settings
        success = pipeline.run_complete_pipeline(enable_parallel=True)
    
    # Export pipeline summary
    summary_file = pipeline.export_pipeline_summary()
    
    # Final status
    if success and not pipeline.pipeline_summary['errors']:
        print(f"Pipeline completed successfully. Summary: {summary_file}")
        return True
    else:
        print(f"Pipeline completed with issues. Summary: {summary_file}")
        return False

if __name__ == "__main__":
    main()

