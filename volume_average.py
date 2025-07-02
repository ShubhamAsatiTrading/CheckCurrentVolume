# volume_average.py - Volume Average Analysis Module
# Calculates average volume and VWAP for stocks over specified duration

import pandas as pd
import os
import glob
from datetime import datetime, timedelta

class Logger:
    """Simple logger for volume average operations"""
    
    @staticmethod
    def log_to_file(message, level="INFO"):
        """Log message to file with timestamp"""
        try:
            log_file = f"trading_system_logs_{datetime.now().strftime('%Y%m%d')}.log"
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"[{timestamp}] {level}: {message}\n"
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Failed to write to log file: {e}")

class ConfigManager:
    """Simple config manager for volume average module"""
    
    @staticmethod
    def load():
        """Load configuration from common.txt"""
        defaults = {
            "stop_loss": 4.0, 
            "target": 10.0, 
            "ohlc_value": "open", 
            "trade_today_flag": "no", 
            "check_from_date": "2020-03-28",
            "avg_volume_days": 30
        }
        
        config_file = "common.txt"
        
        try:
            config = {}
            with open(config_file, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        # Convert numeric values
                        if key in ['stop_loss', 'target', 'avg_volume_days']:
                            config[key] = float(value) if '.' in value else int(value)
                        else:
                            config[key] = value
            
            Logger.log_to_file(f"Configuration loaded: {config}")
            return {**defaults, **config}
        except Exception as e:
            Logger.log_to_file(f"Error loading config, using defaults: {e}", "WARNING")
            return defaults

class VolumeAverage:
    """Volume Average and VWAP Calculation Class"""
    
    @staticmethod
    def calculate_average_volume_data():
        """Calculate average volume data for all stocks based on configured duration"""
        try:
            Logger.log_to_file("=== STARTING VOLUME AVERAGE CALCULATION ===")
            
            # Step 1: Load configuration
            config = ConfigManager.load()
            duration_days = int(config.get('avg_volume_days', 30))
            Logger.log_to_file(f"Duration configured: {duration_days} weekdays")
            
            # Step 2: Setup directories
            input_folder = "stocks_historical_data"
            output_folder = "Average data"
            
            if not os.path.exists(input_folder):
                error_msg = f"Input folder '{input_folder}' not found"
                Logger.log_to_file(error_msg, "ERROR")
                return False, error_msg
            
            # Create output folder if needed
            os.makedirs(output_folder, exist_ok=True)
            
            # Step 3: Find all historical files
            pattern = os.path.join(input_folder, "*_historical.csv")
            historical_files = glob.glob(pattern)
            
            if not historical_files:
                error_msg = f"No historical files found in {input_folder}"
                Logger.log_to_file(error_msg, "ERROR")
                return False, error_msg
            
            Logger.log_to_file(f"Found {len(historical_files)} historical files")
            
            # Step 4: Process each stock
            results = []
            processed_count = 0
            error_count = 0
            
            for file_path in historical_files:
                try:
                    # Extract stock name from filename
                    filename = os.path.basename(file_path)
                    stock_symbol = filename.replace("_historical.csv", "")
                    
                    # Process stock data
                    result = VolumeAverage._process_stock_data(file_path, stock_symbol, duration_days)
                    
                    if result:
                        results.append(result)
                        processed_count += 1
                        Logger.log_to_file(f"✅ Processed {stock_symbol}")
                    else:
                        error_count += 1
                        Logger.log_to_file(f"❌ Failed to process {stock_symbol}", "WARNING")
                        
                except Exception as e:
                    error_count += 1
                    Logger.log_to_file(f"❌ Error processing {file_path}: {e}", "ERROR")
            
            if not results:
                error_msg = "No valid data processed from any stock"
                Logger.log_to_file(error_msg, "ERROR")
                return False, error_msg
            
            # Step 5: Create output DataFrame
            df_results = pd.DataFrame(results)
            
            # Step 6: Generate filename with end date only
            if results:
                # Use end date from first stock (assuming all have similar ranges)
                end_date = results[0]['end_date']
                filename = VolumeAverage._generate_filename(end_date)
                output_path = os.path.join(output_folder, filename)
                
                # Step 7: Remove existing file if same duration
                if os.path.exists(output_path):
                    os.remove(output_path)
                    Logger.log_to_file(f"Removed existing file: {filename}")
                
                # Step 8: Save results
                df_results.to_csv(output_path, index=False)
                
                Logger.log_to_file(f"✅ VOLUME AVERAGE CALCULATION COMPLETED")
                Logger.log_to_file(f"📊 Processed: {processed_count} stocks, Errors: {error_count}")
                Logger.log_to_file(f"💾 Output file: {output_path}")
                
                return True, f"✅ Volume average data saved: {filename} ({processed_count} stocks processed)"
            
        except Exception as e:
            error_msg = f"Critical error in volume average calculation: {e}"
            Logger.log_to_file(error_msg, "ERROR")
            return False, error_msg
    
    @staticmethod
    def _process_stock_data(file_path, stock_symbol, duration_days):
        """Process individual stock data and calculate averages"""
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_cols = ['date', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                Logger.log_to_file(f"Missing columns in {stock_symbol}: {missing_cols}", "WARNING")
                return None
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date'])
            df['date_only'] = df['date'].dt.date
            
            # Sort by date to get latest data first
            df = df.sort_values('date', ascending=False)
            
            # Get latest date from data
            latest_date = df['date_only'].iloc[0]
            
            # Calculate weekdays back
            start_date = VolumeAverage._get_weekdays_back(latest_date, duration_days)
            
            # Filter data for the duration
            df_filtered = df[df['date_only'] >= start_date].copy()
            
            if df_filtered.empty:
                Logger.log_to_file(f"No data available for {stock_symbol} in date range", "WARNING")
                return None
            
            # Group by date and calculate daily metrics
            daily_data = df_filtered.groupby('date_only').agg({
                'close': ['mean', 'last'],  # Average close and last close of day
                'volume': 'sum'  # Total volume for the day
            }).reset_index()
            
            # Flatten column names
            daily_data.columns = ['date', 'avg_close', 'last_close', 'total_volume']
            
            # Calculate Daily VWAP for each day
            daily_vwaps = []
            for date_val in daily_data['date']:
                day_data = df_filtered[df_filtered['date_only'] == date_val]
                if not day_data.empty:
                    # VWAP = sum(close * volume) / sum(volume)
                    vwap = (day_data['close'] * day_data['volume']).sum() / day_data['volume'].sum()
                    daily_vwaps.append(vwap)
                else:
                    daily_vwaps.append(0)
            
            daily_data['daily_vwap'] = daily_vwaps
            
            # Calculate final averages
            avg_close_price = daily_data['last_close'].mean()  # Average of daily closing prices
            avg_volume = daily_data['total_volume'].mean()  # Average of daily volumes
            daily_vwap_average = daily_data['daily_vwap'].mean()  # Average of daily VWAPs
            
            # Calculate Overall VWAP for entire duration
            total_volume_all = df_filtered['volume'].sum()
            total_value_all = (df_filtered['close'] * df_filtered['volume']).sum()
            overall_vwap = total_value_all / total_volume_all if total_volume_all > 0 else 0
            
            return {
                'Symbol': stock_symbol,
                'Yest_Average_Close_Price': round(avg_close_price, 2),
                'Yest_Average_Volume': round(avg_volume, 2),
                'Yest_Daily_VWAP_Average': round(daily_vwap_average, 2),
                'Yest_Overall_VWAP': round(overall_vwap, 2),
                'start_date': start_date,
                'end_date': latest_date,
                'days_processed': len(daily_data)
            }
            
        except Exception as e:
            Logger.log_to_file(f"Error processing {stock_symbol}: {e}", "ERROR")
            return None
    
    @staticmethod
    def _get_weekdays_back(end_date, weekdays_count):
        """Calculate start date by going back specified weekdays"""
        current_date = end_date
        weekdays_found = 0
        
        while weekdays_found < weekdays_count:
            current_date = current_date - timedelta(days=1)
            # Check if it's a weekday (Monday=0, Sunday=6)
            if current_date.weekday() < 5:  # Monday to Friday
                weekdays_found += 1
        
        return current_date
    
    @staticmethod
    def _generate_filename(end_date):
        """Generate filename in format: AvgData_till_02Jul2025.csv"""
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        end_formatted = f"{end_date.day:02d}{months[end_date.month-1]}{end_date.year}"
        
        return f"AvgData_till_{end_formatted}.csv"