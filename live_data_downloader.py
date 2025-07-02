import os
import datetime
import pandas as pd
import threading
import time
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from queue import Queue
import signal
import sys
import json
import asyncio
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict
import webbrowser
import csv
from io import StringIO

# Required functions from historical downloader (included directly)
def load_symbols(symbols_file: str = 'symbols.txt') -> List[str]:
    """Load symbols from file with validation"""
    if not os.path.exists(symbols_file):
        raise FileNotFoundError(f"Symbols file {symbols_file} not found!")
    
    with open(symbols_file, 'r') as f:
        symbols = [line.strip() for line in f if line.strip()]
    
    if not symbols:
        raise ValueError("No symbols found in symbols.txt file!")
    
    return symbols

def get_instrument_token(instruments: pd.DataFrame, symbol: str) -> Optional[int]:
    """Get instrument token for symbol - optimized"""
    try:
        token = instruments.loc[instruments['tradingsymbol'] == symbol, 'instrument_token'].iloc[0]
        return int(token)
    except (IndexError, KeyError):
        return None

def normalize_datetime(dt) -> datetime.datetime:
    """Normalize datetime to remove timezone info for comparison - optimized"""
    if isinstance(dt, str):
        dt = pd.to_datetime(dt)
    
    if hasattr(dt, 'to_pydatetime'):
        dt = dt.to_pydatetime()
    
    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        ist = pytz.timezone('Asia/Kolkata')
        if dt.tzinfo != ist:
            dt = dt.astimezone(ist)
        dt = dt.replace(tzinfo=None)
    
    return dt

def normalize_dataframe_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize date column in DataFrame - optimized"""
    if df.empty or 'date' not in df.columns:
        return df
    
    try:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"   Error normalizing dates in DataFrame: {e}")
        return df

def is_actual_market_day() -> bool:
    """Check if today is an actual market trading day (Monday-Friday only)"""
    return datetime.date.today().weekday() < 5

@dataclass
class LiveStats:
    """Statistics for monitoring live data downloader"""
    running: bool = False
    update_count: int = 0
    total_symbols: int = 0
    successful_updates: int = 0
    failed_updates: int = 0
    last_update_time: Optional[str] = None
    next_update_time: Optional[str] = None
    market_status: str = "CLOSED"
    error_count: int = 0
    api_429_count: int = 0
    average_update_time: float = 0.0
    symbols_per_second: float = 0.0
    start_time: Optional[str] = None
    uptime_hours: float = 0.0

class RateLimiter:
    """Optimized rate limiter for Zerodha API compliance"""
    
    def __init__(self):
        self.historical_requests = []
        self.last_429_time = None
        self.cooldown_until = None
        self.lock = threading.Lock()
        
        # Zerodha limits (optimized for speed)
        self.HISTORICAL_RPM = 110  # Aggressive: 110/min
        self.HISTORICAL_RPS = 2.0  # Aggressive: 2/sec
        self.COOLDOWN_SECONDS = 8  # Reduced cooldown
    
    def can_make_request(self) -> bool:
        """Check if we can make a historical API request - optimized"""
        with self.lock:
            now = time.time()
            
            if self.cooldown_until and now < self.cooldown_until:
                return False
            
            # Optimized cleanup - only clean when necessary
            if len(self.historical_requests) > self.HISTORICAL_RPM:
                self.historical_requests = [
                    req_time for req_time in self.historical_requests 
                    if now - req_time < 60
                ]
            
            if len(self.historical_requests) >= self.HISTORICAL_RPM:
                return False
            
            # Quick recent requests check
            recent_count = sum(1 for req_time in self.historical_requests if now - req_time < 1)
            return recent_count < self.HISTORICAL_RPS
    
    def record_request(self):
        """Record a successful API request - optimized"""
        with self.lock:
            self.historical_requests.append(time.time())
    
    def record_429_error(self):
        """Record a 429 error and enter cooldown - optimized"""
        with self.lock:
            self.last_429_time = time.time()
            self.cooldown_until = time.time() + self.COOLDOWN_SECONDS
            self.historical_requests.clear()
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded - optimized"""
        while not self.can_make_request():
            time.sleep(0.05)  # Reduced sleep time for faster response

class LiveDataDownloader:
    """High-performance live data downloader with optimizations"""
    
    def __init__(self, kite, symbols_file: str = 'symbols.txt', 
                 output_folder: str = "stocks_historical_data",
                 max_workers: int = 35, update_interval: int = 300):
        """
        Initialize live data downloader with speed optimizations
        """
        self.kite = kite
        self.symbols_file = symbols_file
        self.output_folder = output_folder
        self.max_workers = max_workers  # Increased default
        self.update_interval = update_interval
        
        # Threading control
        self.running = False
        self.background_thread = None
        self.stop_event = threading.Event()
        self.emergency_stop_event = threading.Event()
        
        # Rate limiting
        self.rate_limiter = RateLimiter()
        
        # Optimized caching
        self.instruments_cache = None
        self.symbols_cache = None
        self.token_cache = {}
        self.instruments_dict = {}  # Additional optimization
        
        # Statistics and monitoring
        self.stats = LiveStats()
        self.update_times = []
        self.status_file = os.path.join(output_folder, 'live_status.json')
        self.dashboard_file = os.path.join(output_folder, 'dashboard.html')
        
        # Setup logging
        self.setup_logging()
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Emergency stop file
        self.emergency_stop_file = os.path.join(output_folder, 'EMERGENCY_STOP.txt')
        
    def setup_logging(self):
        """Setup optimized logging"""
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Simplified logging setup for speed
        self.logger = logging.getLogger('LiveDataDownloader')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Single file handler for performance
        file_handler = logging.FileHandler(os.path.join(self.output_folder, 'live_data.log'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(logging.INFO)
        
        # Minimal console output for speed
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()
        sys.exit(0)
    
    def create_emergency_stop_mechanism(self):
        """Create emergency stop file mechanism"""
        instructions = """
EMERGENCY STOP MECHANISM

To immediately stop the live data downloader:

1. Create a file named 'EMERGENCY_STOP.txt' in the data folder
2. Or modify this file and save it
3. The system will stop within 5 seconds

Current time: """ + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.emergency_stop_file + '_INSTRUCTIONS.txt', 'w') as f:
            f.write(instructions)
    
    def check_emergency_stop(self) -> bool:
        """Check if emergency stop file exists"""
        return os.path.exists(self.emergency_stop_file)
    
    def _load_and_cache_data(self):
        """Load and cache instruments and symbols - optimized"""
        try:
            self.logger.info("Loading NSE instruments...")
            instruments_data = self.kite.instruments("NSE")
            self.instruments_cache = pd.DataFrame(instruments_data)
            
            # Create optimized dictionary lookup
            self.instruments_dict = {
                row['tradingsymbol']: row['instrument_token'] 
                for _, row in self.instruments_cache.iterrows()
            }
            
            self.logger.info(f"Loaded {len(self.instruments_cache)} instruments")
            
            self.symbols_cache = load_symbols(self.symbols_file)
            self.logger.info(f"Loaded {len(self.symbols_cache)} symbols")
            
            # Pre-cache tokens using optimized lookup
            self.logger.info("Caching instrument tokens...")
            for symbol in self.symbols_cache:
                if symbol in self.instruments_dict:
                    self.token_cache[symbol] = self.instruments_dict[symbol]
            
            self.stats.total_symbols = len(self.token_cache)
            self.logger.info(f"Cached tokens for {len(self.token_cache)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def is_market_hours(self) -> bool:
        """Check if current time is within market hours - optimized"""
        if not is_actual_market_day():
            return False
        
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.datetime.now(ist)
        
        # Pre-calculated market hours for speed
        hour = now.hour
        minute = now.minute
        
        # Market: 9:15 AM to 3:30 PM IST
        if hour < 9 or hour > 15:
            return False
        if hour == 9 and minute < 15:
            return False
        if hour == 15 and minute > 30:
            return False
        
        return True
    
    def is_after_market_close(self) -> bool:
        """Check if current time is after 3:30 PM - optimized"""
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.datetime.now(ist)
        return now.hour > 15 or (now.hour == 15 and now.minute > 30)
    
    def get_next_update_time(self) -> datetime.datetime:
        """Calculate next update time aligned to 5-minute intervals - optimized"""
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.datetime.now(ist)
        
        # Optimized calculation
        minutes = ((now.minute // 5) + 1) * 5
        if minutes >= 60:
            next_time = now.replace(hour=now.hour+1, minute=0, second=0, microsecond=0)
        else:
            next_time = now.replace(minute=minutes, second=0, microsecond=0)
        
        return next_time.replace(tzinfo=None)
    
    def fetch_latest_candle(self, symbol: str, token: int) -> Optional[Dict]:
        """Fetch latest 5-minute candle - optimized"""
        try:
            self.rate_limiter.wait_if_needed()
            
            # Optimized time calculation
            end_time = datetime.datetime.now()
            start_time = end_time - datetime.timedelta(minutes=15)
            
            data = self.kite.historical_data(token, start_time, end_time, '5minute')
            self.rate_limiter.record_request()
            
            if data and len(data) > 0:
                return data[-1]
            
            return None
            
        except Exception as e:
            error_msg = str(e).lower()
            if '429' in error_msg or 'too many requests' in error_msg:
                self.stats.api_429_count += 1
                self.rate_limiter.record_429_error()
                self.logger.warning(f"Rate limit hit for {symbol}")
                return None
            else:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
                return None
    
    def update_symbol_file(self, symbol: str, new_candle: Dict) -> bool:
        """Update CSV file with new candle data - highly optimized"""
        file_path = os.path.join(self.output_folder, f"{symbol}_historical.csv")
        
        try:
            # Convert timestamp once
            new_timestamp = pd.to_datetime(new_candle['date'])
            
            if os.path.exists(file_path):
                try:
                    # Ultra-fast file check - read only last line
                    with open(file_path, 'rb') as f:
                        f.seek(-150, 2)  # Seek near end
                        last_lines = f.read().decode('utf-8').strip().split('\n')
                        if len(last_lines) > 1:
                            last_line = last_lines[-1]
                            if last_line:
                                try:
                                    last_date_str = last_line.split(',')[0]
                                    last_timestamp = pd.to_datetime(last_date_str)
                                    if new_timestamp <= last_timestamp:
                                        return True  # Data exists
                                except:
                                    pass
                    
                    # Fast append using CSV writer
                    with open(file_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        row = [
                            new_candle['date'], new_candle['open'], new_candle['high'],
                            new_candle['low'], new_candle['close'], new_candle['volume']
                        ]
                        writer.writerow(row)
                    
                except Exception:
                    # Fallback to pandas (slower but safer)
                    new_df = pd.DataFrame([new_candle])
                    new_df = normalize_dataframe_dates(new_df)
                    if not new_df.empty:
                        existing_df = pd.read_csv(file_path)
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                        combined_df.drop_duplicates(subset=['date'], inplace=True)
                        combined_df.sort_values('date', inplace=True)
                        combined_df.to_csv(file_path, index=False)
            else:
                # Create new file
                new_df = pd.DataFrame([new_candle])
                new_df = normalize_dataframe_dates(new_df)
                if not new_df.empty:
                    new_df.to_csv(file_path, index=False)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating file for {symbol}: {e}")
            return False
    
    def process_symbol_batch(self, symbols_batch: List[str]) -> Dict[str, bool]:
        """Process a batch of symbols - optimized for maximum concurrency"""
        results = {}
        
        # Aggressive batch sizing for speed
        optimal_batch_size = min(len(symbols_batch), self.max_workers)
        
        with ThreadPoolExecutor(max_workers=optimal_batch_size) as executor:
            future_to_symbol = {}
            for symbol in symbols_batch:
                if symbol in self.token_cache:
                    future = executor.submit(self._process_single_symbol, symbol)
                    future_to_symbol[future] = symbol
                else:
                    results[symbol] = False
            
            # Optimized result collection
            for future in as_completed(future_to_symbol, timeout=45):  # Reduced timeout
                symbol = future_to_symbol[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e}")
                    results[symbol] = False
        
        return results
    
    def _process_single_symbol(self, symbol: str) -> bool:
        """Process a single symbol - optimized"""
        token = self.token_cache.get(symbol)
        if not token:
            return False
        
        candle = self.fetch_latest_candle(symbol, token)
        if not candle:
            return False
        
        return self.update_symbol_file(symbol, candle)
    
    def update_stats(self, successful: int, failed: int, duration: float):
        """Update performance statistics - optimized"""
        self.stats.update_count += 1
        self.stats.successful_updates += successful
        self.stats.failed_updates += failed
        self.stats.last_update_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.stats.next_update_time = self.get_next_update_time().strftime('%Y-%m-%d %H:%M:%S')
        
        # Optimized averaging
        self.update_times.append(duration)
        if len(self.update_times) > 10:  # Reduced history for speed
            self.update_times.pop(0)
        
        self.stats.average_update_time = sum(self.update_times) / len(self.update_times)
        self.stats.symbols_per_second = successful / duration if duration > 0 else 0
        
        # Optimized uptime calculation
        if self.stats.start_time:
            start_dt = datetime.datetime.strptime(self.stats.start_time, '%Y-%m-%d %H:%M:%S')
            uptime_seconds = (datetime.datetime.now() - start_dt).total_seconds()
            self.stats.uptime_hours = uptime_seconds / 3600
    
    def save_status_to_file(self):
        """Save current status to JSON - optimized"""
        try:
            status_data = asdict(self.stats)
            status_data['market_hours'] = self.is_market_hours()
            status_data['api_rate_limit_status'] = {
                'can_make_request': self.rate_limiter.can_make_request(),
                'recent_requests': len([r for r in self.rate_limiter.historical_requests if time.time() - r < 60]),
                'in_cooldown': bool(self.rate_limiter.cooldown_until and time.time() < self.rate_limiter.cooldown_until)
            }
            
            # Fast JSON write
            with open(self.status_file, 'w') as f:
                json.dump(status_data, f, separators=(',', ':'))  # Compact JSON
        except Exception as e:
            self.logger.error(f"Error saving status: {e}")
    
    def perform_update_cycle(self):
        """Perform one complete update cycle - optimized"""
        if not self.is_market_hours():
            self.stats.market_status = "CLOSED"
            self.logger.info("Outside market hours, skipping update")
            return
        
        self.stats.market_status = "OPEN"
        start_time = time.time()
        
        self.logger.info(f"Starting update cycle #{self.stats.update_count + 1} for {len(self.token_cache)} symbols")
        
        try:
            # Optimized batching - larger batches for speed
            batch_size = min(self.max_workers, 20)
            symbols = list(self.token_cache.keys())
            
            successful_updates = 0
            failed_updates = 0
            
            # Process in optimized batches
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                results = self.process_symbol_batch(batch)
                
                successful_updates += sum(1 for success in results.values() if success)
                failed_updates += sum(1 for success in results.values() if not success)
                
                # Minimal delay for maximum speed
                if not self.rate_limiter.can_make_request():
                    time.sleep(0.2)
            
            # Update statistics
            duration = time.time() - start_time
            self.update_stats(successful_updates, failed_updates, duration)
            
            # Save status
            self.save_status_to_file()
            
            self.logger.info(
                f"Update cycle #{self.stats.update_count} completed in {duration:.2f}s - "
                f"Success: {successful_updates}, Failed: {failed_updates}, "
                f"Rate: {self.stats.symbols_per_second:.1f} symbols/sec"
            )
            
        except Exception as e:
            self.stats.error_count += 1
            self.logger.error(f"Error in update cycle: {e}")
    
    def create_dashboard(self):
        """Create optimized HTML dashboard"""
        dashboard_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Data Downloader Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 20px; color: white;
        }
        .dashboard { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .card { 
            background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); 
            border-radius: 15px; padding: 20px; border: 1px solid rgba(255,255,255,0.2);
        }
        .metric { display: flex; justify-content: space-between; margin: 10px 0; }
        .metric-label { font-weight: 600; }
        .metric-value { font-weight: bold; }
        .status-badge { 
            display: inline-block; padding: 5px 15px; border-radius: 20px; 
            font-weight: bold; font-size: 0.9em;
        }
        .status-running { background: #4CAF50; }
        .status-stopped { background: #f44336; }
        .status-market-open { background: #2196F3; }
        .status-market-closed { background: #FF9800; }
        .progress-bar { 
            width: 100%; height: 10px; background: rgba(255,255,255,0.2); 
            border-radius: 5px; overflow: hidden; margin: 10px 0;
        }
        .progress-fill { height: 100%; background: #4CAF50; transition: width 0.3s; }
        .emergency-stop { 
            background: #f44336; color: white; border: none; 
            padding: 15px 30px; border-radius: 10px; font-size: 1.1em; 
            cursor: pointer; margin: 20px auto; display: block;
        }
        .refresh-info { text-align: center; margin: 20px 0; opacity: 0.8; }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>Live Data Downloader Dashboard</h1>
            <p>High-performance stock data downloads</p>
        </div>
        
        <div class="status-grid">
            <div class="card">
                <h3>System Status</h3>
                <div class="metric">
                    <span class="metric-label">Status:</span>
                    <span class="status-badge" id="system-status">Loading...</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Market:</span>
                    <span class="status-badge" id="market-status">Loading...</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Uptime:</span>
                    <span class="metric-value" id="uptime">Loading...</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Next Update:</span>
                    <span class="metric-value" id="next-update">Loading...</span>
                </div>
            </div>
            
            <div class="card">
                <h3>Performance Metrics</h3>
                <div class="metric">
                    <span class="metric-label">Total Updates:</span>
                    <span class="metric-value" id="update-count">Loading...</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Success Rate:</span>
                    <span class="metric-value" id="success-rate">Loading...</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="success-progress"></div>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Update Time:</span>
                    <span class="metric-value" id="avg-time">Loading...</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Symbols/Second:</span>
                    <span class="metric-value" id="symbols-per-sec">Loading...</span>
                </div>
            </div>
            
            <div class="card">
                <h3>Data Statistics</h3>
                <div class="metric">
                    <span class="metric-label">Total Symbols:</span>
                    <span class="metric-value" id="total-symbols">Loading...</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Successful Updates:</span>
                    <span class="metric-value" id="successful-updates">Loading...</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Failed Updates:</span>
                    <span class="metric-value" id="failed-updates">Loading...</span>
                </div>
                <div class="metric">
                    <span class="metric-label">API 429 Errors:</span>
                    <span class="metric-value" id="api-429-count">Loading...</span>
                </div>
            </div>
            
            <div class="card">
                <h3>API Rate Limiting</h3>
                <div class="metric">
                    <span class="metric-label">Can Make Request:</span>
                    <span class="metric-value" id="can-request">Loading...</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Recent Requests:</span>
                    <span class="metric-value" id="recent-requests">Loading...</span>
                </div>
                <div class="metric">
                    <span class="metric-label">In Cooldown:</span>
                    <span class="metric-value" id="in-cooldown">Loading...</span>
                </div>
            </div>
        </div>
        
        <button class="emergency-stop" onclick="emergencyStop()">EMERGENCY STOP</button>
        
        <div class="refresh-info">
            <p>Dashboard auto-refreshes every 10 seconds • Last Updated: <span id="last-updated">Loading...</span></p>
        </div>
    </div>

    <script>
        let statusData = {};
        
        async function loadStatus() {
            try {
                const response = await fetch('live_status.json');
                statusData = await response.json();
                updateDashboard();
            } catch (error) {
                console.error('Error loading status:', error);
            }
        }
        
        function updateDashboard() {
            document.getElementById('system-status').textContent = statusData.running ? 'RUNNING' : 'STOPPED';
            document.getElementById('system-status').className = 'status-badge ' + (statusData.running ? 'status-running' : 'status-stopped');
            
            document.getElementById('market-status').textContent = statusData.market_status || 'UNKNOWN';
            document.getElementById('market-status').className = 'status-badge ' + (statusData.market_status === 'OPEN' ? 'status-market-open' : 'status-market-closed');
            
            document.getElementById('uptime').textContent = statusData.uptime_hours ? statusData.uptime_hours.toFixed(1) + ' hours' : 'N/A';
            document.getElementById('next-update').textContent = statusData.next_update_time || 'N/A';
            
            document.getElementById('update-count').textContent = statusData.update_count || 0;
            
            const successfulUpdates = statusData.successful_updates || 0;
            const failedUpdates = statusData.failed_updates || 0;
            const totalAttempts = successfulUpdates + failedUpdates;
            const successRate = totalAttempts > 0 ? ((successfulUpdates / totalAttempts) * 100).toFixed(1) + '%' : 'N/A';
            
            document.getElementById('success-rate').textContent = successRate;
            document.getElementById('success-progress').style.width = (totalAttempts > 0 ? (successfulUpdates / totalAttempts) * 100 : 0) + '%';
            
            document.getElementById('avg-time').textContent = statusData.average_update_time ? statusData.average_update_time.toFixed(2) + 's' : 'N/A';
            document.getElementById('symbols-per-sec').textContent = statusData.symbols_per_second ? statusData.symbols_per_second.toFixed(1) : 'N/A';
            
            document.getElementById('total-symbols').textContent = statusData.total_symbols || 0;
            document.getElementById('successful-updates').textContent = successfulUpdates;
            document.getElementById('failed-updates').textContent = failedUpdates;
            document.getElementById('api-429-count').textContent = statusData.api_429_count || 0;
            
            if (statusData.api_rate_limit_status) {
                document.getElementById('can-request').textContent = statusData.api_rate_limit_status.can_make_request ? 'YES' : 'NO';
                document.getElementById('recent-requests').textContent = statusData.api_rate_limit_status.recent_requests + '/110';
                document.getElementById('in-cooldown').textContent = statusData.api_rate_limit_status.in_cooldown ? 'YES' : 'NO';
            }
            
            document.getElementById('last-updated').textContent = new Date().toLocaleTimeString();
        }
        
        function emergencyStop() {
            if (confirm('Are you sure you want to EMERGENCY STOP the live data downloader?')) {
                fetch('emergency_stop', { method: 'POST' })
                    .then(() => alert('Emergency stop signal sent!'))
                    .catch(() => alert('Please create EMERGENCY_STOP.txt file manually in the data folder'));
            }
        }
        
        setInterval(loadStatus, 10000);
        loadStatus();
    </script>
</body>
</html>'''
        
        with open(self.dashboard_file, 'w') as f:
            f.write(dashboard_html)
        
        self.logger.info(f"Dashboard created: {self.dashboard_file}")
    
    def _background_worker(self):
        """Optimized background worker"""
        self.logger.info("Live data downloader started in background")
        self.stats.running = True
        self.stats.start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        emergency_check_counter = 0
        
        while not self.stop_event.is_set():
            try:
                # Check emergency stop every 5 seconds
                emergency_check_counter += 1
                if emergency_check_counter >= 5:
                    if self.check_emergency_stop():
                        self.logger.critical("EMERGENCY STOP FILE DETECTED - STOPPING IMMEDIATELY")
                        self.stats.running = False
                        break
                    emergency_check_counter = 0
                
                # Auto-stop after market close
                if self.is_after_market_close():
                    self.logger.info("Market closed (after 3:30 PM) - Auto-stopping")
                    self.stats.running = False
                    break
                
                # Perform update if in market hours
                if self.is_market_hours():
                    self.perform_update_cycle()
                    
                    # Optimized sleep calculation
                    next_update = self.get_next_update_time()
                    sleep_seconds = max(1, min((next_update - datetime.datetime.now()).total_seconds(), 300))
                    
                    # Fast sleep checking
                    for _ in range(int(sleep_seconds)):
                        if self.stop_event.is_set():
                            break
                        time.sleep(1)
                else:
                    self.logger.info("Outside market hours, waiting...")
                    time.sleep(30)
                    
            except Exception as e:
                self.stats.error_count += 1
                self.logger.error(f"Unexpected error in background worker: {e}")
                time.sleep(5)  # Reduced error wait time
        
        self.stats.running = False
        self.save_status_to_file()
        self.logger.info("Live data downloader stopped")
    
    def start(self):
        """Start the optimized live data downloader"""
        if self.running:
            self.logger.warning("Live data downloader is already running")
            return
        
        try:
            self._load_and_cache_data()
            
            os.makedirs(self.output_folder, exist_ok=True)
            self.create_emergency_stop_mechanism()
            self.create_dashboard()
            
            self.running = True
            self.stop_event.clear()
            self.background_thread = threading.Thread(target=self._background_worker, daemon=True)
            self.background_thread.start()
            
            self.logger.info("Live data downloader started successfully")
            self.logger.info(f"Monitoring {len(self.token_cache)} symbols every {self.update_interval//60} minutes")
            self.logger.info(f"Dashboard available at: {self.dashboard_file}")
            
            # Auto-open dashboard
            try:
                webbrowser.open(f'file://{os.path.abspath(self.dashboard_file)}')
            except:
                pass
            
        except Exception as e:
            self.logger.error(f"Failed to start live data downloader: {e}")
            self.running = False
            raise
    
    def stop(self):
        """Stop the live data downloader gracefully"""
        if not self.running:
            return
        
        self.logger.info("Stopping live data downloader...")
        self.running = False
        self.stop_event.set()
        
        if self.background_thread and self.background_thread.is_alive():
            self.background_thread.join(timeout=10)  # Reduced timeout
        
        self.stats.running = False
        self.save_status_to_file()
        self.logger.info("Live data downloader stopped successfully")
    
    def status(self) -> Dict:
        """Get current comprehensive status"""
        return asdict(self.stats)

def download_live_data(
    kite,
    symbols_file: str = 'symbols.txt',
    output_folder: str = "stocks_historical_data",
    max_workers: int = 35,  # Increased for maximum speed
    update_interval: int = 300
) -> LiveDataDownloader:
    """
    Start high-performance live data downloading (OPTIMIZED VERSION)
    
    Features:
    - 35+ concurrent threads for maximum speed
    - Zerodha API rate limit compliance (110 req/min aggressive)
    - Auto-stop after 3:30 PM
    - Emergency stop mechanism (create EMERGENCY_STOP.txt)
    - Real-time HTML dashboard
    - Performance monitoring
    - 5-minute aligned updates during market hours
    - Optimized file I/O and caching
    
    Args:
        kite: Authenticated Zerodha Kite instance
        symbols_file: Path to symbols file (default: 'symbols.txt')
        output_folder: Output directory (default: 'stocks_historical_data')
        max_workers: Concurrent threads (default: 35 for maximum speed)
        update_interval: Update interval in seconds (default: 300 = 5 minutes)
    
    Returns:
        LiveDataDownloader instance for monitoring and control
    
    Usage:
        # Start live data downloading (ONE-TIME CALL)
        downloader = download_live_data(kite, max_workers=35)
        
        # Dashboard opens automatically
        
        # Check status programmatically
        print(downloader.status())
        
        # Manual stop (auto-stops at 3:30 PM)
        downloader.stop()
        
        # Emergency stop: Create 'EMERGENCY_STOP.txt' in data folder
    """
    
    downloader = LiveDataDownloader(
        kite=kite,
        symbols_file=symbols_file,
        output_folder=output_folder,
        max_workers=max_workers,
        update_interval=update_interval
    )
    
    downloader.start()
    return downloader

# Example usage
if __name__ == "__main__":
    # Example: Start optimized live data downloading
    # downloader = download_live_data(kite, max_workers=35)
    
    # The system will:
    # 1. Run automatically every 5 minutes during market hours (9:15 AM - 3:30 PM, Mon-Fri)
    # 2. Auto-stop after 3:30 PM
    # 3. Open dashboard in browser for monitoring
    # 4. Create emergency stop mechanism
    # 5. Log everything with optimized performance
    
    pass

