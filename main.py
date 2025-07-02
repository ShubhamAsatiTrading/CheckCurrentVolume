# main.py - COMPLETE TRADING SYSTEM DASHBOARD
# Individual function controls + Live Analysis + Full Trading System

import streamlit as st
import subprocess
import os
import pandas as pd
from datetime import datetime
import time
import json
import threading
import queue
import sys

# Load environment variables
from dotenv import load_dotenv
from volume_average import VolumeAverage
import subprocess, sys
import os
import sys
import glob

# Force UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

# Fix stdout/stderr if on Windows
if sys.platform.startswith('win') and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
        sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
    except:
        pass
load_dotenv()

# Configuration
CONFIG_FILE = "common.txt"
TOKEN_FILE = "kite_token.txt"
LOG_FILE = f"trading_system_logs_{datetime.now().strftime('%Y%m%d')}.log"
REQUIRED_DIRS = ["stocks_historical_data", "aggregated_data", "Volume_boost_consolidated"]

# Kite Connect Configuration from environment
KITE_API_KEY = os.getenv('KITE_API_KEY', '')
KITE_API_SECRET = os.getenv('KITE_API_SECRET', '')

st.set_page_config(page_title="Trading System", page_icon="📈", layout="wide")

# Enhanced CSS
st.markdown("""
<style>
.metric-card { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    color: white; 
    padding: 0.8rem; 
    border-radius: 8px; 
    text-align: center;
    height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.metric-card h3 {
    margin: 0;
    font-size: 0.9rem;
    font-weight: 500;
}
.metric-card h2 {
    margin: 0.2rem 0;
    font-size: 1.1rem;
    font-weight: 600;
}
.metric-card p {
    margin: 0;
    font-size: 0.75rem;
    opacity: 0.9;
}
.signal-card { 
    background: white; padding: 1rem; border-radius: 8px; 
    box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 0.5rem 0; 
    border-left: 4px solid #2196F3;
}
.auth-card {
    background: #f0f8ff; padding: 1rem; border-radius: 8px; 
    border: 1px solid #4CAF50; margin: 1rem 0;
}
.function-card {
    background: #f8f9fa; 
    border: 1px solid #dee2e6; 
    border-radius: 8px; 
    padding: 1rem; 
    margin: 0.5rem 0;
}
.log-container {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
    max-height: 400px;
    overflow-y: auto;
}
.progress-step {
    background: #e3f2fd;
    border-left: 4px solid #2196f3;
    padding: 0.5rem;
    margin: 0.5rem 0;
    border-radius: 4px;
}
.progress-step.completed {
    background: #e8f5e8;
    border-left-color: #4caf50;
}
.progress-step.error {
    background: #ffebee;
    border-left-color: #f44336;
}
</style>
""", unsafe_allow_html=True)

class Logger:
    @staticmethod
    def log_to_file(message, level="INFO"):
        """Log message to file with timestamp"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"[{timestamp}] {level}: {message}\n"
            
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Failed to write to log file: {e}")
    
    @staticmethod
    def get_log_file_path():
        return os.path.abspath(LOG_FILE)

class TokenManager:
    @staticmethod
    def get_valid_token():
        """Check if we have a valid token for today, return it if available"""
        if os.path.exists(TOKEN_FILE):
            try:
                with open(TOKEN_FILE, "r") as f:
                    token_data = json.loads(f.read().strip())
                    today = datetime.now().strftime("%Y-%m-%d")
                    
                    if token_data.get("date") == today and token_data.get("access_token"):
                        Logger.log_to_file(f"Valid token found for {today}")
                        return token_data["access_token"]
                    else:
                        Logger.log_to_file("Token expired, removing old token file")
                        os.remove(TOKEN_FILE)  # Delete old token
            except Exception as e:
                Logger.log_to_file(f"Error reading token file: {e}", "ERROR")
                if os.path.exists(TOKEN_FILE):
                    os.remove(TOKEN_FILE)
        return None
    
    @staticmethod
    def save_token(access_token):
        """Save access token with today's date"""
        try:
            token_data = {
                "access_token": access_token, 
                "date": datetime.now().strftime("%Y-%m-%d")
            }
            with open(TOKEN_FILE, "w") as f:
                f.write(json.dumps(token_data))
            Logger.log_to_file("Access token saved successfully")
            return True
        except Exception as e:
            Logger.log_to_file(f"Failed to save token: {e}", "ERROR")
            return False
    
    @staticmethod
    def generate_login_url():
        """Generate Kite login URL"""
        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=KITE_API_KEY)
            url = kite.login_url()
            Logger.log_to_file("Generated Kite login URL")
            return url
        except Exception as e:
            Logger.log_to_file(f"Error generating login URL: {e}", "ERROR")
            st.error(f"Error generating login URL: {e}")
            return None
    
    @staticmethod
    def generate_session_from_request_token(request_token):
        """Generate access token from request token"""
        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=KITE_API_KEY)
            session = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
            access_token = session["access_token"]
            
            Logger.log_to_file(f"Generated session from request token")
            
            # Save the token
            if TokenManager.save_token(access_token):
                return access_token
            return None
        except Exception as e:
            Logger.log_to_file(f"Failed to generate session: {e}", "ERROR")
            st.error(f"Failed to generate session: {e}")
            return None
    
    @staticmethod
    def get_token_status():
        """Get current token status"""
        token = TokenManager.get_valid_token()
        if token:
            return {
                "has_token": True,
                "token": token,
                "status": "✅ Valid token available",
                "date": datetime.now().strftime("%Y-%m-%d")
            }
        else:
            return {
                "has_token": False,
                "token": None,
                "status": "❌ No valid token",
                "date": None
            }

class ConfigManager:
    @staticmethod
    def load():
        defaults = {"stop_loss": 4.0, "target": 10.0, "ohlc_value": "open", "trade_today_flag": "no", "check_from_date": "2020-03-28"}
        try:
            config = {}
            with open(CONFIG_FILE, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        config[key] = float(value) if key in ['stop_loss', 'target'] else value
            Logger.log_to_file(f"Configuration loaded: {config}")
            return {**defaults, **config}
        except Exception as e:
            Logger.log_to_file(f"Error loading config, using defaults: {e}", "WARNING")
            ConfigManager.save(defaults)
            return defaults
    
    @staticmethod
    def save(config):
        try:
            with open(CONFIG_FILE, 'w') as f:
                for key, value in config.items():
                    f.write(f"{key}={value}\n")
            Logger.log_to_file(f"Configuration saved: {config}")
            return True
        except Exception as e:
            Logger.log_to_file(f"Failed to save config: {e}", "ERROR")
            return False

class TradingSystem:
    @staticmethod
    def validate_setup():
        missing = [d for d in REQUIRED_DIRS if not os.path.exists(d)]
        if missing:
            Logger.log_to_file(f"Missing directories: {missing}", "WARNING")
        else:
            Logger.log_to_file("All required directories exist")
        return len(missing) == 0, missing
    
    @staticmethod
    def is_market_hours():
        """Check if within market hours (9:15 AM - 3:30 PM)"""
        now = datetime.now().time()
        market_start = datetime.strptime("09:15", "%H:%M").time()
        market_end = datetime.strptime("15:30", "%H:%M").time()
        return market_start <= now <= market_end
    
    @staticmethod
    def get_kite_instance():
        """Get initialized Kite instance using web token"""
        try:
            token_status = TokenManager.get_token_status()
            if not token_status["has_token"]:
                raise Exception("No valid token available")
            
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=KITE_API_KEY, timeout=60)
            kite.set_access_token(token_status["token"])
            
            # Test connection
            profile = kite.profile()
            Logger.log_to_file(f"Kite instance created for: {profile['user_name']}")
            return kite
            
        except Exception as e:
            Logger.log_to_file(f"Error creating Kite instance: {e}", "ERROR")
            raise
    
    @staticmethod
    def run_individual_function(module_name, function_name, **kwargs):
        """Run individual trading system function"""
        try:
            Logger.log_to_file(f"Running {module_name}.{function_name} with params: {kwargs}")
            
            # Import module
            try:
                module = __import__(module_name)
            except ImportError as e:
                return False, f"❌ Module {module_name} not found: {e}"
            
            # Get function
            try:
                function = getattr(module, function_name)
            except AttributeError as e:
                return False, f"❌ Function {function_name} not found in {module_name}: {e}"
            
            # Execute function
            result = function(**kwargs)
            Logger.log_to_file(f"✅ {module_name}.{function_name} completed successfully")
            return True, f"✅ {module_name}.{function_name} completed successfully"
            
        except Exception as e:
            error_msg = f"❌ Error in {module_name}.{function_name}: {e}"
            Logger.log_to_file(error_msg, "ERROR")
            return False, error_msg
    
    @staticmethod
    def run_full_trading_system(access_token, progress_container, log_container):
        """Run complete trading system (same as before)"""
        
        Logger.log_to_file("=== FULL TRADING SYSTEM STARTED ===")
        
        # Prepare environment
        env = os.environ.copy()
        load_dotenv(override=True)
        
        env.update({
            'TRADING_ACCESS_TOKEN': access_token,
            'KITE_API_KEY': KITE_API_KEY or os.getenv('KITE_API_KEY', ''),
            'KITE_API_SECRET': KITE_API_SECRET or os.getenv('KITE_API_SECRET', ''),
            'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN', ''),
            'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID', ''),
            'PYTHONPATH': os.getcwd(),
            'PYTHONUNBUFFERED': '1',
        })
        
        # Progress tracking steps
        progress_steps = [
            "🔄 Starting Trading System...",
            "📊 Scraping stock fundamentals...",
            "🔗 Testing kiteconnect import...",
            "🔐 Initializing authentication...",
            "📈 Downloading live/historical data...",
            "📋 Aggregating data...",
            "🧪 Running backtest...",
            "🔍 Running live scanner & notifications...",
            "✅ Trading system completed!"
        ]
        
        # Initialize progress
        progress_container.markdown("### 📊 Progress")
        progress_placeholders = []
        for i, step in enumerate(progress_steps):
            placeholder = progress_container.empty()
            progress_placeholders.append(placeholder)
        
        # Initialize log container
        log_container.markdown("### 📜 Live Logs")
        log_placeholder = log_container.empty()
        
        try:
            # Verify code_1.py exists
            if not os.path.exists('code_1.py'):
                error_msg = "❌ code_1.py not found in current directory!"
                Logger.log_to_file(error_msg, "ERROR")
                log_placeholder.error(error_msg)
                return False, "", "code_1.py not found"
            
            log_placeholder.markdown(f"""
            <div class="log-container">
            <strong>🚀 FULL TRADING SYSTEM STARTUP</strong><br/>
            <strong>⏰ Started at:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
            <strong>📄 Log File:</strong> {Logger.get_log_file_path()}<br/>
            <br/>
            <strong>📡 Real-time output:</strong><br/>
            <div id="log-content">Initializing subprocess...</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Update first progress step
            progress_placeholders[0].markdown('<div class="progress-step">🔄 Starting Trading System...</div>', unsafe_allow_html=True)
            
            Logger.log_to_file("Starting subprocess for code_1.py")
            
            # Start the subprocess
            process = subprocess.Popen(
                [sys.executable, os.path.abspath('code_1.py')],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                encoding='utf-8',
                errors='ignore',
                bufsize=1,
                env=env,
                cwd=os.getcwd()
            )
            
            Logger.log_to_file(f"Subprocess started with PID: {process.pid}")
            
            # Real-time output streaming
            log_content = []
            current_step = 0
            
            # Keywords to detect progress steps
            step_keywords = [
                "TRADING SYSTEM STARTING",
                "Scraping stock fundamentals",
                "Testing kiteconnect import",
                "Getting credentials",
                "Downloading historical data",
                "Aggregating data", 
                "Running backtest",
                "Running live scanner",
                "TRADING SYSTEM COMPLETED"
            ]
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    try:
                        line_clean = line.rstrip()
                        log_content.append(line_clean)
                        Logger.log_to_file(f"SUBPROCESS: {line_clean}")
                    except UnicodeDecodeError:
                        continue
                    
                    # Update progress based on keywords
                    for i, keyword in enumerate(step_keywords):
                        if keyword.lower() in line_clean.lower() and i > current_step:
                            # Mark previous steps as completed
                            for j in range(current_step + 1):
                                if j < len(progress_steps):
                                    progress_placeholders[j].markdown(
                                        f'<div class="progress-step completed">✅ {progress_steps[j]}</div>', 
                                        unsafe_allow_html=True
                                    )
                            
                            # Mark current step as active
                            if i < len(progress_steps):
                                progress_placeholders[i].markdown(
                                    f'<div class="progress-step">🔄 {progress_steps[i]}</div>', 
                                    unsafe_allow_html=True
                                )
                            current_step = i
                            break
                    
                    # Check for errors
                    if any(error_word in line_clean.lower() for error_word in ['error', 'failed', 'exception']):
                        if current_step < len(progress_steps):
                            progress_placeholders[current_step].markdown(
                                f'<div class="progress-step error">❌ {progress_steps[current_step]} - Error detected</div>', 
                                unsafe_allow_html=True
                            )
                    
                    # Update log display
                    formatted_logs = "<br/>".join([
                        f"<span style='color: #666;'>{i+1:3d}:</span> {log_line}" 
                        for i, log_line in enumerate(log_content[-50:])
                    ])
                    
                    log_placeholder.markdown(f"""
                    <div class="log-container">
                    <strong>🚀 TRADING SYSTEM LIVE LOG</strong> <small>(Last 50 lines)</small><br/>
                    <strong>⏰ Started at:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
                    <strong>📄 Full logs:</strong> <code>{Logger.get_log_file_path()}</code><br/>
                    <br/>
                    {formatted_logs}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    time.sleep(0.05)
            
            # Wait for process to complete
            return_code = process.wait()
            Logger.log_to_file(f"Subprocess completed with return code: {return_code}")
            
            # Final update
            if return_code == 0:
                # Mark all steps as completed
                for i, step in enumerate(progress_steps):
                    progress_placeholders[i].markdown(
                        f'<div class="progress-step completed">✅ {step}</div>', 
                        unsafe_allow_html=True
                    )
                Logger.log_to_file("=== FULL TRADING SYSTEM COMPLETED SUCCESSFULLY ===")
                return True, "\n".join(log_content), ""
            else:
                # Mark current step as error
                if current_step < len(progress_steps):
                    progress_placeholders[current_step].markdown(
                        f'<div class="progress-step error">❌ {progress_steps[current_step]} - Failed</div>', 
                        unsafe_allow_html=True
                    )
                Logger.log_to_file(f"=== FULL TRADING SYSTEM FAILED WITH CODE {return_code} ===", "ERROR")
                return False, "\n".join(log_content), f"Process exited with code {return_code}"
                
        except Exception as e:
            error_msg = f"Exception occurred: {str(e)}"
            Logger.log_to_file(f"CRITICAL ERROR: {error_msg}", "ERROR")
            return False, "", error_msg

class LiveAnalysis:
    @staticmethod
    def consolidate_if_needed():
        """Smart consolidation - only if data is missing or old"""
        consolidated_file = "Volume_boost_consolidated/consolidated_data.csv"
        
        # Check if file exists and is from today
        if os.path.exists(consolidated_file):
            try:
                file_mod_time = datetime.fromtimestamp(os.path.getmtime(consolidated_file))
                today = datetime.now().date()
                
                if file_mod_time.date() == today:
                    Logger.log_to_file("Consolidated data is current, skipping consolidation")
                    return True
            except Exception as e:
                Logger.log_to_file(f"Error checking file date: {e}", "WARNING")
        
        # Need to consolidate
        Logger.log_to_file("Consolidating volume boost data...")
        try:
            try:
                import consolidated_volume
            except ImportError as e:
                Logger.log_to_file(f"Error importing consolidated_volume: {e}", "ERROR")
                return False
            
            consolidated_volume.consolidate_volume_boost_data(
                interval_minutes=0,
                interval_days=1,
                input_folder="aggregated_data",
                save_files=True
            )
            Logger.log_to_file("Data consolidation completed")
            return True
        except Exception as e:
            Logger.log_to_file(f"Error consolidating data: {e}", "ERROR")
            return False
    
    @staticmethod
    def get_live_signals():
        """Get live signals for analysis (same logic as Telegram but for web display)"""
        try:
            # Step 1: Check market hours
            if not TradingSystem.is_market_hours():
                return [], "🕒 Market is closed (Trading hours: 9:15 AM - 3:30 PM)"
            
            # Step 2: Smart consolidation
            if not LiveAnalysis.consolidate_if_needed():
                return [], "❌ Error: Failed to consolidate data"
            
            # Step 3: Check consolidated data
            consolidated_file = "Volume_boost_consolidated/consolidated_data.csv"
            if not os.path.exists(consolidated_file):
                return [], "❌ Error: No consolidated data found. Run 'Trading System' first."
            
            # Step 4: Read consolidated data
            try:
                df = pd.read_csv(consolidated_file)
                if df.empty:
                    return [], "⚠️ Warning: Consolidated data file is empty"
            except Exception as e:
                return [], f"❌ Error reading consolidated data: {e}"
            
            # Step 5: Get configuration
            config = ConfigManager.load()
            ohlc_value = config.get('ohlc_value', 'open')
            
            if ohlc_value not in df.columns:
                return [], f"❌ Error: Column '{ohlc_value}' not found in data"
            
            # Step 6: Initialize Kite API
            try:
                kite = TradingSystem.get_kite_instance()
            except Exception as e:
                return [], f"❌ Error: Failed to connect to Kite API - {e}"
            
            # Step 7: Prepare symbols
            symbols = df['symbol'].unique().tolist()
            formatted_symbols = [f"NSE:{symbol}" for symbol in symbols]
            
            Logger.log_to_file(f"Scanning {len(symbols)} symbols for live signals...")
            
            # Step 8: Get live market data
            try:
                live_data = kite.ltp(formatted_symbols)
                Logger.log_to_file(f"Successfully fetched live data for {len(live_data)} symbols")
            except Exception as e:
                return [], f"❌ Error: Failed to fetch live prices - {e}"
            
            # Step 9: Generate signals (exact same logic as Telegram)
            signals = []
            current_time = datetime.now().strftime("%H:%M:%S")
            current_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            errors_count = 0
            
            for _, row in df.iterrows():
                try:
                    symbol = str(row['symbol']).strip()
                    
                    # Safe float conversion
                    try:
                        csv_value = float(row[ohlc_value])
                    except (ValueError, TypeError) as e:
                        Logger.log_to_file(f"Invalid CSV value for {symbol}: {row[ohlc_value]}", "WARNING")
                        errors_count += 1
                        continue
                    
                    formatted_symbol = f"NSE:{symbol}"
                    
                    if formatted_symbol in live_data:
                        try:
                            live_price = float(live_data[formatted_symbol]['last_price'])
                        except (ValueError, TypeError, KeyError) as e:
                            Logger.log_to_file(f"Invalid live price for {symbol}: {e}", "WARNING")
                            errors_count += 1
                            continue
                        
                        # Check if live price > CSV price (same condition as Telegram)
                        if live_price > csv_value:
                            try:
                                percentage_increase = ((live_price - csv_value) / csv_value) * 100
                                csv_date = str(row.get('date', 'N/A'))
                                
                                # Match exact Telegram signal format
                                signal = {
                                    'symbol': symbol,
                                    'signal_type': 'TRADING SIGNAL',
                                    'csv_date': csv_date,
                                    'ohlc_type': ohlc_value.upper(),
                                    'csv_price': round(csv_value, 2),
                                    'live_price': round(live_price, 2),
                                    'increase': round(percentage_increase, 2),
                                    'current_time': current_time,
                                    'timestamp': current_timestamp,
                                    'volume': str(row.get('volume', 'N/A'))
                                }
                                signals.append(signal)
                                Logger.log_to_file(f"Signal: {symbol} - Live: ₹{live_price:,.2f}, CSV: ₹{csv_value:,.2f}, +{percentage_increase:.2f}%")
                            except Exception as e:
                                Logger.log_to_file(f"Error creating signal for {symbol}: {e}", "WARNING")
                                errors_count += 1
                    else:
                        errors_count += 1
                        
                except Exception as e:
                    Logger.log_to_file(f"Error processing {row.get('symbol', 'Unknown')}: {e}", "WARNING")
                    errors_count += 1
                    continue
            
            # Return results
            Logger.log_to_file(f"Live analysis completed: {len(signals)} signals generated, {errors_count} errors")
            
            if signals:
                if errors_count > 0:
                    return signals, f"✅ {len(signals)} signals found (⚠️ {errors_count} symbols had errors)"
                else:
                    return signals, f"✅ {len(signals)} active signals detected"
            else:
                if errors_count > 0:
                    return [], f"📊 No signals detected. ⚠️ {errors_count} symbols had data errors."
                else:
                    return [], "📊 No signals detected. All live prices below baseline."
                
        except Exception as e:
            error_msg = f"❌ Critical error in live analysis: {e}"
            Logger.log_to_file(error_msg, "ERROR")
            return [], error_msg

def main():
    st.title("📈 Complete Trading System Dashboard")
    
    # Initialize session state
     # CORRECT session state initialization
    session_keys = ['config', 'analysis_signals', 'analysis_running', 'last_function_run', 
                    'volume_change_data', 'volume_change_collapsed', 'trading_system_collapsed']
    
    for key in session_keys:
        if key not in st.session_state:
            if key == 'config':
                st.session_state[key] = ConfigManager.load()
            elif key == 'analysis_signals':
                st.session_state[key] = []
            elif key == 'volume_change_data':
                st.session_state[key] = []
            elif key in ['volume_change_collapsed', 'trading_system_collapsed']:
                st.session_state[key] = False
            elif key == 'last_function_run':
                st.session_state[key] = None
            else:
                st.session_state[key] = False
    
    # Check token status
    token_status = TokenManager.get_token_status()
    
    # Sidebar - Authentication & Config
    with st.sidebar:
        st.header("🔐 Authentication")
        
        if not KITE_API_KEY or not KITE_API_SECRET:
            st.error("❌ KITE_API_KEY or KITE_API_SECRET not found in .env file!")
            st.info("Please add your Kite credentials to the .env file")
            return
        
        if token_status["has_token"]:
            st.markdown(f"""
            <div class="auth-card">
                <strong>{token_status['status']}</strong><br>
                <small>Valid for: {token_status['date']}</small><br>
                <small>Token: {token_status['token'][:15]}...</small>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh"):
                    if os.path.exists(TOKEN_FILE):
                        os.remove(TOKEN_FILE)
                    st.rerun()
            with col2:
                if st.button("🗑️ Clear"):
                    if os.path.exists(TOKEN_FILE):
                        os.remove(TOKEN_FILE)
                    st.success("Token cleared!")
                    st.rerun()
        else:
            st.warning(token_status["status"])
            
            if st.button("🔗 Get Kite Login URL"):
                login_url = TokenManager.generate_login_url()
                if login_url:
                    st.markdown(f"**Click here:** [Kite Login]({login_url})")
                    st.info("Copy request_token from URL after login")
            
            request_token = st.text_input("Request Token:")
            
            if st.button("✅ Generate Token") and request_token:
                with st.spinner("Generating token..."):
                    access_token = TokenManager.generate_session_from_request_token(request_token.strip())
                    if access_token:
                        st.success("✅ Token generated!")
                        st.rerun()
                    else:
                        st.error("❌ Token generation failed")
        
        # Configuration
        if token_status["has_token"]:
            st.markdown("---")
            st.subheader("⚙️ Configuration")
            config = st.session_state.config.copy()
            
            config['stop_loss'] = st.number_input("Stop Loss %", 0.1, 20.0, config['stop_loss'], 0.1)
            config['target'] = st.number_input("Target %", 0.1, 50.0, config['target'], 0.1)
            config['ohlc_value'] = st.selectbox("OHLC", ["open", "high", "low", "close"], 
                                               ["open", "high", "low", "close"].index(config['ohlc_value']))
            config['trade_today_flag'] = st.selectbox("Trade Flag", ["yes", "no"], 
                                                      ["yes", "no"].index(config['trade_today_flag']))
            
            # Add check_from_date field
            from datetime import datetime
            current_date = config.get('check_from_date', '2020-03-28')
            if isinstance(current_date, str):
                try:
                    date_obj = datetime.strptime(current_date, '%Y-%m-%d').date()
                except:
                    date_obj = datetime(2020, 3, 28).date()
            else:
                date_obj = current_date
            
            new_date = st.date_input("Check From Date", value=date_obj)
            config['check_from_date'] = new_date.strftime('%Y-%m-%d')
            
            if st.button("💾 Update Config"):
                if ConfigManager.save(config):
                    st.session_state.config = config
                    st.success("Configuration updated!")
                else:
                    st.error("Failed to update configuration")

    # Main Dashboard - Status Cards
    col1, col2, col3, col4 = st.columns(4)
    
    now = datetime.now()
    market_open = TradingSystem.is_market_hours()
    
    with col1:
        st.markdown(f"""<div class="metric-card">
        <h3>Market</h3><h2>{'🟢 OPEN' if market_open else '🔴 CLOSED'}</h2>
        <p>{now.strftime('%H:%M:%S')}</p></div>""", unsafe_allow_html=True)
    
    with col2:
        token_color = "🟢" if token_status["has_token"] else "🔴"
        token_text = "AUTH" if token_status["has_token"] else "NO TOKEN"
        st.markdown(f"""<div class="metric-card">
        <h3>Authentication</h3><h2>{token_color} {token_text}</h2>
        <p>{"Ready" if token_status["has_token"] else "Login required"}</p></div>""", 
        unsafe_allow_html=True)
    
    with col3:
        status = st.session_state.config['trade_today_flag']
        st.markdown(f"""<div class="metric-card">
        <h3>Trading</h3><h2>{'🟢' if status == 'yes' else '🔴'} {status.upper()}</h2>
        <p>SL: {st.session_state.config['stop_loss']}% | T: {st.session_state.config['target']}%</p></div>""", 
        unsafe_allow_html=True)
    
    with col4:
        if st.session_state.analysis_running and market_open:
            analysis_status = "🟢 ACTIVE"
            analysis_desc = "Analysis running"
        elif st.session_state.analysis_running and not market_open:
            analysis_status = "🟡 PAUSED"
            analysis_desc = "Market closed"
        else:
            analysis_status = "🔴 STOPPED"
            analysis_desc = "Click to start"
        
        signals_count = len(st.session_state.analysis_signals)
        st.markdown(f"""<div class="metric-card">
        <h3>Live Analysis</h3><h2>{analysis_status}</h2>
        <p>{analysis_desc} | {signals_count} signals</p></div>""", unsafe_allow_html=True)

    # Main Content - Only show if authenticated
    if token_status["has_token"]:
        
        # Section 1: Individual Function Controls
        st.header("🔧 Individual Function Controls")
        
        # Stock Scraping
        with st.expander("📊 Stock Scraping (Screener.in)", expanded=False):
            st.markdown("""
            <div class="function-card">
            <strong>Function:</strong> scrapper.scrape_and_save_all()<br/>
            <strong>Purpose:</strong> Scrapes latest stock fundamentals from screener.in<br/>
            <strong>Output:</strong> Updated stock fundamental data files
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                verbose_mode = st.checkbox("Verbose Mode", value=False, key="scrape_verbose")
            with col2:
                scrape_delay = st.number_input("Delay (seconds)", 0.1, 5.0, 1.0, 0.1, key="scrape_delay")
            
            if st.button("▶️ Run Stock Scraping", key="stock_scrape"):
                with st.spinner("📊 Scraping stock data from screener.in..."):
                    success, message = TradingSystem.run_individual_function(
                        'scrapper', 
                        'scrape_and_save_all',
                        verbose=verbose_mode,
                        delay=scrape_delay
                    )
                    if success:
                        st.success(message)
                        st.session_state.last_function_run = "Stock Scraping"
                    else:
                        st.error(message)
        
        # Historical Data Download
        # Historical Data Download
        # Historical Data Download
        with st.expander("📈 Historical Data Download", expanded=False):
            st.markdown("""
            <div class="function-card">
            <strong>Function:</strong> historical_data_download.download_historical_data(kite)<br/>
            <strong>Purpose:</strong> Downloads historical price data for all symbols<br/>
            <strong>Output:</strong> Files in stocks_historical_data/ folder
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                symbol_file = st.text_input("Symbol Names File", value="symbols.txt", key="hist_symbol_file")
                start_date = st.date_input("Start Date", value=datetime(2020, 1, 1).date(), key="hist_start_date")
                max_workers = st.number_input("Max Workers", min_value=1, max_value=4, value=3, key="hist_max_workers")
            with col2:
                output_folder = st.text_input("Output Folder", value="stocks_historical_data", key="hist_output_folder")
                interval_minutes = st.number_input("Interval (mins)", min_value=1, max_value=1440, value=5, key="hist_interval_mins")
            
            if st.button("▶️ Run Historical Data Download", key="hist_data"):
                with st.spinner("📥 Downloading historical data..."):
                    try:
                        kite = TradingSystem.get_kite_instance()
                        
                        # Convert date picker to datetime.datetime format
                        start_datetime = datetime.combine(start_date, datetime.min.time())
                        
                        # Convert minutes to interval string format
                        interval_str = f"{interval_minutes}minute"
                        
                        success, message = TradingSystem.run_individual_function(
                            'historical_data_download', 
                            'download_historical_data', 
                            kite=kite,
                            symbol_names_file=symbol_file,
                            output_folder=output_folder,
                            start=start_datetime,
                            interval=interval_str,
                            max_workers=max_workers
                        )
                        if success:
                            st.success(message)
                            st.session_state.last_function_run = "Historical Data Download"
                        else:
                            st.error(message)
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        # Data Aggregation
        with st.expander("📊 Data Aggregation", expanded=False):
            st.markdown("""
            <div class="function-card">
            <strong>Function:</strong> data_aggregate.interactive_aggregation()<br/>
            <strong>Purpose:</strong> Aggregates historical data with volume boost analysis<br/>
            <strong>Output:</strong> Files in aggregated_data/ folder
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                interval_minutes = st.number_input("Interval Minutes", 0, 60, 0, key="agg_min")
            with col2:
                interval_days = st.number_input("Interval Days", 1, 30, 1, key="agg_days")
            with col3:
                input_folder = st.text_input("Input Folder", "stocks_historical_data", key="agg_folder")
            
            if st.button("▶️ Run Data Aggregation", key="data_agg"):
                with st.spinner("📋 Aggregating data..."):
                    success, message = TradingSystem.run_individual_function(
                        'data_aggregate', 
                        'interactive_aggregation',
                        interval_minutes=interval_minutes,
                        interval_days=interval_days,
                        input_folder=input_folder,
                        save_files=True
                    )
                    if success:
                        st.success(message)
                        st.session_state.last_function_run = "Data Aggregation"
                    else:
                        st.error(message)
        
        # Backtest
        with st.expander("🧪 Backtest Analysis", expanded=False):
            st.markdown("""
            <div class="function-card">
            <strong>Function:</strong> long_term_backtest.run_backtest()<br/>
            <strong>Purpose:</strong> Runs backtesting on aggregated data<br/>
            <strong>Output:</strong> Backtest results and performance metrics
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("▶️ Run Backtest", key="backtest"):
                with st.spinner("🧪 Running backtest..."):
                    success, message = TradingSystem.run_individual_function(
                        'long_term_backtest', 
                        'run_backtest'
                    )
                    if success:
                        st.success(message)
                        st.session_state.last_function_run = "Backtest"
                    else:
                        st.error(message)
        
        # Consolidated Volume
        with st.expander("🔍 Consolidated Volume Analysis", expanded=False):
            st.markdown("""
            <div class="function-card">
            <strong>Function:</strong> consolidated_volume.complete_trading_workflow()<br/>
            <strong>Purpose:</strong> Consolidates data and runs live market scanning<br/>
            <strong>Output:</strong> Consolidated data + Telegram signals (if configured)
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                cv_interval_minutes = st.number_input("Interval Minutes", 0, 60, 0, key="cv_min")
            with col2:
                cv_interval_days = st.number_input("Interval Days", 1, 30, 1, key="cv_days")
            with col3:
                enable_telegram = st.checkbox("Enable Telegram", value=True, key="enable_tg")
            
            if st.button("▶️ Run Consolidated Volume", key="cons_vol"):
                with st.spinner("🔍 Running consolidated volume analysis..."):
                    try:
                        kite = TradingSystem.get_kite_instance()
                        kwargs = {
                            'interval_minutes': cv_interval_minutes,
                            'interval_days': cv_interval_days,
                            'kite_instance': kite
                        }
                        
                        if enable_telegram:
                            kwargs.update({
                                'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
                                'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID')
                            })
                        
                        success, message = TradingSystem.run_individual_function(
                            'consolidated_volume', 
                            'complete_trading_workflow',
                            **kwargs
                        )
                        if success:
                            st.success(message)
                            st.session_state.last_function_run = "Consolidated Volume"
                        else:
                            st.error(message)
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
 # Volume Average Analysis
        with st.expander("📊 Volume Average Analysis", expanded=False):
            st.markdown("""
            <div class="function-card">
            <strong>Function:</strong> VolumeAverage.calculate_average_volume_data()<br/>
            <strong>Purpose:</strong> Calculate average volume and VWAP for all stocks over specified duration<br/>
            <strong>Output:</strong> AvgData_till_[date].csv in Average data/ folder<br/>
            <strong>Duration:</strong> Reads avg_volume_days from common.txt (weekdays only)
            </div>
            """, unsafe_allow_html=True)
            
            # Show current configuration
            current_config = ConfigManager.load()
            current_duration = current_config.get('avg_volume_days', 30)
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Current Duration:** {current_duration} weekdays")
                st.info(f"**Input Folder:** stocks_historical_data/")
            with col2:
                st.info(f"**Output Folder:** Average data/")
                st.info(f"**File Format:** AvgData_till_DDMmmYYYY.csv")
            
            # Show what will be calculated
            st.markdown("**📈 Output Columns:**")
            st.markdown("""
            - **Symbol:** Stock name
            - **Yest_Average_Close_Price:** Average of daily closing prices
            - **Yest_Average_Volume:** Average of daily total volumes  
            - **Yest_Daily_VWAP_Average:** Average of daily VWAPs
            - **Yest_Overall_VWAP:** VWAP for entire duration
            """)
            
            if st.button("▶️ Calculate Volume Averages", key="volume_avg"):
                with st.spinner(f"📊 Calculating volume averages for {current_duration} weekdays..."):
                    try:
                        success, message = VolumeAverage.calculate_average_volume_data()
                        if success:
                            st.success(message)
                            st.session_state.last_function_run = "Volume Average Analysis"
                            
                            # Show output file info
                            output_folder = "Average data"
                            if os.path.exists(output_folder):
                                files = [f for f in os.listdir(output_folder) if f.startswith("AvgData_till_")]
                                if files:
                                    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(output_folder, x)))
                                    st.info(f"📄 Latest output: {latest_file}")
                                    
                                    # Check if Live Volume Change data needs refresh
                                    if 'volume_change_data' in st.session_state:
                                        st.info("💡 New AvgData generated! Refresh Live Volume Change section to see updated comparison.")
                        else:
                            st.error(message)
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        # Live Data Downloader Control
        with st.expander("📡 Live Data Downloader Control", expanded=False):
            st.markdown("""
            <div class="function-card">
            <strong>Function:</strong> Start/Stop Live Data Collection<br/>
            <strong>Purpose:</strong> Control live stock data downloading during market hours<br/>
            <strong>Output:</strong> StockLiveData_YYYY-MM-DD.csv files<br/>
            <strong>Frequency:</strong> Updates every rerun_minute from common.txt
            </div>
            """, unsafe_allow_html=True)
            
            # Check current status
            current_config = ConfigManager.load()
            live_download_enabled = current_config.get('live_data_download', 'no').lower() == 'yes'
            rerun_minute = current_config.get('rerun_minute', 1)
            
            # Status display
            col1, col2 = st.columns(2)
            with col1:
                status_color = "🟢" if live_download_enabled else "🔴"
                status_text = "ENABLED" if live_download_enabled else "DISABLED"
                st.markdown(f"**Status:** {status_color} {status_text}")
                st.info(f"**Update Interval:** {rerun_minute} minute(s)")
            
            with col2:
                # Check if today's file exists
                today = datetime.now().strftime("%Y-%m-%d")
                todays_file = f"StockLiveData_{today}.csv"
                file_exists = os.path.exists(todays_file)
                file_status = "✅ EXISTS" if file_exists else "❌ NOT FOUND"
                st.markdown(f"**Today's Data:** {file_status}")
                if file_exists:
                    mod_time = datetime.fromtimestamp(os.path.getmtime(todays_file))
                    st.info(f"**Last Updated:** {mod_time.strftime('%H:%M:%S')}")
                else:
                    st.info("**File:** StockLiveData_YYYY-MM-DD.csv")
            
            # Configuration controls
            st.markdown("**⚙️ Configuration:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Enable/Disable toggle
                new_status = "yes" if st.checkbox("Enable Live Download", value=live_download_enabled) else "no"
                
                if new_status != current_config.get('live_data_download', 'no'):
                    if st.button("💾 Update Config", key="update_live_config"):
                        try:
                            # Update common.txt
                            config_lines = []
                            config_updated = False
                            
                            if os.path.exists('common.txt'):
                                with open('common.txt', 'r') as f:
                                    config_lines = f.readlines()
                            
                            # Update or add live_data_download
                            for i, line in enumerate(config_lines):
                                if line.strip().startswith('live_data_download='):
                                    config_lines[i] = f"live_data_download={new_status}\n"
                                    config_updated = True
                                    break
                            
                            if not config_updated:
                                config_lines.append(f"live_data_download={new_status}\n")
                            
                            # Write back to file
                            with open('common.txt', 'w') as f:
                                f.writelines(config_lines)
                            
                            st.success(f"✅ Live download {new_status.upper()}D in configuration")
                            time.sleep(1)
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error updating config: {e}")
            
            with col2:
                # Manual start option
                if live_download_enabled:
                    st.markdown("**🚀 Start Options:**")
                    if st.button("📋 Show Start Command", key="show_start_cmd"):
                        st.code("python live_data_downloader_parallel.py", language="bash")
                        st.info("💡 Run this command in terminal to start live data collection")
                else:
                    st.warning("Enable live download first")
            
            with col3:
                # Quick status check
                st.markdown("**📊 Quick Actions:**")
                if st.button("🔄 Refresh Status", key="refresh_live_status"):
                    st.rerun()
                
                if st.button("📂 Open Data Folder", key="open_data_folder"):
                    try:
                        if os.name == 'nt':  # Windows
                            os.startfile('.')
                        elif os.name == 'posix':  # Linux/Mac
                            subprocess.run(['xdg-open', '.'])
                        st.success("📂 Opened project folder")
                    except Exception as e:
                        st.info("💡 Navigate to your project folder to see data files")
            
            # Instructions and tips
            st.markdown("**💡 Instructions:**")
            st.markdown("""
            1. **Enable** live download checkbox above
            2. **Update Config** to save the setting  
            3. **Run in terminal:** `python live_data_downloader_parallel.py`
            4. **Monitor** this dashboard for data updates
            5. **Use Live Volume Change** section to analyze results
            """)
            
            # Show current files
            if st.checkbox("Show Recent Data Files", key="show_recent_files"):
                st.markdown("**📄 Recent Data Files:**")
                try:
                    # Show StockLiveData files
                    live_files = [f for f in os.listdir('.') if f.startswith('StockLiveData_')]
                    live_files.sort(reverse=True)
                    
                    if live_files:
                        st.markdown("**Live Data Files:**")
                        for file in live_files[:5]:  # Show last 5 files
                            mod_time = datetime.fromtimestamp(os.path.getmtime(file))
                            st.text(f"• {file} - {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    else:
                        st.text("No StockLiveData files found")
                    
                    # Show AvgData files
                    if os.path.exists('Average data'):
                        avg_files = [f for f in os.listdir('Average data') if f.startswith('AvgData_')]
                        avg_files.sort(reverse=True)
                        
                        if avg_files:
                            st.markdown("**Average Data Files:**")
                            for file in avg_files[:3]:  # Show last 3 files
                                full_path = os.path.join('Average data', file)
                                mod_time = datetime.fromtimestamp(os.path.getctime(full_path))
                                st.text(f"• {file} - {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        else:
                            st.text("No AvgData files found in Average data folder")
                    
                except Exception as e:
                    st.error(f"Error listing files: {e}")
        st.markdown("---")
        
        # Section 2: Full Trading System
        col1, col2 = st.columns([4, 1])
        with col1:
            st.header("🚀 Complete Trading System")
        with col2:
            if st.button("📂" if st.session_state.trading_system_collapsed else "📁", 
                         key="toggle_trading_system_main",
                         help="Collapse/Expand Complete Trading System"):
                st.session_state.trading_system_collapsed = not st.session_state.trading_system_collapsed
                st.rerun()

        # Complete Trading System Content
        if not st.session_state.trading_system_collapsed:
            st.info("**Runs the complete code_1.py workflow:** All 9 steps including data scraping, historical download, aggregation, backtest, and live scanning with Telegram notifications.")
            
            if st.button("▶️ Run Complete Trading System", type="primary", key="full_system_execute"):
                st.header("🔄 Full Trading System Execution")
                
                progress_container = st.container()
                log_container = st.container()
                
                Logger.log_to_file("User started complete trading system")
                
                success, stdout, stderr = TradingSystem.run_full_trading_system(
                    token_status["token"], progress_container, log_container
                )
                
                if success:
                    st.success("✅ Complete trading system finished successfully!")
                    st.session_state.analysis_signals = []
                    st.session_state.last_function_run = "Complete Trading System"
                else:
                    st.error("❌ Trading system encountered errors!")
                    if stderr:
                        st.error(f"Details: {stderr}")

        st.markdown("---")

        # Live Volume Change Section
        col1, col2 = st.columns([4, 1])
        with col1:
            st.header("📊 Live Volume Change")
        with col2:
            if st.button("📂" if st.session_state.volume_change_collapsed else "📁", 
                         key="toggle_volume_change_main",
                         help="Collapse/Expand Live Volume Change"):
                st.session_state.volume_change_collapsed = not st.session_state.volume_change_collapsed
                st.rerun()

        # Live Volume Change Content
        if not st.session_state.volume_change_collapsed:
            st.info("**Compare historical volume averages with live market data** to identify significant volume changes and trading opportunities.")
            
            # File Status and Refresh Controls
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                file_info = LiveVolumeChange.get_file_info()
                st.markdown(f"""
                **📄 Source Files:**
                - **AvgData:** {file_info['avg_file']}
                - **LiveData:** {file_info['live_file']}
                """)
            
            with col2:
                st.markdown(f"""
                **⏰ Last Modified:**
                - **AvgData:** {file_info['avg_modified']}
                - **LiveData:** {file_info['live_modified']}
                """)
            
            with col3:
                if st.button("🔄 Refresh", key="refresh_volume_change_main"):
                    with st.spinner("📊 Loading volume change data..."):
                        data, error = LiveVolumeChange.load_and_merge_data()
                        if error:
                            st.error(f"❌ {error}")
                            st.session_state.volume_change_data = []
                        else:
                            st.session_state.volume_change_data = data
                            st.success(f"✅ Loaded {len(data)} symbols")
                        st.rerun()
            
            # Initialize data if empty
            if not st.session_state.volume_change_data:
                with st.spinner("📊 Loading volume change data..."):
                    data, error = LiveVolumeChange.load_and_merge_data()
                    if error:
                        st.error(f"❌ {error}")
                        st.session_state.volume_change_data = []
                    else:
                        st.session_state.volume_change_data = data
            
            # Display Data
            if st.session_state.volume_change_data:
                # Filter Controls
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    max_diff = st.number_input(
                        "Max %Diff Filter", 
                        min_value=-100.0, 
                        max_value=1000.0, 
                        value=1000.0, 
                        step=1.0,
                        key="max_diff_volume_change",
                        help="Show only symbols with percentage difference less than or equal to this value"
                    )
                with col2:
                    min_diff = st.number_input(
                        "Min %Diff Filter", 
                        min_value=-100.0, 
                        max_value=1000.0, 
                        value=-100.0, 
                        step=1.0,
                        key="min_diff_volume_change",
                        help="Show only symbols with percentage difference greater than or equal to this value"
                    )
                with col3:
                    st.markdown("**Filter Examples:**")
                    st.markdown("• Min -10%, Max 50% = Show -10% to +50% range")
                    st.markdown("• Min 20% = Show only positive volume spikes >20%")
                
                # Prepare filtered data
                filtered_data = []
                for row in st.session_state.volume_change_data:
                    percentage_diff = row.get('Percentage_Diff', 'N/A')
                    
                    # Apply filters
                    if percentage_diff != 'N/A':
                        if min_diff <= percentage_diff <= max_diff:
                            filtered_data.append(row)
                    else:
                        # Include N/A values
                        filtered_data.append(row)
                
                if filtered_data:
                    # Convert to DataFrame with proper column order
                    df_volume_change = pd.DataFrame(filtered_data)
                    
                    # Reorder columns as specified
                    column_order = [
                        'Symbol', 'Last_Price', 'VWAP', 'Yest_Daily_VWAP_Average', 
                        'Yest_Overall_VWAP', 'Yest_Average_Volume', 'Volume_Avg', 
                        'Yest_Average_Close_Price', 'Percentage_Diff', 'Last_Updated'
                    ]
                    
                    # Ensure all columns exist
                    for col in column_order:
                        if col not in df_volume_change.columns:
                            df_volume_change[col] = 'N/A'
                    
                    # Reorder DataFrame
                    df_volume_change = df_volume_change[column_order]
                    
                    # Display count
                    st.markdown(f"### 📊 Volume Change Analysis (Showing: {len(filtered_data)} symbols)")
                    
                    # Format the DataFrame for display with color coding
                    def format_percentage_diff(val):
                        if val == 'N/A' or pd.isna(val):
                            return val
                        try:
                            num_val = float(val)
                            if num_val > 0:
                                return f"🟢 +{num_val:.2f}%"
                            elif num_val < 0:
                                return f"🔴 {num_val:.2f}%"
                            else:
                                return f"{num_val:.2f}%"
                        except:
                            return val
                    
                    # Apply formatting
                    display_df = df_volume_change.copy()
                    display_df['Percentage_Diff'] = display_df['Percentage_Diff'].apply(format_percentage_diff)
                    
                    # Configure column display
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        height=600,
                        column_config={
                            "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                            "Last_Price": st.column_config.NumberColumn("Last Price", width="small", format="%.2f"),
                            "VWAP": st.column_config.NumberColumn("Live VWAP", width="small", format="%.2f"),
                            "Yest_Daily_VWAP_Average": st.column_config.NumberColumn("Yest Daily VWAP", width="small", format="%.2f"),
                            "Yest_Overall_VWAP": st.column_config.NumberColumn("Yest Overall VWAP", width="small", format="%.2f"),
                            "Yest_Average_Volume": st.column_config.NumberColumn("Yest Avg Volume", width="medium", format="%.0f"),
                            "Volume_Avg": st.column_config.NumberColumn("Live Avg Volume", width="medium", format="%.0f"),
                            "Yest_Average_Close_Price": st.column_config.NumberColumn("Yest Avg Price", width="small", format="%.2f"),
                            "Percentage_Diff": st.column_config.TextColumn("% Diff", width="small"),
                            "Last_Updated": st.column_config.TextColumn("Last Updated", width="small")
                        }
                    )
                    
                    # Export Options
                    st.markdown("### 📥 Export Options")
                    col1, col2, col3 = st.columns([1, 1, 2])
                    
                    with col1:
                        # Download as Excel
                        try:
                            import io
                            
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                df_volume_change.to_excel(writer, sheet_name='Volume_Change_Analysis', index=False)
                            
                            excel_data = output.getvalue()
                            
                            st.download_button(
                                label="📊 Download Excel",
                                data=excel_data,
                                file_name=f"volume_change_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        except ImportError:
                            st.error("📋 Install openpyxl: pip install openpyxl")
                    
                    with col2:
                        # Download as CSV
                        csv_data = df_volume_change.to_csv(index=False)
                        st.download_button(
                            label="📋 Download CSV",
                            data=csv_data,
                            file_name=f"volume_change_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col3:
                        st.info(f"📈 **{len(filtered_data)} symbols analyzed** | Last refreshed: {datetime.now().strftime('%H:%M:%S')}")
                
                else:
                    st.warning(f"🔍 No symbols found with %diff between {min_diff}% and {max_diff}%")
                    st.info(f"Total symbols available: {len(st.session_state.volume_change_data)}")
                    if st.button("🔄 Reset Filters", key="reset_volume_filters_main"):
                        st.rerun()
            
            else:
                st.warning("📊 No volume change data available")
                st.markdown("""
                **📋 Requirements:**
                - ✅ Latest AvgData file in `Average data/` folder
                - ✅ Today's StockLiveData file (`StockLiveData_YYYY-MM-DD.csv`)
                
                **💡 To generate data:**
                1. Run **Volume Average Analysis** to create AvgData file
                2. Run **Live Data Downloader** to create today's live data
                3. Click **🔄 Refresh** to load the comparison
                """)
# =============================================================================
# LIVE VOLUME CHANGE IMPLEMENTATION FOR main.py
# =============================================================================

# 1. ADD THIS CLASS AFTER THE EXISTING CLASSES (after LiveAnalysis class)

class LiveVolumeChange:
    """Live Volume Change Analysis - Compare historical averages with live data"""
    
    @staticmethod
    def find_latest_avgdata_file():
        """Find the latest AvgData file in Average data folder"""
        try:
            avg_folder = "Average data"
            if not os.path.exists(avg_folder):
                return None, "Average data folder not found"
            
            # Find all AvgData files
            pattern = os.path.join(avg_folder, "AvgData_till_*.csv")
            files = glob.glob(pattern)
            
            if not files:
                return None, "No AvgData files found"
            
            # Get the latest file by modification time
            latest_file = max(files, key=os.path.getctime)
            Logger.log_to_file(f"Found latest AvgData file: {latest_file}")
            return latest_file, None
            
        except Exception as e:
            error_msg = f"Error finding AvgData file: {e}"
            Logger.log_to_file(error_msg, "ERROR")
            return None, error_msg
    
    @staticmethod
    def find_todays_livedata_file():
        """Find today's StockLiveData file"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            filename = f"StockLiveData_{today}.csv"
            
            if os.path.exists(filename):
                Logger.log_to_file(f"Found today's live data file: {filename}")
                return filename, None
            else:
                return None, f"Today's live data file not found: {filename}"
                
        except Exception as e:
            error_msg = f"Error finding live data file: {e}"
            Logger.log_to_file(error_msg, "ERROR")
            return None, error_msg
    
    @staticmethod
    def load_and_merge_data():
        """Load and merge AvgData and LiveData files"""
        try:
            Logger.log_to_file("Starting Live Volume Change analysis")
            
            # Find files
            avg_file, avg_error = LiveVolumeChange.find_latest_avgdata_file()
            live_file, live_error = LiveVolumeChange.find_todays_livedata_file()
            
            # Check for file errors
            if avg_error:
                return [], avg_error
            if live_error:
                return [], live_error
            
            # Load AvgData
            try:
                df_avg = pd.read_csv(avg_file)
                Logger.log_to_file(f"Loaded AvgData: {len(df_avg)} stocks")
            except Exception as e:
                return [], f"Error reading AvgData file: {e}"
            
            # Load LiveData
            try:
                df_live = pd.read_csv(live_file)
                Logger.log_to_file(f"Loaded LiveData: {len(df_live)} stocks")
            except Exception as e:
                return [], f"Error reading LiveData file: {e}"
            
            # Merge data
            merged_data = []
            
            # Get all unique symbols from both datasets
            all_symbols = set(df_avg['Symbol'].tolist() + df_live['Symbol'].tolist())
            
            for symbol in all_symbols:
                try:
                    # Get data from both sources
                    avg_row = df_avg[df_avg['Symbol'] == symbol]
                    live_row = df_live[df_live['Symbol'] == symbol]
                    
                    # Initialize row data
                    row_data = {'Symbol': symbol}
                    
                    # Add live data (if available)
                    if not live_row.empty:
                        live_data = live_row.iloc[0]
                        row_data.update({
                            'Last_Price': live_data.get('Last_Price', 'N/A'),
                            'VWAP': live_data.get('VWAP', 'N/A'),
                            'Volume_Avg': live_data.get('Volume_Avg', 'N/A'),
                            'Last_Updated': live_data.get('Last_Updated', 'N/A')
                        })
                    else:
                        row_data.update({
                            'Last_Price': 'N/A',
                            'VWAP': 'N/A', 
                            'Volume_Avg': 'N/A',
                            'Last_Updated': 'N/A'
                        })
                    
                    # Add average data (if available)
                    if not avg_row.empty:
                        avg_data = avg_row.iloc[0]
                        row_data.update({
                            'Yest_Daily_VWAP_Average': avg_data.get('Yest_Daily_VWAP_Average', 'N/A'),
                            'Yest_Overall_VWAP': avg_data.get('Yest_Overall_VWAP', 'N/A'),
                            'Yest_Average_Volume': avg_data.get('Yest_Average_Volume', 'N/A'),
                            'Yest_Average_Close_Price': avg_data.get('Yest_Average_Close_Price', 'N/A')
                        })
                    else:
                        row_data.update({
                            'Yest_Daily_VWAP_Average': 'N/A',
                            'Yest_Overall_VWAP': 'N/A',
                            'Yest_Average_Volume': 'N/A',
                            'Yest_Average_Close_Price': 'N/A'
                        })
                    
                    # Calculate percentage difference
                    percentage_diff = 'N/A'
                    if (row_data['Volume_Avg'] != 'N/A' and 
                        row_data['Yest_Average_Volume'] != 'N/A'):
                        try:
                            volume_avg = float(row_data['Volume_Avg'])
                            yest_avg_volume = float(row_data['Yest_Average_Volume'])
                            
                            if yest_avg_volume != 0:
                                percentage_diff = ((volume_avg - yest_avg_volume) / yest_avg_volume) * 100
                                percentage_diff = round(percentage_diff, 2)
                        except (ValueError, ZeroDivisionError):
                            percentage_diff = 'N/A'
                    
                    row_data['Percentage_Diff'] = percentage_diff
                    
                    merged_data.append(row_data)
                    
                except Exception as e:
                    Logger.log_to_file(f"Error processing symbol {symbol}: {e}", "WARNING")
                    continue
            
            Logger.log_to_file(f"Live Volume Change analysis completed: {len(merged_data)} symbols processed")
            return merged_data, None
            
        except Exception as e:
            error_msg = f"Critical error in Live Volume Change analysis: {e}"
            Logger.log_to_file(error_msg, "ERROR")
            return [], error_msg
    
    @staticmethod
    def get_file_info():
        """Get information about source files"""
        try:
            info = {
                'avg_file': 'Not found',
                'avg_modified': 'N/A',
                'live_file': 'Not found', 
                'live_modified': 'N/A'
            }
            
            # AvgData file info
            avg_file, _ = LiveVolumeChange.find_latest_avgdata_file()
            if avg_file:
                info['avg_file'] = os.path.basename(avg_file)
                mod_time = datetime.fromtimestamp(os.path.getmtime(avg_file))
                info['avg_modified'] = mod_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # LiveData file info
            live_file, _ = LiveVolumeChange.find_todays_livedata_file()
            if live_file:
                info['live_file'] = os.path.basename(live_file)
                mod_time = datetime.fromtimestamp(os.path.getmtime(live_file))
                info['live_modified'] = mod_time.strftime('%Y-%m-%d %H:%M:%S')
            
            return info
            
        except Exception as e:
            Logger.log_to_file(f"Error getting file info: {e}", "ERROR")
            return {
                'avg_file': 'Error',
                'avg_modified': 'Error',
                'live_file': 'Error',
                'live_modified': 'Error'
            }

# =============================================================================
# 2. ADD THESE IMPORTS AT THE TOP OF main.py (if not already present)
# =============================================================================

import glob  # Add this import if not present

# =============================================================================
# 3. ADD THIS SESSION STATE INITIALIZATION (in the main() function, with other session states)
# =============================================================================

# Add this to your existing session state initialization:
session_keys = ['config', 'analysis_signals', 'analysis_running', 'last_function_run', 
                'volume_change_data', 'volume_change_collapsed', 'trading_system_collapsed']

for key in session_keys:
    if key not in st.session_state:
        if key == 'config':
            st.session_state[key] = ConfigManager.load()
        elif key == 'analysis_signals':
            st.session_state[key] = []
        elif key == 'volume_change_data':
            st.session_state[key] = []
        elif key in ['volume_change_collapsed', 'trading_system_collapsed']:
            st.session_state[key] = False
        elif key == 'last_function_run':
            st.session_state[key] = None
        else:
            st.session_state[key] = False

# =============================================================================
# 4. UPDATE THE COMPLETE TRADING SYSTEM SECTION HEADER
# =============================================================================

# Replace the existing "st.header("🚀 Complete Trading System")" with this:

col1, col2 = st.columns([4, 1])
with col1:
    st.header("🚀 Complete Trading System")
with col2:
    if st.button("📂" if st.session_state.trading_system_collapsed else "📁", 
                 key="toggle_trading_system",
                 help="Collapse/Expand Complete Trading System"):
        st.session_state.trading_system_collapsed = not st.session_state.trading_system_collapsed
        st.rerun()

# =============================================================================
# 5. WRAP THE COMPLETE TRADING SYSTEM CONTENT
# =============================================================================

# Wrap all the existing Complete Trading System content (everything from the description 
# to the end of that section) in this if statement:

if st.button("▶️ Run Complete Trading System", type="primary", key="full_system_execute"):
    st.header("🔄 Full Trading System Execution")  # ← Properly indented
    
    progress_container = st.container()
    log_container = st.container()
    
    Logger.log_to_file("User started complete trading system")
    token_status = TokenManager.get_token_status()
    success, stdout, stderr = TradingSystem.run_full_trading_system(
        token_status["token"], progress_container, log_container
    )
    
    if success:
        st.success("✅ Complete trading system finished successfully!")
        st.session_state.analysis_signals = []
        st.session_state.last_function_run = "Complete Trading System"
    else:
        st.error("❌ Trading system encountered errors!")
        if stderr:
            st.error(f"Details: {stderr}")

# =============================================================================
# 6. ADD THE NEW LIVE VOLUME CHANGE SECTION (after Complete Trading System)
# =============================================================================

st.markdown("---")

# Live Volume Change Section Header
col1, col2 = st.columns([4, 1])
with col1:
    st.header("📊 Live Volume Change")
with col2:
    if st.button("📂" if st.session_state.volume_change_collapsed else "📁", 
                 key="toggle_volume_change",
                 help="Collapse/Expand Live Volume Change"):
        st.session_state.volume_change_collapsed = not st.session_state.volume_change_collapsed
        st.rerun()

# Live Volume Change Content
if not st.session_state.volume_change_collapsed:
    st.info("**Compare historical volume averages with live market data** to identify significant volume changes and trading opportunities.")
    
    # File Status and Refresh Controls
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        file_info = LiveVolumeChange.get_file_info()
        st.markdown(f"""
        **📄 Source Files:**
        - **AvgData:** {file_info['avg_file']}
        - **LiveData:** {file_info['live_file']}
        """)
    
    with col2:
        st.markdown(f"""
        **⏰ Last Modified:**
        - **AvgData:** {file_info['avg_modified']}
        - **LiveData:** {file_info['live_modified']}
        """)
    
    with col3:
        if st.button("🔄 Refresh", key="refresh_volume_change"):
            with st.spinner("📊 Loading volume change data..."):
                data, error = LiveVolumeChange.load_and_merge_data()
                if error:
                    st.error(f"❌ {error}")
                    st.session_state.volume_change_data = []
                else:
                    st.session_state.volume_change_data = data
                    st.success(f"✅ Loaded {len(data)} symbols")
                st.rerun()
    
    # Initialize data if empty
    if not st.session_state.volume_change_data:
        with st.spinner("📊 Loading volume change data..."):
            data, error = LiveVolumeChange.load_and_merge_data()
            if error:
                st.error(f"❌ {error}")
                st.session_state.volume_change_data = []
            else:
                st.session_state.volume_change_data = data
    
    # Display Data
    if st.session_state.volume_change_data:
        # Filter Controls (same as Trading Signals)
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            max_diff = st.number_input(
                "Max %Diff Filter", 
                min_value=-100.0, 
                max_value=1000.0, 
                value=1000.0, 
                step=1.0,
                help="Show only symbols with percentage difference less than or equal to this value"
            )
        with col2:
            min_diff = st.number_input(
                "Min %Diff Filter", 
                min_value=-100.0, 
                max_value=1000.0, 
                value=-100.0, 
                step=1.0,
                help="Show only symbols with percentage difference greater than or equal to this value"
            )
        with col3:
            st.markdown("**Filter Examples:**")
            st.markdown("• Min -10%, Max 50% = Show -10% to +50% range")
            st.markdown("• Min 20% = Show only positive volume spikes >20%")
        
        # Prepare filtered data
        filtered_data = []
        for row in st.session_state.volume_change_data:
            percentage_diff = row.get('Percentage_Diff', 'N/A')
            
            # Apply filters
            if percentage_diff != 'N/A':
                if min_diff <= percentage_diff <= max_diff:
                    filtered_data.append(row)
            else:
                # Include N/A values if within range allows
                filtered_data.append(row)
        
        if filtered_data:
            # Convert to DataFrame with proper column order
            df_volume_change = pd.DataFrame(filtered_data)
            
            # Reorder columns as specified
            column_order = [
                'Symbol', 'Last_Price', 'VWAP', 'Yest_Daily_VWAP_Average', 
                'Yest_Overall_VWAP', 'Yest_Average_Volume', 'Volume_Avg', 
                'Yest_Average_Close_Price', 'Percentage_Diff', 'Last_Updated'
            ]
            
            # Ensure all columns exist
            for col in column_order:
                if col not in df_volume_change.columns:
                    df_volume_change[col] = 'N/A'
            
            # Reorder DataFrame
            df_volume_change = df_volume_change[column_order]
            
            # Display count
            st.markdown(f"### 📊 Volume Change Analysis (Showing: {len(filtered_data)} symbols)")
            
            # Format the DataFrame for display with color coding
            def format_percentage_diff(val):
                if val == 'N/A' or pd.isna(val):
                    return val
                try:
                    num_val = float(val)
                    if num_val > 0:
                        return f"🟢 +{num_val:.2f}%"
                    elif num_val < 0:
                        return f"🔴 {num_val:.2f}%"
                    else:
                        return f"{num_val:.2f}%"
                except:
                    return val
            
            # Apply formatting
            display_df = df_volume_change.copy()
            display_df['Percentage_Diff'] = display_df['Percentage_Diff'].apply(format_percentage_diff)
            
            # Configure column display
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=600,
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "Last_Price": st.column_config.NumberColumn("Last Price", width="small", format="%.2f"),
                    "VWAP": st.column_config.NumberColumn("Live VWAP", width="small", format="%.2f"),
                    "Yest_Daily_VWAP_Average": st.column_config.NumberColumn("Yest Daily VWAP", width="small", format="%.2f"),
                    "Yest_Overall_VWAP": st.column_config.NumberColumn("Yest Overall VWAP", width="small", format="%.2f"),
                    "Yest_Average_Volume": st.column_config.NumberColumn("Yest Avg Volume", width="medium", format="%.0f"),
                    "Volume_Avg": st.column_config.NumberColumn("Live Avg Volume", width="medium", format="%.0f"),
                    "Yest_Average_Close_Price": st.column_config.NumberColumn("Yest Avg Price", width="small", format="%.2f"),
                    "Percentage_Diff": st.column_config.TextColumn("% Diff", width="small"),
                    "Last_Updated": st.column_config.TextColumn("Last Updated", width="small")
                }
            )
            
            # Export Options
            st.markdown("### 📥 Export Options")
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                # Download as Excel (filtered data)
                try:
                    import io
                    from datetime import datetime
                    
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_volume_change.to_excel(writer, sheet_name='Volume_Change_Analysis', index=False)
                    
                    excel_data = output.getvalue()
                    
                    st.download_button(
                        label="📊 Download Excel",
                        data=excel_data,
                        file_name=f"volume_change_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except ImportError:
                    st.error("📋 Install openpyxl: pip install openpyxl")
            
            with col2:
                # Download as CSV (filtered data)
                csv_data = df_volume_change.to_csv(index=False)
                st.download_button(
                    label="📋 Download CSV",
                    data=csv_data,
                    file_name=f"volume_change_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col3:
                st.info(f"📈 **{len(filtered_data)} symbols analyzed** | Last refreshed: {datetime.now().strftime('%H:%M:%S')}")
        
        else:
            st.warning(f"🔍 No symbols found with %diff between {min_diff}% and {max_diff}%")
            st.info(f"Total symbols available: {len(st.session_state.volume_change_data)}")
            if st.button("🔄 Reset Filters", key="reset_volume_filters"):
                st.rerun()
    
    else:
        st.warning("📊 No volume change data available")
        st.markdown("""
        **📋 Requirements:**
        - ✅ Latest AvgData file in `Average data/` folder
        - ✅ Today's StockLiveData file (`StockLiveData_YYYY-MM-DD.csv`)
        
        **💡 To generate data:**
        1. Run **Volume Average Analysis** to create AvgData file
        2. Run **Live Data Downloader** to create today's live data
        3. Click **🔄 Refresh** to load the comparison
        """)
        

if __name__ == "__main__":
    Logger.log_to_file("=== TRADING SYSTEM DASHBOARD STARTED ===")
    main()