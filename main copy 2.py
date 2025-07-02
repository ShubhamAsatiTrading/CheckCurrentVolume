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
import os
import sys

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
    session_keys = ['config', 'analysis_signals', 'analysis_running', 'last_function_run']
    for key in session_keys:
        if key not in st.session_state:
            if key == 'config':
                st.session_state[key] = ConfigManager.load()
            elif key == 'analysis_signals':
                st.session_state[key] = []
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

        st.markdown("---")
        
        # Section 2: Full Trading System
        st.header("🚀 Complete Trading System")
        
        st.info("**Runs the complete code_1.py workflow:** All 9 steps including data scraping, historical download, aggregation, backtest, and live scanning with Telegram notifications.")
        
        if st.button("▶️ Run Complete Trading System", type="primary", key="full_system"):
            st.header("🔄 Full Trading System Execution")
            
            progress_container = st.container()
            log_container = st.container()
            
            Logger.log_to_file("User started complete trading system")
            
            success, stdout, stderr = TradingSystem.run_full_trading_system(
                token_status["token"], progress_container, log_container
            )
            
            if success:
                st.success("✅ Complete trading system finished successfully!")
                # Clear analysis signals after full system run
                st.session_state.analysis_signals = []
                st.session_state.last_function_run = "Complete Trading System"
            else:
                st.error("❌ Trading system encountered errors!")
                if stderr:
                    st.error(f"Details: {stderr}")

        st.markdown("---")
        
        # Section 3: Live Analysis (Toggle Analysis)
        st.header("📊 Live Market Analysis")
        
        # Analysis Controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            market_status = "🟢 OPEN" if market_open else "🔴 CLOSED"
            st.markdown(f"**Market Status:** {market_status} | **Time:** {now.strftime('%H:%M:%S')}")
        with col2:
            toggle_text = "⏹️ Stop Analysis" if st.session_state.analysis_running else "▶️ Start Analysis"
            if st.button(toggle_text, key="toggle_analysis"):
                if not st.session_state.analysis_running:
                    # Starting analysis - clear old signals
                    st.session_state.analysis_signals = []
                    st.session_state.analysis_running = True
                    st.success("🟢 Live analysis started!")
                    Logger.log_to_file("Live analysis started by user")
                else:
                    # Stopping analysis
                    st.session_state.analysis_running = False
                    st.success("🔴 Live analysis stopped!")
                    Logger.log_to_file("Live analysis stopped by user")
                st.rerun()
        with col3:
            if st.session_state.analysis_running:
                if st.button("🔄 Refresh Now", key="refresh_analysis"):
                    if market_open:
                        with st.spinner("📡 Scanning live market..."):
                            signals, status_message = LiveAnalysis.get_live_signals()
                            
                            # Display status
                            if "✅" in status_message:
                                st.success(status_message)
                            elif "❌" in status_message:
                                st.error(status_message)
                            elif "⚠️" in status_message:
                                st.warning(status_message)
                            else:
                                st.info(status_message)
                            
                            # Store new signals
                            if signals:
                                existing_timestamps = {s.get('timestamp', '') for s in st.session_state.analysis_signals}
                                new_signals_added = 0
                                
                                for signal in signals:
                                    if signal.get('timestamp') not in existing_timestamps:
                                        st.session_state.analysis_signals.append(signal)
                                        new_signals_added += 1
                                
                                # Keep last 100 signals
                                st.session_state.analysis_signals = st.session_state.analysis_signals[-100:]
                                
                                if new_signals_added > 0:
                                    st.success(f"✅ {new_signals_added} new signals detected!")
                                else:
                                    st.info("ℹ️ No new signals (same as previous scan)")
                    else:
                        st.warning("⏰ Cannot refresh - Market is closed")

        # Replace the "Display Analysis Results" section in main.py with this code:

        # Display Analysis Results
        if st.session_state.analysis_running:
            all_signals = st.session_state.analysis_signals if st.session_state.analysis_signals else []
            
            if all_signals:
                st.markdown(f"**📈 Live Trading Signals** *(Total: {len(all_signals)} signals)*")
                
                # Add filter controls
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    max_increase = st.number_input(
                        "Max Increase % Filter", 
                        min_value=0.0, 
                        max_value=100.0, 
                        value=100.0, 
                        step=0.1,
                        help="Show only signals with increase % less than or equal to this value"
                    )
                with col2:
                    min_increase = st.number_input(
                        "Min Increase % Filter", 
                        min_value=0.0, 
                        max_value=100.0, 
                        value=0.0, 
                        step=0.1,
                        help="Show only signals with increase % greater than or equal to this value"
                    )
                with col3:
                    st.markdown("**Filter Examples:**")
                    st.markdown("• Max 5% = Show signals ≤ 5%")
                    st.markdown("• Min 2% + Max 10% = Show 2% to 10%")
                
                # Convert signals to DataFrame for table display
                signal_data = []
                for signal in reversed(all_signals):
                    increase_val = signal['increase']
                    
                    # Apply filters
                    if min_increase <= increase_val <= max_increase:
                        signal_data.append({
                            'Symbol': signal['symbol'],
                            'CSV Date': signal['csv_date'],
                            'OHLC Type': signal['ohlc_type'],
                            'CSV Price': signal['csv_price'],  # Keep as number for sorting
                            'Live Price': signal['live_price'],  # Keep as number for sorting
                            'Increase %': increase_val,  # Keep as number for sorting
                            'Time': signal['current_time'],
                            'Volume': signal.get('volume', 'N/A'),
                            'Timestamp': signal.get('timestamp', '')
                        })
                
                if signal_data:
                    # Create DataFrame
                    df_signals = pd.DataFrame(signal_data)
                    
                    # Display filtered count
                    st.markdown(f"### 📊 Trading Signals Table (Showing: {len(signal_data)} signals)")
                    
                    # Configure column widths and styling with proper number formatting
                    st.dataframe(
                        df_signals,
                        use_container_width=True,
                        hide_index=True,
                        height=600,  # Set fixed height for better pagination
                        column_config={
                            "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                            "CSV Date": st.column_config.TextColumn("CSV Date", width="medium"),
                            "OHLC Type": st.column_config.TextColumn("OHLC", width="small"),
                            "CSV Price": st.column_config.NumberColumn(
                                "CSV Price (₹)", 
                                width="small",
                                format="₹%.2f"
                            ),
                            "Live Price": st.column_config.NumberColumn(
                                "Live Price (₹)", 
                                width="small",
                                format="₹%.2f"
                            ),
                            "Increase %": st.column_config.NumberColumn(
                                "Increase %", 
                                width="small",
                                format="%.2f%%"
                            ),
                            "Time": st.column_config.TextColumn("Time", width="small"),
                            "Volume": st.column_config.TextColumn("Volume", width="small"),
                            "Timestamp": None  # Hide timestamp column from display
                        }
                    )
                    
                    # Excel Download Section
                    st.markdown("### 📥 Export Options")
                    col1, col2, col3 = st.columns([1, 1, 2])
                    
                    with col1:
                        # Download as Excel (filtered data)
                        try:
                            import io
                            from datetime import datetime
                            
                            # Create Excel file in memory with filtered data
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                # Use the already filtered df_signals for Excel export
                                df_excel = df_signals.copy()
                                df_excel.to_excel(writer, sheet_name='Trading_Signals_Filtered', index=False)
                            
                            excel_data = output.getvalue()
                            
                            st.download_button(
                                label="📊 Download Excel (Filtered)",
                                data=excel_data,
                                file_name=f"trading_signals_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        except ImportError:
                            st.error("📋 Install openpyxl: pip install openpyxl")
                    
                    with col2:
                        # Download as CSV (filtered data)
                        csv_data = df_signals.to_csv(index=False)
                        st.download_button(
                            label="📋 Download CSV (Filtered)",
                            data=csv_data,
                            file_name=f"trading_signals_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col3:
                        st.info(f"📈 **{len(signal_data)} filtered signals** | Last updated: {datetime.now().strftime('%H:%M:%S')}")
                
                else:
                    st.warning(f"🔍 No signals found with increase between {min_increase}% and {max_increase}%")
                    st.info(f"Total signals available: {len(all_signals)}")
                    if st.button("🔄 Reset Filters"):
                        st.rerun()
                
            else:
                if market_open:
                    st.info("🔍 **Ready for live market analysis**")
                    st.markdown("""
                    **📋 Signal Detection:**
                    - ✅ Live price > CSV baseline price
                    - ✅ Real-time Kite API data
                    - ✅ Market hours: 9:15 AM - 3:30 PM
                    - ✅ Manual refresh control
                    - ✅ Filter by increase percentage
                    
                    Click **'🔄 Refresh Now'** to scan for trading signals.
                    """)
                else:
                    st.info("🕒 **Market is Closed**")
                    st.markdown(f"""
                    **Trading Hours:** 9:15 AM - 3:30 PM  
                    **Current Time:** {now.strftime('%H:%M:%S')}  
                    **Status:** Analysis available only during market hours.
                    """)
        else:
            st.info("📊 **Live Market Analysis**")
            st.markdown("""
            Click **'▶️ Start Analysis'** to enable live market monitoring.
            
            **🚀 Features:**
            - ✅ Real-time price comparison with baseline data
            - ✅ Manual refresh control - scan when you want  
            - ✅ Same signal logic as Telegram notifications
            - ✅ Market hours validation (9:15 AM - 3:30 PM)
            - ✅ No simulation - only actual market data
            - ✅ Table format with Excel/CSV export
            - ✅ Sortable columns with proper number formatting
            - ✅ Filter by increase percentage (min/max)
            - ✅ View all signals (no 15-limit pagination)
            """)

    else:
        st.header("🔐 Authentication Required")
        st.warning("Please authenticate with Kite Connect to access all trading features.")
        st.info("Use the sidebar to login with your Kite credentials.")

        

if __name__ == "__main__":
    Logger.log_to_file("=== TRADING SYSTEM DASHBOARD STARTED ===")
    main()