# main.py - Trading System Streamlit Web Interface with Smart Token Management
# This file provides the web UI for the trading system with automatic token management

import streamlit as st
import subprocess
import os
import pandas as pd
from datetime import datetime
import time
import json
import threading
import queue

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
CONFIG_FILE = "common.txt"
TOKEN_FILE = "kite_token.txt"
REQUIRED_DIRS = ["stocks_historical_data", "aggregated_data", "Volume_boost_consolidated"]

# Kite Connect Configuration from environment
KITE_API_KEY = os.getenv('KITE_API_KEY', '')
KITE_API_SECRET = os.getenv('KITE_API_SECRET', '')

st.set_page_config(page_title="Trading System", page_icon="[TRADING]", layout="wide")

# CSS
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
                        return token_data["access_token"]
                    else:
                        os.remove(TOKEN_FILE)  # Delete old token
            except:
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
            return True
        except:
            return False
    
    @staticmethod
    def generate_login_url():
        """Generate Kite login URL"""
        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=KITE_API_KEY)
            return kite.login_url()
        except Exception as e:
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
            
            # Save the token
            if TokenManager.save_token(access_token):
                return access_token
            return None
        except Exception as e:
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
                "status": "[SUCCESS] Valid token available",
                "date": datetime.now().strftime("%Y-%m-%d")
            }
        else:
            return {
                "has_token": False,
                "token": None,
                "status": "[ERROR] No valid token",
                "date": None
            }

class ConfigManager:
    @staticmethod
    def load():
        defaults = {"stop_loss": 4.0, "target": 10.0, "ohlc_value": "open", "trade_today_flag": "no"}
        try:
            config = {}
            with open(CONFIG_FILE, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        config[key] = float(value) if key in ['stop_loss', 'target'] else value
            return {**defaults, **config}
        except:
            ConfigManager.save(defaults)
            return defaults
    
    @staticmethod
    def save(config):
        try:
            with open(CONFIG_FILE, 'w') as f:
                for key, value in config.items():
                    f.write(f"{key}={value}\n")
            return True
        except:
            return False

class TradingSystem:
    @staticmethod
    def validate_setup():
        missing = [d for d in REQUIRED_DIRS if not os.path.exists(d)]
        return len(missing) == 0, missing
    
    @staticmethod
    def run_main_script_with_streaming(access_token, progress_container, log_container):
        """Run main script with real-time output streaming"""
        import sys
        env = os.environ.copy()
        env.update({
            'TRADING_ACCESS_TOKEN': access_token,
            'KITE_API_KEY': KITE_API_KEY,
            'KITE_API_SECRET': KITE_API_SECRET,
            'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN', ''),
            'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID', '')
        })
        
        # Progress tracking steps
        progress_steps = [
            "[LOADING] Starting Trading System...",
            "[DATA] Scraping stock fundamentals from screener.in...",
            "[LINK] Testing kiteconnect import...",
            "[AUTH] Initializing authentication...",
            "[TRADING] Downloading live/historical data...",
            "[INFO] Aggregating data...",
            "[TEST] Running backtest...",
            "[SCANNER] Running live scanner & notifications...",
            "[SUCCESS] Trading system completed!"
        ]
        
        # Initialize progress
        progress_container.markdown("### [INFO] Progress")
        progress_placeholders = []
        for i, step in enumerate(progress_steps):
            placeholder = progress_container.empty()
            progress_placeholders.append(placeholder)
        
        # Initialize log container
        log_container.markdown("### 📜 Live Logs")
        log_placeholder = log_container.empty()
        
        try:
            # Check if symbols.txt exists and show its contents
            symbols_info = ""
            if os.path.exists('symbols.txt'):
                with open('symbols.txt', 'r') as f:
                    symbols = [line.strip() for line in f if line.strip()]
                    symbols_info = f"[INFO] **Symbols from symbols.txt**: {', '.join(symbols)} ({len(symbols)} symbols)"
            else:
                symbols_info = "[WARNING] **symbols.txt not found** - system may use default symbols"
            
            log_placeholder.markdown(f"""
            <div class="log-container">
            <strong>[START] TRADING SYSTEM STARTUP LOG</strong><br/>
            <strong>[TIME] Started at:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
            {symbols_info}<br/>
            <strong>[CONFIG] Environment:</strong> Python {sys.version.split()[0]}<br/>
            <strong> Working Directory:</strong> {os.getcwd()}<br/>
            <br/>
            <strong>[DATA] Real-time output:</strong><br/>
            <div id="log-content">Initializing...</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Update first progress step
            progress_placeholders[0].markdown('<div class="progress-step">[LOADING] Starting Trading System...</div>', unsafe_allow_html=True)
            
            # Start the subprocess
            process = subprocess.Popen(
                [sys.executable, 'code_1.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=env
            )
            
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
                    log_content.append(line.rstrip())
                    
                    # Update progress based on keywords
                    for i, keyword in enumerate(step_keywords):
                        if keyword.lower() in line.lower() and i > current_step:
                            # Mark previous steps as completed
                            for j in range(current_step + 1):
                                if j < len(progress_steps):
                                    progress_placeholders[j].markdown(
                                        f'<div class="progress-step completed">[SUCCESS] {progress_steps[j]}</div>', 
                                        unsafe_allow_html=True
                                    )
                            
                            # Mark current step as active
                            if i < len(progress_steps):
                                progress_placeholders[i].markdown(
                                    f'<div class="progress-step">[LOADING] {progress_steps[i]}</div>', 
                                    unsafe_allow_html=True
                                )
                            current_step = i
                            break
                    
                    # Check for errors
                    if any(error_word in line.lower() for error_word in ['error', 'failed', 'exception']):
                        if current_step < len(progress_steps):
                            progress_placeholders[current_step].markdown(
                                f'<div class="progress-step error">[ERROR] {progress_steps[current_step]} - Error detected</div>', 
                                unsafe_allow_html=True
                            )
                    
                    # Update log display (last 30 lines for readability)
                    recent_logs = log_content[-30:] if len(log_content) > 30 else log_content
                    formatted_logs = "<br/>".join([
                        f"<span style='color: #666;'>{i+1:3d}:</span> {line}" 
                        for i, line in enumerate(recent_logs)
                    ])
                    
                    log_placeholder.markdown(f"""
                    <div class="log-container">
                    <strong>[START] TRADING SYSTEM LIVE LOG</strong> <small>(Showing last 30 lines)</small><br/>
                    <strong>[TIME] Started at:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
                    {symbols_info}<br/>
                    <br/>
                    {formatted_logs}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Small delay to prevent too rapid updates
                    time.sleep(0.1)
            
            # Wait for process to complete
            return_code = process.wait()
            
            # Final update
            if return_code == 0:
                # Mark all steps as completed
                for i, step in enumerate(progress_steps):
                    progress_placeholders[i].markdown(
                        f'<div class="progress-step completed">[SUCCESS] {step}</div>', 
                        unsafe_allow_html=True
                    )
                return True, "\n".join(log_content), ""
            else:
                # Mark current step as error
                if current_step < len(progress_steps):
                    progress_placeholders[current_step].markdown(
                        f'<div class="progress-step error">[ERROR] {progress_steps[current_step]} - Failed</div>', 
                        unsafe_allow_html=True
                    )
                return False, "\n".join(log_content), f"Process exited with code {return_code}"
                
        except Exception as e:
            error_msg = f"Exception occurred: {str(e)}"
            log_placeholder.markdown(f"""
            <div class="log-container">
            <strong style="color: red;">[CRITICAL] CRITICAL ERROR</strong><br/>
            {error_msg}<br/>
            <br/>
            <strong>Logs before error:</strong><br/>
            {"<br/>".join(log_content) if 'log_content' in locals() else "No logs captured"}
            </div>
            """, unsafe_allow_html=True)
            return False, "", error_msg
    
    @staticmethod
    def run_main_script(access_token):
        """Legacy method for backward compatibility"""
        import sys
        env = os.environ.copy()
        env.update({
            'TRADING_ACCESS_TOKEN': access_token,
            'KITE_API_KEY': KITE_API_KEY,
            'KITE_API_SECRET': KITE_API_SECRET,
            'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN', ''),
            'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID', '')
        })
        
        try:
            result = subprocess.run([sys.executable, 'code_1.py'], 
                                  capture_output=True, text=True, env=env, timeout=300)
            return result.returncode == 0, result.stdout, result.stderr
        except Exception as e:
            return False, "", str(e)
    
    @staticmethod
    def get_signals():
        """Get real trading signals from consolidated data - NO SIMULATION"""
        try:
            # Only work with real data - no simulation
            if not os.path.exists("Volume_boost_consolidated/consolidated_data.csv"):
                return []
            
            df = pd.read_csv("Volume_boost_consolidated/consolidated_data.csv")
            config = ConfigManager.load()
            ohlc_col = config['ohlc_value']
            
            # This would require real Kite API integration for live prices
            # For now, return empty array since we want NO simulation
            # In production, this would fetch real live prices from Kite API
            
            signals = []
            # TODO: Integrate with real Kite API for live price fetching
            # Example implementation:
            # for _, row in df.iterrows():
            #     live_price = kite.ltp(f"NSE:{row['symbol']}")['NSE:{row['symbol']}']['last_price']
            #     if live_price > row[ohlc_col]:
            #         signals.append({...})
            
            return signals
        except:
            return []

def main():
    st.title("[START] Trading System Dashboard")
    
    # Initialize session state
    for key in ['config', 'signals', 'system_running']:
        if key not in st.session_state:
            if key == 'config':
                st.session_state[key] = ConfigManager.load()
            elif key == 'signals':
                st.session_state[key] = []
            else:
                st.session_state[key] = False
    
    # Check token status
    token_status = TokenManager.get_token_status()
    
    # Sidebar - Controls
    with st.sidebar:
        st.header("[CONFIG] Controls")
        
        # Validation
        setup_valid, missing_dirs = TradingSystem.validate_setup()
        if not setup_valid:
            st.error(f"Missing directories: {missing_dirs}")
            return
        
        # Token Management Section
        st.subheader("[AUTH] Authentication")
        
        if not KITE_API_KEY or not KITE_API_SECRET:
            st.error("[ERROR] KITE_API_KEY or KITE_API_SECRET not found in .env file!")
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
                if st.button("[LOADING] Refresh Token"):
                    if os.path.exists(TOKEN_FILE):
                        os.remove(TOKEN_FILE)
                    st.rerun()
            with col2:
                if st.button("🗑️ Clear Token"):
                    if os.path.exists(TOKEN_FILE):
                        os.remove(TOKEN_FILE)
                    st.success("Token cleared!")
                    st.rerun()
            
            # Use the valid token
            access_token = token_status["token"]
            
        else:
            st.warning(token_status["status"])
            
            # Only show Kite login method
            st.write("**Authenticate with Kite Connect:**")
            
            # Method: Generate via Kite login
            if st.button("[LINK] Get Kite Login URL"):
                login_url = TokenManager.generate_login_url()
                if login_url:
                    st.markdown(f"**Click here to login:** [Kite Login]({login_url})")
                    st.info("After login, copy the request_token from URL and paste below")
                else:
                    st.error("Failed to generate login URL")
            
            request_token = st.text_input("Paste Request Token:", 
                                        help="Copy from URL after login (parameter: request_token=...)")
            
            if st.button("[SUCCESS] Generate Access Token") and request_token:
                with st.spinner("Generating access token..."):
                    access_token = TokenManager.generate_session_from_request_token(request_token.strip())
                    if access_token:
                        st.success("[SUCCESS] Access token generated and saved!")
                        st.rerun()
                    else:
                        st.error("[ERROR] Failed to generate access token")
            
            access_token = None  # No token available yet
        
        # Configuration (only show if we have token)
        if token_status["has_token"] or access_token:
            st.markdown("---")
            st.subheader("⚙️ Configuration")
            config = st.session_state.config.copy()
            
            config['stop_loss'] = st.number_input("Stop Loss %", 0.1, 20.0, config['stop_loss'], 0.1)
            config['target'] = st.number_input("Target %", 0.1, 50.0, config['target'], 0.1)
            config['ohlc_value'] = st.selectbox("OHLC", ["open", "high", "low", "close"], 
                                               ["open", "high", "low", "close"].index(config['ohlc_value']))
            config['trade_today_flag'] = st.selectbox("Trade Flag", ["yes", "no"], 
                                                      ["yes", "no"].index(config['trade_today_flag']))
            
            if st.button("[SAVE] Update Config"):
                if ConfigManager.save(config):
                    st.session_state.config = config
                    st.success("Configuration updated!")
                else:
                    st.error("Failed to update configuration")
            
            st.markdown("---")
            
            # File status check
            st.subheader(" File Status")
            
            # Check symbols.txt
            if os.path.exists('symbols.txt'):
                with open('symbols.txt', 'r') as f:
                    symbols = [line.strip() for line in f if line.strip()]
                st.success(f"[SUCCESS] symbols.txt: {len(symbols)} symbols")
                with st.expander("View symbols"):
                    st.write(", ".join(symbols))
            else:
                st.warning("[WARNING] symbols.txt not found")
            
            # Check other important files
            important_files = ['code_1.py', 'common.txt', '.env']
            for file in important_files:
                if os.path.exists(file):
                    st.success(f"[SUCCESS] {file}")
                else:
                    st.error(f"[ERROR] {file} missing")
            
            st.markdown("---")
            
            # Actions (only show if we have token)
            if st.button("[START] Run Trading System", type="primary"):
                final_token = token_status["token"] if token_status["has_token"] else access_token
                
                # Create containers for progress and logs
                st.header("[LOADING] Trading System Execution")
                
                progress_container = st.container()
                log_container = st.container()
                
                # Run with streaming output
                success, stdout, stderr = TradingSystem.run_main_script_with_streaming(
                    final_token, progress_container, log_container
                )
                
                # Final status
                if success:
                    st.success("[COMPLETE] Trading system completed successfully!")
                else:
                    st.error("[ERROR] Trading system encountered errors!")
                    if stderr:
                        st.error(f"Error details: {stderr}")
            
            if st.button("[SCANNER] Toggle Live Signals"):
                st.session_state.system_running = not st.session_state.system_running
                status = "started" if st.session_state.system_running else "stopped"
                st.success(f"Live signals {status}!")
    
    # Main Dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    # Status Cards
    now = datetime.now()
    market_open = 9 <= now.hour <= 15
    
    with col1:
        st.markdown(f"""<div class="metric-card">
        <h3>Market</h3><h2>{'[ACTIVE] OPEN' if market_open else '[INACTIVE] CLOSED'}</h2>
        <p>{now.strftime('%H:%M:%S')}</p></div>""", unsafe_allow_html=True)
    
    with col2:
        token_color = "[ACTIVE]" if token_status["has_token"] else "[INACTIVE]"
        token_text = "AUTHENTICATED" if token_status["has_token"] else "NO TOKEN"
        st.markdown(f"""<div class="metric-card">
        <h3>Authentication</h3><h2>{token_color} {token_text}</h2>
        <p>{"Valid for today" if token_status["has_token"] else "Login required"}</p></div>""", 
        unsafe_allow_html=True)
    
    with col3:
        status = st.session_state.config['trade_today_flag']
        st.markdown(f"""<div class="metric-card">
        <h3>Trading</h3><h2>{'[ACTIVE]' if status == 'yes' else '[INACTIVE]'} {status.upper()}</h2>
        <p>SL: {st.session_state.config['stop_loss']}% | T: {st.session_state.config['target']}%</p></div>""", 
        unsafe_allow_html=True)
    
    with col4:
        signals_count = len(st.session_state.signals)
        st.markdown(f"""<div class="metric-card">
        <h3>Signals</h3><h2>[DATA] {signals_count}</h2>
        <p>Live Updates</p></div>""", unsafe_allow_html=True)
    
    # Live Signals (only show if authenticated)
    if token_status["has_token"]:
        st.header("[TRADING] Live Signals")
        
        if st.session_state.system_running:
            signals = TradingSystem.get_signals()
            if signals:
                st.session_state.signals.extend(signals)
                
                # Keep last 50 signals
                st.session_state.signals = st.session_state.signals[-50:]
            
            # Display recent signals
            recent_signals = st.session_state.signals[-10:] if st.session_state.signals else []
            
            if recent_signals:
                for signal in reversed(recent_signals):
                    signal_color = {"BUY": "[ACTIVE]", "WATCH": "🟡", "SELL": "[INACTIVE]"}
                    st.markdown(f"""
                    <div class="signal-card">
                        <strong>{signal_color.get(signal['signal'], '⚪')} {signal['signal']} {signal['symbol']}</strong> | 
                        Live: ₹{signal['live_price']:.2f} | CSV: ₹{signal['csv_price']:.2f} | 
                        +{signal['increase']:.1f}% | {signal['time']}
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("[LOADING] Monitoring for real signals... No simulation - only actual market data will appear here.")
                st.write("**Note**: Signals will only appear when:")
                st.write("• Market is open (9:15 AM - 3:30 PM)")
                st.write("• Live prices exceed CSV baseline prices")
                st.write("• Consolidated data file exists")
            
            time.sleep(10)  # Refresh every 10 seconds for real signals
            st.rerun()
        else:
            st.info("Click 'Toggle Live Signals' in sidebar to start monitoring real market data")
            st.warning("**No simulation mode** - Only real signals during market hours will be displayed")
    else:
        st.header("[AUTH] Authentication Required")
        st.warning("Please authenticate with Kite Connect to access live signals and trading features.")
        st.info("Use the sidebar to login via Kite Connect.")

if __name__ == "__main__":
    main()

