# log_runner.py - Run trading system with complete logging to file
# Save this as log_runner.py and run: python log_runner.py

import subprocess
import sys
import os
from datetime import datetime
from dotenv import load_dotenv
import json

class TradingLogger:
    def __init__(self):
        self.log_file = f"trading_system_full_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.setup_logging()
    
    def setup_logging(self):
        """Initialize the log file"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== TRADING SYSTEM FULL LOG ===\n")
                f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Log file: {os.path.abspath(self.log_file)}\n")
                f.write("=" * 80 + "\n\n")
            print(f"[LOG] Log file created: {os.path.abspath(self.log_file)}")
        except Exception as e:
            print(f"[ERROR] Failed to create log file: {e}")
    
    def log(self, message, level="INFO"):
        """Log message to file and console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log_entry = f"[{timestamp}] {level}: {message}"
        
        # Print to console
        print(log_entry)
        
        # Write to file
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + "\n")
        except Exception as e:
            print(f"[ERROR] Failed to write to log: {e}")
    
    def get_log_path(self):
        return os.path.abspath(self.log_file)

def get_access_token():
    """Get access token from file or user input"""
    token_file = "kite_token.txt"
    
    # Try to get from saved file first
    if os.path.exists(token_file):
        try:
            with open(token_file, "r") as f:
                token_data = json.loads(f.read().strip())
                today = datetime.now().strftime("%Y-%m-%d")
                
                if token_data.get("date") == today and token_data.get("access_token"):
                    print(f"[SUCCESS] Found valid token for {today}")
                    return token_data["access_token"]
                else:
                    print("[WARNING] Saved token expired")
        except Exception as e:
            print(f"[WARNING] Error reading token file: {e}")
    
    # Ask user for token
    print("\n🔑 ACCESS TOKEN REQUIRED")
    print("You can get this by:")
    print("1. Running the web interface (python run.py) and authenticating")
    print("2. Or manually getting token from Kite Connect")
    
    access_token = input("\nEnter your Kite access token (or press Enter to skip): ").strip()
    
    if not access_token:
        print("[WARNING] No token provided. Some features may not work.")
        return None
    
    return access_token

def check_environment():
    """Check if environment is properly set up"""
    issues = []
    
    # Check files
    required_files = ['code_1.py', '.env']
    for file in required_files:
        if not os.path.exists(file):
            issues.append(f"Missing file: {file}")
    
    # Check environment variables
    load_dotenv()
    required_env = ['KITE_API_KEY', 'KITE_API_SECRET']
    for env_var in required_env:
        if not os.getenv(env_var):
            issues.append(f"Missing environment variable: {env_var}")
    
    # Check Python packages
    try:
        import kiteconnect
        print(f"[SUCCESS] kiteconnect version: {kiteconnect.__version__}")
    except ImportError:
        issues.append("kiteconnect package not installed")
    
    if issues:
        print("[ERROR] Environment issues found:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    
    print("[SUCCESS] Environment check passed")
    return True

def run_trading_system_with_full_logging():
    """Run the trading system with complete logging"""
    
    logger = TradingLogger()
    
    print("[START] TRADING SYSTEM - FULL LOGGING MODE")
    print("=" * 60)
    print(f"[LOG] All logs will be saved to: {logger.get_log_path()}")
    print("=" * 60)
    
    logger.log("=== TRADING SYSTEM STARTUP ===")
    logger.log(f"Python executable: {sys.executable}")
    logger.log(f"Python version: {sys.version}")
    logger.log(f"Working directory: {os.getcwd()}")
    
    # Environment check
    logger.log("Checking environment...")
    if not check_environment():
        logger.log("Environment check failed", "ERROR")
        return False
    
    # Get access token
    logger.log("Getting access token...")
    access_token = get_access_token()
    if access_token:
        logger.log("Access token obtained")
    else:
        logger.log("No access token provided", "WARNING")
    
    # Setup environment
    logger.log("Setting up environment variables...")
    load_dotenv()
    env = os.environ.copy()
    
    if access_token:
        env['TRADING_ACCESS_TOKEN'] = access_token
    
    env.update({
        'KITE_API_KEY': os.getenv('KITE_API_KEY', ''),
        'KITE_API_SECRET': os.getenv('KITE_API_SECRET', ''),
        'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN', ''),
        'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID', ''),
        'PYTHONPATH': os.getcwd(),
        'PYTHONUNBUFFERED': '1',
    })
    
    logger.log(f"Environment variables set: {len(env)} total")
    
    # Check if code_1.py exists
    if not os.path.exists('code_1.py'):
        logger.log("code_1.py not found in current directory", "ERROR")
        return False
    
    logger.log("Starting code_1.py subprocess...")
    
    try:
        # Start the subprocess
        process = subprocess.Popen(
            [sys.executable, 'code_1.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env
        )
        
        logger.log(f"Subprocess started with PID: {process.pid}")
        
        # Real-time output streaming
        line_number = 1
        for line in iter(process.stdout.readline, ''):
            if line:
                line_clean = line.rstrip()
                logger.log(f"[{line_number:4d}] {line_clean}")
                line_number += 1
        
        # Wait for process to complete
        return_code = process.wait()
        logger.log(f"Subprocess completed with return code: {return_code}")
        
        if return_code == 0:
            logger.log("=== TRADING SYSTEM COMPLETED SUCCESSFULLY ===")
            print("\n[SUCCESS] TRADING SYSTEM COMPLETED SUCCESSFULLY!")
        else:
            logger.log(f"=== TRADING SYSTEM FAILED WITH CODE {return_code} ===", "ERROR")
            print(f"\n[ERROR] TRADING SYSTEM FAILED WITH CODE {return_code}")
        
        print(f"\n[LOG] Complete logs saved to: {logger.get_log_path()}")
        return return_code == 0
        
    except KeyboardInterrupt:
        logger.log("Process interrupted by user", "WARNING")
        print("\n\n[WARNING] Interrupted by user")
        try:
            process.terminate()
        except:
            pass
        return False
        
    except Exception as e:
        logger.log(f"CRITICAL ERROR: {str(e)}", "ERROR")
        print(f"\n[CRITICAL] CRITICAL ERROR: {e}")
        return False

def main():
    """Main function"""
    success = run_trading_system_with_full_logging()
    
    print("\n" + "=" * 60)
    if success:
        print("[COMPLETE] Trading system completed successfully!")
    else:
        print("💔 Trading system encountered issues.")
    
    print("\n[TIP] Tips:")
    print("• Check the log file for complete details")
    print("• Compare this run with direct 'python code_1.py' execution")
    print("• Ensure all environment variables are set correctly")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()

