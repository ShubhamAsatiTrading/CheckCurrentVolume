import os
import sys
import logging
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('signals_system.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

TOKEN_FILE = "kite_token.txt"

def get_credentials():
    """Get API credentials from environment or token file"""
    access_token = os.getenv('TRADING_ACCESS_TOKEN')
    
    if not access_token:
        access_token = get_saved_token()
    
    api_key = os.getenv('KITE_API_KEY')
    
    if not access_token:
        logger.error("No access token available")
        return None, None
    
    if not api_key:
        logger.error("KITE_API_KEY not found in environment")
        return None, None
    
    return access_token, api_key

def get_saved_token():
    """Get saved token if valid for today"""
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r", encoding='utf-8') as f:
                token_data = json.loads(f.read().strip())
                today = datetime.now().strftime("%Y-%m-%d")
                
                if token_data.get("date") == today and token_data.get("access_token"):
                    logger.info("Using saved token from file")
                    return token_data["access_token"]
                else:
                    logger.info("Saved token expired")
                    os.remove(TOKEN_FILE)
        except Exception as e:
            logger.warning(f"Error reading token file: {e}")
            if os.path.exists(TOKEN_FILE):
                os.remove(TOKEN_FILE)
    return None

def setup_kite():
    """Initialize Kite connection"""
    try:
        from kiteconnect import KiteConnect
        
        access_token, api_key = get_credentials()
        if not access_token or not api_key:
            logger.error("Failed to get credentials")
            return None
        
        kite = KiteConnect(api_key=api_key, timeout=60)
        session = requests.Session()
        session.adapters.DEFAULT_RETRIES = 3
        kite._session = session
        kite.set_access_token(access_token)
        
        profile = kite.profile()
        logger.info(f"Logged in as: {profile['user_name']} (ID: {profile['user_id']})")
        return kite
        
    except Exception as e:
        logger.error(f"Kite setup failed: {e}")
        return None

def run_signals(kite):
    """Run the complete trading workflow for signals"""
    try:
        from consolidated_volume import complete_trading_workflow
        
        logger.info("Running volume boost scanner and signals...")
        
        complete_trading_workflow(
            interval_minutes=0, 
            interval_days=1, 
            kite_instance=kite,
            telegram_bot_token=os.getenv('TELEGRAM_BOT_TOKEN'),
            telegram_chat_id=os.getenv('TELEGRAM_CHAT_ID')
        )
        
        logger.info("Signals workflow completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in signals workflow: {e}")
        return False

def main():
    """Main function to run signals every hour"""
    logger.info("SIGNALS SYSTEM STARTING")
    logger.info("=" * 50)
    
    # Setup Kite connection once
    kite = setup_kite()
    if not kite:
        logger.error("Failed to setup Kite connection")
        return
    
    logger.info("Starting hourly signal scanner...")
    
    while True:
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Running signals at {current_time}")
            
            # Run signals
            success = run_signals(kite)
            
            if success:
                logger.info("Signals run completed - waiting 1 hour for next run")
            else:
                logger.warning("Signals run failed - will retry in 1 hour")
            
            # Wait for 1 hour (3600 seconds)
            logger.info("Sleeping for 1 hour...")
            time.sleep(3600)
            
        except KeyboardInterrupt:
            logger.info("Signal system stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            logger.info("Waiting 1 hour before retry...")
            time.sleep(3600)

if __name__ == "__main__":
    main()