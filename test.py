from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

complete_trading_workflow(
                      interval_minutes=0, interval_days=1, 
                      kite_instance=kite,
                      telegram_bot_token=os.getenv('TELEGRAM_BOT_TOKEN'),
                      telegram_chat_id=os.getenv('TELEGRAM_CHAT_ID'))