import requests
from constants import *

def telegram_log(message):
    try:
        response = requests.get(
            'https://api.telegram.org/bot' + TELEGRAM_TOKEN + '/sendMessage',
            json = {
                'chat_id': TELEGRAM_CHAT_ID, 
                'parse_mode': 'Markdown', 
                'text': message
            }
        )
    except:
        None