import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# API Keys
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
GROK_API_KEY = os.getenv("XAI_API_KEY")  # Using XAI_API_KEY for Grok model as well

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Application settings
APP_NAME = "醫美診所智能評估系統"
VERSION = "1.0.0"

# File upload settings
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Report generation settings
REPORT_FONT_PATH = "static/fonts/msyh.ttc"
DEFAULT_REPORT_FILENAME = "醫美診所智能評估報告.pdf"

# Analysis settings
FACE_DETECTION_CONFIDENCE = 0.8
ANALYSIS_TIMEOUT = 30  # seconds

# Cache settings
CACHE_ENABLED = True
CACHE_TIMEOUT = 3600  # 1 hour
