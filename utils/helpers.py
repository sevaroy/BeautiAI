import io
import base64
import logging
from typing import Optional
from PIL import Image, ImageFile
import json
import os
from datetime import datetime

# Allow Pillow to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

def encode_image_to_base64(image_file: io.BytesIO) -> str:
    """Convert an image file to base64 string."""
    try:
        return base64.b64encode(image_file.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image to base64: {str(e)}")
        raise

def save_api_response(response_type: str, response_data) -> Optional[str]:
    """Save API response to corresponding folder."""
    try:
        # Create directory if it doesn't exist
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = f"temp/{response_type}"
        os.makedirs(directory, exist_ok=True)
        
        # Save response
        filename = f"{directory}/{timestamp}.json"
        
        # 處理不同類型的響應對象
        if hasattr(response_data, 'model_dump'):
            # Pydantic v2 方法
            data_to_save = response_data.model_dump()
        elif hasattr(response_data, 'dict'):
            # Pydantic v1 方法
            data_to_save = response_data.dict()
        else:
            # 普通字典或其他可序列化物件
            data_to_save = response_data
            
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved {response_type} response to {filename}")
        return filename
    except Exception as e:
        logger.error(f"保存API响应失败: {str(e)}")
        return None

def validate_image(image_file: io.BytesIO) -> Optional[Image.Image]:
    """Validate and process uploaded image."""
    try:
        # Reset file pointer to beginning
        image_file.seek(0)
        
        # Open and load the image
        image = Image.open(image_file)
        image.load()  # Explicitly load the image to catch truncation issues
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Validate image size
        if image.size[0] < 100 or image.size[1] < 100:
            raise ValueError("Image is too small")
        
        # Reset file pointer for future use
        image_file.seek(0)
            
        return image
    except OSError as e:
        logger.error(f"Error loading image (possibly truncated): {str(e)}")
        raise ValueError(f"圖片可能已損壞或不完整: {str(e)}")
    except Exception as e:
        logger.error(f"Error validating image: {str(e)}")
        raise

def get_text(key: str, lang: str = "zh") -> str:
    """Get localized text."""
    texts = {
        "zh": {
            "upload_title": "上傳照片",
            "upload_help": "支援 JPG、JPEG 和 PNG 格式的圖片",
            "analysis_title": "AI 分析結果",
            "report_title": "評估報告",
            "download_report": "下載 PDF 報告",
            "error_processing": "處理照片時發生錯誤，請重試",
            "waiting_analysis": "等待分析...",
        }
    }
    return texts.get(lang, {}).get(key, key)
