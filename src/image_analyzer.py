import io
import os
import base64
import logging
import numpy as np
import uuid
import time
from typing import Dict, Any, List, Tuple
from PIL import Image as PILImage, ImageFile
import cv2
import replicate
from openai import OpenAI
import dlib
import time
import streamlit as st
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageAnalyzer:
    def __init__(self):
        """
        Initialize the ImageAnalyzer with API clients.
        """
        # Initialize API clients
        self.xai_api_key = os.environ.get("XAI_API_KEY")
        
        # 檢查 API 金鑰格式並設置正確的 base URL
        if self.xai_api_key:
            if self.xai_api_key.startswith("sk-proj-"):
                # 新的 OpenAI API 金鑰格式
                self.xai_client = OpenAI(api_key=self.xai_api_key)
                self.xai_base_url = "https://api.openai.com/v1"
                logger.info("Initialized OpenAI client with project API key")
            elif self.xai_api_key.startswith("xai-"):
                # X AI 的 API 金鑰格式
                self.xai_base_url = "https://api.x.ai/v1"
                logger.info("Initialized X AI client")
            else:
                logger.warning("Unknown API key format")
                self.xai_base_url = "https://api.openai.com/v1"  # 默認使用 OpenAI
        else:
            logger.warning("XAI_API_KEY not found in environment variables")
        
        # Initialize Replicate client
        self.replicate_api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not self.replicate_api_token:
            logger.warning("REPLICATE_API_TOKEN not found in environment variables")
        os.environ["REPLICATE_API_TOKEN"] = self.replicate_api_token
        
        self.face_detector = dlib.get_frontal_face_detector()

    def analyze_image(self, image_file: io.BytesIO, model: str = "DeepSeek VL2") -> Dict[str, Any]:
        try:
            # Create progress placeholders if Streamlit is running
            progress_placeholder = None
            progress_bar = None
            status_text = None
            
            try:
                # Check if we're in a Streamlit context
                if st._is_running:
                    progress_placeholder = st.empty()
                    progress_bar = progress_placeholder.progress(0)
                    status_text = st.empty()
                    status_text.text("Initializing image analysis...")
            except:
                # Not in Streamlit context or error occurred
                logger.info("Not in Streamlit context or error initializing progress indicators")
                pass
            
            # Update progress - 10%
            self._update_progress(progress_bar, status_text, 10, "Loading and processing image...")
            
            # Set to handle truncated images
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            
            # Reset file pointer to the beginning
            image_file.seek(0)
            
            # Open and explicitly load the image
            image = PILImage.open(image_file)
            image.load()
            
            # Resize and compress the image to reduce size
            max_size = (800, 800)  # Reduced maximum dimensions
            image.thumbnail(max_size, PILImage.LANCZOS)
            
            # Update progress - 20%
            self._update_progress(progress_bar, status_text, 20, "Optimizing image...")
            
            # Convert to RGB if it's not already (handles RGBA, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save to a BytesIO object with reduced quality
            compressed_image = io.BytesIO()
            image.save(compressed_image, format='JPEG', quality=70)  # Lower quality for smaller size
            compressed_image.seek(0)
            
            # Get the size of the compressed image in bytes
            compressed_size = len(compressed_image.getvalue())
            logger.info(f"Compressed image size: {compressed_size / 1024:.2f} KB")
            
            # If still too large, compress further
            if compressed_size > 1000000:  # 1MB
                logger.warning("Image still too large, compressing further")
                compressed_image.seek(0)
                image = PILImage.open(compressed_image)
                image.thumbnail((600, 600), PILImage.LANCZOS)  # Even smaller dimensions
                
                # Create a new BytesIO object with even lower quality
                compressed_image = io.BytesIO()
                image.save(compressed_image, format='JPEG', quality=50)
                compressed_image.seek(0)
                
                logger.info(f"Further compressed image size: {len(compressed_image.getvalue()) / 1024:.2f} KB")
            
            # Update progress - 30%
            self._update_progress(progress_bar, status_text, 30, "Detecting facial features...")
            
            # Get face regions from the original image for better detection
            image_file.seek(0)
            original_image = PILImage.open(image_file)
            face_regions = self.detect_face_regions(np.array(original_image))
            
            # Get analysis based on model
            if model == "DeepSeek VL2":
                # Update progress - 40%
                self._update_progress(progress_bar, status_text, 40, "Preparing for DeepSeek VL2 analysis...")
                
                # Save compressed image to a temporary file with unique name
                compressed_image.seek(0)
                temp_image_path = f"temp_image_{uuid.uuid4().hex}.jpg"
                with open(temp_image_path, "wb") as f:
                    f.write(compressed_image.getvalue())
                
                # Update progress - 50%
                self._update_progress(progress_bar, status_text, 50, "Sending image to DeepSeek VL2...")
                
                # Use Replicate API for DeepSeek VL2
                analysis_result = self._get_deepseek_analysis(temp_image_path, progress_bar, status_text)
                
                # Update progress - 90%
                self._update_progress(progress_bar, status_text, 90, "Analysis complete, cleaning up...")
                
                # Remove temporary file with retry mechanism
                self._safely_remove_file(temp_image_path)
                
            elif model == "grok-2-vision-1212":
                # Update progress - 40%
                self._update_progress(progress_bar, status_text, 40, "Preparing for grok-2-vision-1212 analysis...")
                
                # Use X AI API directly
                compressed_image.seek(0)
                image_base64 = base64.b64encode(compressed_image.read()).decode('utf-8')
                
                # Update progress - 50%
                self._update_progress(progress_bar, status_text, 50, "Sending image to grok-2-vision-1212 model...")
                
                analysis_result = self._get_xai_analysis(image_base64, progress_bar, status_text)
            
            elif model == "GPT-4o":
                # Redirect to DeepSeek VL2 since GPT-4o is not supported
                logger.info("GPT-4o model selected but not supported, using DeepSeek VL2 instead")
                
                # Update progress - 40%
                self._update_progress(progress_bar, status_text, 40, "Preparing for DeepSeek VL2 analysis (fallback)...")
                
                # Save compressed image to a temporary file with unique name
                compressed_image.seek(0)
                temp_image_path = f"temp_image_{uuid.uuid4().hex}.jpg"
                with open(temp_image_path, "wb") as f:
                    f.write(compressed_image.getvalue())
                
                # Update progress - 50%
                self._update_progress(progress_bar, status_text, 50, "Sending image to DeepSeek VL2...")
                
                # Use Replicate API for DeepSeek VL2
                analysis_result = self._get_deepseek_analysis(temp_image_path, progress_bar, status_text)
                
                # Update progress - 90%
                self._update_progress(progress_bar, status_text, 90, "Analysis complete, cleaning up...")
                
                # Remove temporary file with retry mechanism
                self._safely_remove_file(temp_image_path)
            
            else:
                return {"error": f"不支持的模型: {model}"}
            
            # Update progress - 95%
            self._update_progress(progress_bar, status_text, 95, "Processing results...")
            
            # Check if analysis_result contains an error
            if isinstance(analysis_result, dict) and "error" in analysis_result:
                return analysis_result  # Return the error directly
            
            # If analysis_result is a string, wrap it in a dictionary
            if isinstance(analysis_result, str):
                analysis_result = {
                    "model": model,
                    "result": analysis_result
                }
            
            # Update progress - 100%
            self._update_progress(progress_bar, status_text, 100, "Analysis complete!")
            
            # Clear progress indicators
            if progress_placeholder:
                try:
                    progress_placeholder.empty()
                    status_text.empty()
                except:
                    pass
            
            return {'face_regions': face_regions, 'analysis': analysis_result}
            
        except OSError as e:
            logger.error(f"Image loading error: {str(e)}")
            return {"error": f"Failed to load image: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error during image analysis: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}

    def _update_progress(self, progress_bar, status_text, progress_value, message):
        """
        Update progress indicators if they exist
        """
        try:
            if progress_bar:
                progress_bar.progress(progress_value)
            if status_text:
                status_text.text(message)
            # Log progress regardless of UI
            logger.info(f"Progress {progress_value}%: {message}")
        except:
            # If updating progress fails, just log it
            logger.info(f"Progress {progress_value}%: {message}")
            pass

    def _safely_remove_file(self, file_path, max_attempts=5, delay=0.5):
        """
        Safely remove a file with retry mechanism to handle file access conflicts
        """
        if not os.path.exists(file_path):
            return True
            
        for attempt in range(max_attempts):
            try:
                os.remove(file_path)
                logger.info(f"Successfully removed temporary file: {file_path}")
                return True
            except PermissionError as e:
                logger.warning(f"Attempt {attempt+1}/{max_attempts} to remove file failed: {str(e)}")
                time.sleep(delay)  # Wait before retrying
            except Exception as e:
                logger.error(f"Unexpected error removing file: {str(e)}")
                return False
                
        logger.error(f"Failed to remove temporary file after {max_attempts} attempts: {file_path}")
        return False

    def detect_face_regions(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = self.face_detector(gray)
            
            if not faces:
                return None
                
            face = faces[0]
            regions = {
                'forehead': ((face.left(), face.top()), (face.right(), face.top() + (face.bottom() - face.top()) // 3)),
                'cheeks': ((face.left(), face.top() + (face.bottom() - face.top()) // 3),
                          (face.right(), face.bottom() - (face.bottom() - face.top()) // 3)),
                'chin': ((face.left(), face.bottom() - (face.bottom() - face.top()) // 3),
                        (face.right(), face.bottom()))
            }
            return regions
        except Exception as e:
            logger.error(f"Error detecting face regions: {str(e)}")
            return None

    def _get_deepseek_analysis(self, image_path: str, progress_bar=None, status_text=None) -> Dict[str, Any]:
        """
        Send image to DeepSeek VL2 using Replicate API and get analysis.
        """
        try:
            logger.info("Sending request to DeepSeek VL2 via Replicate API...")
            
            # Check if REPLICATE_API_TOKEN is set
            if not self.replicate_api_token:
                error_msg = "REPLICATE_API_TOKEN not set in environment variables"
                logger.error(error_msg)
                return {"error": error_msg}
            
            # Prepare input for Replicate API
            input_data = {
                "image": open(image_path, "rb"),
                "prompt": "請詳細分析這張照片中的面部特徵，包括：皮膚紋理、色素分布、毛孔狀況、細紋分布、面部輪廓特徵等。並針對觀察到的狀況提供專業建議。"
            }
            
            # Log that we're sending the request (without the actual image data)
            logger.info("Sending request to Replicate API with prompt for facial analysis")
            
            # Create a progress indicator for the console
            progress_steps = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
            
            # Update UI progress
            self._update_progress(progress_bar, status_text, 60, "Waiting for DeepSeek analysis...")
            
            # Show animation in console logs
            for i in range(3):  # Show animation for a few cycles
                for step in progress_steps:
                    logger.info(f"Waiting for DeepSeek analysis {step}")
                    time.sleep(0.1)
            
            # Update progress before API call
            self._update_progress(progress_bar, status_text, 70, "Processing with DeepSeek VL2...")
            
            # Run the model using Replicate API
            output = replicate.run(
                "deepseek-ai/deepseek-vl2:e5caf557dd9e5dcee46442e1315291ef1867f027991ede8ff95e304d4f734200",
                input=input_data
            )
            
            # Update progress after API call
            self._update_progress(progress_bar, status_text, 85, "Received results from DeepSeek VL2...")
            
            logger.info("Successfully received DeepSeek VL2 response via Replicate")
            return output
            
        except Exception as e:
            error_msg = f"Replicate API error: {str(e)}"
            logger.error(error_msg)
            
            return {"error": error_msg}

    def _get_xai_analysis(self, image_base64: str, progress_bar=None, status_text=None) -> Dict[str, Any]:
        """
        Send image to XAI API and get analysis.
        """
        try:
            logger.info("Sending request to XAI API...")
            
            # 構建請求頭部
            headers = {
                "Authorization": f"Bearer {self.xai_api_key}",
                "Content-Type": "application/json"
            }
            
            # 構建請求內容
            data = {
                "model": "gpt-4-vision-preview" if self.xai_api_key.startswith("sk-proj-") else "grok-2-vision-1212",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "請詳細分析這張照片中的面部特徵，包括：皮膚紋理、色素分布、毛孔狀況、細紋分布、面部輪廓特徵等。並針對觀察到的狀況提供專業建議。"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1000
            }
            
            # 根據 API 金鑰格式選擇不同的處理方式
            if self.xai_api_key.startswith("sk-proj-"):
                try:
                    # 使用 OpenAI 客戶端
                    response = self.xai_client.chat.completions.create(**data)
                    result = response.choices[0].message.content
                    return result
                except Exception as e:
                    logger.error(f"OpenAI API error: {str(e)}")
                    raise e
            else:
                # 使用 X AI API
                response = requests.post(
                    f"{self.xai_base_url}/chat/completions",
                    headers=headers,
                    json=data
                )
                
                if response.status_code != 200:
                    error_msg = f"XAI API error: Status code {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return {"error": error_msg}
                
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                if not content:
                    return {"error": "No content in response"}
                    
                return content
            
        except Exception as e:
            error_msg = f"API error: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
