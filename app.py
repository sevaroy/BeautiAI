import os
import tempfile
import datetime
import logging
import re
import base64
import time
from typing import Optional, Tuple
from PIL import Image as PILImage
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import cv2
from dotenv import load_dotenv
import streamlit as st
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import plotly.express as px
from openai import OpenAI
import concurrent.futures
import sqlite3
import json
import dlib
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
from fpdf import FPDF
import shutil

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 環境變量加載
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

if not DEEPSEEK_API_KEY or not XAI_API_KEY:
    st.error("環境變量 DEEPSEEK_API_KEY 或 XAI_API_KEY 未設置，請檢查 .env 文件")
    logger.error("API 密鑰缺失")
    logger.info(f"DEEPSEEK_API_KEY: {DEEPSEEK_API_KEY}")
    logger.info(f"XAI_API_KEY: {XAI_API_KEY}")
    st.stop()

# 初始化 OpenAI 客戶端（用於 DeepSeek R1）
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# 初始化 xAI 客戶端（用於 Grok-2-Vision-1212）
xai_client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")

# Streamlit 頁面配置
st.set_page_config(
    page_title="醫美診所智能評估系統",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS 主題
st.markdown("""
<style>
    /* 主色調 */
    :root {
        --primary-color: #9C89B8;       /* 淺紫主色調 */
        --primary-light: #F0E6FF;       /* 淺紫背景色 */
        --secondary-color: #F0A6CA;     /* 柔粉輔助色 */
        --accent-color: #B8BEDD;        /* 淺藍點綴色 */
        --neutral-dark: #5E6472;        /* 高級灰 */
        --neutral-light: #F7F7FC;       /* 背景色 */
        --success-color: #A0C4B9;       /* 成功提示色 */
        --error-color: #E08F8F;         /* 錯誤提示色 */
    }
    
    /* 全局樣式 */
    body { 
        background-color: var(--neutral-light); 
        font-family: 'Arial', 'Microsoft YaHei', sans-serif;
        color: var(--neutral-dark);
    }
    
    .stApp { 
        background-color: var(--neutral-light);
    }
    
    /* 標題樣式 */
    h1 { 
        color: var(--neutral-dark); 
        font-size: 32px; 
        font-weight: 600;
        margin-bottom: 20px;
        text-align: center;
    }
    
    h2 { 
        color: var(--neutral-dark); 
        font-size: 24px; 
        margin-bottom: 15px;
        font-weight: 500;
    }
    
    h3 {
        color: var(--neutral-dark);
        font-size: 20px;
        margin-bottom: 12px;
        font-weight: 500;
    }
    
    /* 卡片樣式優化 */
    .card { 
        background: white; 
        padding: 25px; 
        border-radius: 16px; 
        box-shadow: 0 10px 30px rgba(0,0,0,0.05), 0 1px 8px rgba(0,0,0,0.02); 
        margin-bottom: 25px; 
        border: none;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
        height: 100%;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.08), 0 5px 15px rgba(0,0,0,0.04);
    }
    
    /* 卡片頂部漸變裝飾 */
    .card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color), var(--accent-color));
        border-radius: 5px 5px 0 0;
    }
    
    /* 按鈕樣式 */
    div.stButton > button { 
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)); 
        color: white; 
        border-radius: 12px; 
        padding: 12px 24px; 
        font-weight: 500; 
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(156, 137, 184, 0.3);
        width: 100%;
        margin: 10px 0;
        font-size: 16px;
    }
    
    div.stButton > button:hover { 
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(156, 137, 184, 0.4);
        background: linear-gradient(135deg, var(--secondary-color), var(--primary-color));
    }
    
    /* 上傳區域美化 */
    .stFileUploader { 
        border: 2px dashed var(--primary-color); 
        border-radius: 16px; 
        padding: 20px; 
        background-color: var(--primary-light);
        transition: all 0.3s ease;
        margin-bottom: 20px;
    }
    
    .stFileUploader:hover {
        border-color: var(--secondary-color);
        background-color: rgba(240, 230, 255, 0.7);
    }
    
    /* 統一圓角設計 */
    .stImage, .stFileUploader, div.stButton > button,
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stDateInput > div > div > input {
        border-radius: 12px !important;
    }
    
    /* 進度條美化 */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color)) !important;
        border-radius: 10px !important;
    }
    
    .stProgress > div {
        border-radius: 10px !important;
        background-color: var(--primary-light) !important;
    }
    
    /* 圖像容器美化 */
    .stImage {
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        transition: transform 0.3s ease;
        border-radius: 12px;
        margin-bottom: 15px;
    }
    
    .stImage:hover {
        transform: scale(1.02);
    }
    
    /* 頁面標題區域 */
    .title-container {
        text-align: center;
        padding: 15px 0 25px 0;
        margin-bottom: 20px;
        position: relative;
        background: linear-gradient(180deg, rgba(156, 137, 184, 0.1) 0%, rgba(255, 255, 255, 0) 100%);
        border-radius: 16px;
    }
    
    .title-container::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        border-radius: 2px;
    }
    
    /* 步驟指示器美化 */
    .step-indicator {
        display: flex;
        justify-content: space-between;
        margin: 0 auto 30px;
        padding: 20px;
        max-width: 800px;
        background: white;
        border-radius: 16px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    
    .step {
        display: flex;
        flex-direction: column;
        align-items: center;
        flex: 1;
        position: relative;
    }
    
    .step-circle {
        width: 45px;
        height: 45px;
        border-radius: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
        color: white;
        font-weight: bold;
        margin-bottom: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        position: relative;
        z-index: 2;
        transition: all 0.3s ease;
    }
    
    .step-circle:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .step-title {
        text-align: center;
        font-size: 14px;
        font-weight: 500;
        color: var(--neutral-dark);
        transition: color 0.3s ease;
    }
    
    .step-line {
        flex: 1;
        height: 3px;
        margin-top: 22px;
        position: relative;
        z-index: 1;
        transition: background 0.3s ease;
    }
    
    /* 分析結果卡片美化 */
    .result-card {
        min-height: 600px;
        display: flex;
        flex-direction: column;
    }
    
    /* 視覺化圖表容器 */
    .chart-container {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }
    
    .chart-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }
    
    .chart-title {
        font-size: 16px;
        color: var(--neutral-dark);
        text-align: center;
        margin-bottom: 15px;
        font-weight: 500;
        position: relative;
        padding-bottom: 10px;
    }
    
    .chart-title::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 50px;
        height: 2px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        border-radius: 1px;
    }
    
    /* 報告卡片美化 */
    .report-option {
        background: white;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
    }
    
    .report-option::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    }
    
    .report-option:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    }
    
    .report-option h4 {
        color: var(--neutral-dark);
        margin-bottom: 15px;
        font-size: 18px;
    }
    
    /* 側邊欄優化 */
    .sidebar-card {
        margin-bottom: 20px;
        padding: 20px;
        border-radius: 12px;
        background: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    
    .footer {
        position: absolute;
        bottom: 20px;
        left: 0;
        right: 0;
        text-align: center;
        font-size: 12px;
        color: var(--neutral-dark);
        opacity: 0.7;
        padding: 10px;
    }
    
    /* 分析步驟狀態顯示優化 */
    .analysis-status {
        padding: 15px;
        border-radius: 10px;
        background-color: var(--primary-light);
        margin-bottom: 12px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.03);
        border-left: 4px solid var(--primary-color);
    }
    
    /* 提示信息美化 */
    .info-box {
        background-color: var(--primary-light);
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 4px solid var(--accent-color);
    }
    
    /* Tab 樣式優化 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: var(--primary-light);
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 8px;
        background-color: white;
        color: var(--neutral-dark);
        border: none;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--accent-color);
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
        color: white !important;
    }
    
    /* 响应式布局调整 */
    @media screen and (max-width: 768px) {
        .card {
            padding: 15px;
        }
        
        h1 {
            font-size: 26px;
        }
        
        h2 {
            font-size: 20px;
        }
        
        .step-circle {
            width: 36px;
            height: 36px;
            font-size: 14px;
        }
        
        .step-title {
            font-size: 12px;
        }
        
        .chart-container {
            padding: 15px;
        }
        
        .report-option {
            padding: 20px;
        }
        
        .step-indicator {
            padding: 15px;
            margin-bottom: 20px;
        }
    }
    
    @media screen and (max-width: 480px) {
        .step-circle {
            width: 32px;
            height: 32px;
            font-size: 12px;
        }
        
        .step-title {
            font-size: 10px;
        }
        
        .chart-title {
            font-size: 14px;
        }
    }
</style>
""", unsafe_allow_html=True)

# 設置中文字體
try:
    # 使用更通用的字体设置，不依赖于SimHei
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 添加警告信息，告知用户中文可能无法正确显示
    logger.info("已设置通用字体，中文字符可能无法正确显示")
except Exception as e:
    logger.warning(f"设置字體失敗: {str(e)}，將使用默認字體")

# 支持多語言版本
TRANSLATIONS = {
    "zh": {
        "skin_condition": "皮膚狀況",
        "wrinkles": "皺紋",
        "spots": "色斑",
        # 其他翻譯...
    },
    "en": {
        "skin_condition": "Skin Condition",
        "wrinkles": "Wrinkles",
        "spots": "Spots",
        # 其他翻譯...
    }
}

def get_text(key, lang="zh"):
    return TRANSLATIONS[lang][key]

# 工具函數
def encode_image_to_base64(image_file: io.BytesIO) -> str:
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

@st.cache_data(ttl=3600)
def analyze_image(image_file: io.BytesIO) -> dict:
    try:
        logger.info("調用 Grok-2-Vision-1212 進行圖片分析")
        base64_image = encode_image_to_base64(image_file)
        
        # 創建進度條佔位符
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)
        status_text = st.empty()
        
        # 更新進度條 - 10%
        progress_bar.progress(10)
        status_text.text("正在處理圖片...")
        time.sleep(0.5)  # 添加短暫延遲以顯示進度
        
        # Grok API 調用
        try:
            # 更新進度條 - 20%
            progress_bar.progress(20)
            status_text.text("正在分析面部特徵...")
            time.sleep(0.5)
            
            grok_response = xai_client.chat.completions.create(
                model="grok-2-vision-1212",
                messages=[
                    {
                        "role": "system",
                        "content": "你是專業醫美顧問，請對此面部照片進行詳細分析，提供結構化報告。"
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """
                                    請對此面部照片進行詳細分析，提供結構化報告。針對以下區域：額頭、眼周、鼻子、頰骨、嘴唇、下巴，評估：
                                    1. 皮膚狀況（乾燥、油性、痤瘡等）
                                    2. 皺紋（深度、分布）
                                    3. 色斑（類型、範圍）
                                    4. 緊致度（鬆弛程度）
                                    5. 其他特徵（毛孔、黑眼圈等）
                                    對每個維度給出 0-5 分評分（0 表示嚴重問題，5 表示完美），並附上簡短描述。
                                """
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )
            
            # 更新進度條 - 50%
            progress_bar.progress(50)
            status_text.text("Grok 分析完成，正在處理結果...")
            time.sleep(0.5)
            
            # 儲存 Grok 回應
            grok_filename = save_api_response("grok", grok_response.dict())
            grok_analysis = grok_response.choices[0].message.content
            logger.info(f"Grok分析成功，內容長度：{len(grok_analysis)}")
        except Exception as e:
            logger.error(f"Grok API調用失敗: {str(e)}")
            grok_filename = None
            grok_analysis = "Grok API調用失敗，無法提供分析。"
            
            # 更新進度條 - 顯示錯誤但繼續
            progress_bar.progress(50)
            status_text.text("Grok 分析失敗，嘗試使用 DeepSeek...")
            time.sleep(0.5)
        
        # DeepSeek V3 API 調用
        try:
            # 更新進度條 - 60%
            progress_bar.progress(60)
            status_text.text("正在進行深度皮膚分析...")
            time.sleep(0.5)
            
            deepseek_response = deepseek_client.chat.completions.create(
                model="deepseek-vision-v3",  # 更新為 V3 版本
                messages=[
                    {
                        "role": "system",
                        "content": "你是專業醫美顧問，請對此面部照片進行詳細分析，提供結構化報告。"
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """
                                    請提供面部分析報告，包含：
                                    1. 整體膚質評估
                                    2. 問題區域識別
                                    3. 改善建議
                                    4. 護理重點
                                    請提供結構化的回應，並為每個方面提供具體的評分和建議。
                                """
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )
            
            # 更新進度條 - 90%
            progress_bar.progress(90)
            status_text.text("DeepSeek 分析完成，正在整合結果...")
            time.sleep(0.5)
            
            # 儲存 DeepSeek 回應
            deepseek_filename = save_api_response("deepseek", deepseek_response.dict())
            deepseek_analysis = deepseek_response.choices[0].message.content
            logger.info(f"DeepSeek分析成功，內容長度：{len(deepseek_analysis)}")
        except Exception as e:
            logger.error(f"DeepSeek API調用失敗: {str(e)}")
            deepseek_filename = None
            deepseek_analysis = "DeepSeek API調用失敗，無法提供分析。"
            
            # 更新進度條 - 顯示錯誤但繼續
            progress_bar.progress(90)
            status_text.text("DeepSeek 分析失敗，正在整合可用結果...")
            time.sleep(0.5)
        
        # 合併兩個 API 的分析結果
        combined_analysis = {
            "grok_analysis": grok_analysis,
            "deepseek_analysis": deepseek_analysis,
            "grok_file": grok_filename,
            "deepseek_file": deepseek_filename,
            "status": "success"
        }
        
        # 更新進度條 - 100%
        progress_bar.progress(100)
        status_text.text("分析完成！")
        time.sleep(1)  # 顯示完成訊息
        
        # 清除進度條和狀態文字
        progress_placeholder.empty()
        status_text.empty()
        
        return combined_analysis
        
    except Exception as e:
        logger.error(f"圖片分析失敗: {str(e)}")
        
        # 如果進度條已經創建，顯示錯誤
        if 'progress_bar' in locals():
            progress_bar.progress(100)
            status_text.text("分析過程中發生錯誤！")
            time.sleep(1)
            progress_placeholder.empty()
            status_text.empty()
            
        return {
            "status": "error",
            "error": str(e),
            "grok_analysis": "分析失敗",
            "deepseek_analysis": "分析失敗"
        }

@st.cache_data
def generate_report(analysis_result: str) -> str:
    try:
        logger.info("調用 DeepSeek R1 生成報告")
        
        # 確保分析結果不為空
        if not analysis_result or analysis_result == "分析失敗":
            logger.error("無法生成報告：分析結果為空")
            return "無法生成報告：分析結果為空。請重新上傳照片進行分析。"
            
        try:
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": """
                        你是資深醫美專家，請根據以下面部分析結果生成一份專業、詳盡的醫美建議報告，字數至少 500 字。報告應包含以下內容，並確保語言邏輯清晰、結構分明，符合醫美行業標準：
                        1. 面部狀況綜合評估：
                           - 針對額頭、眼周、鼻子、頰骨、嘴唇、下巴，總結各區域的皮膚狀況、皺紋、色斑、緊致度等。
                           - 分析整體面部健康狀態，提供專業診斷，結合數據進行深入推理。
                        2. 推薦的醫美治療方案：
                           - 提供至少 5 種具體治療方案，按優先級排序。
                           - 每項包括治療名稱、適用區域、實施方式（如注射劑量、療程次數）。
                        3. 預期效果：
                           - 詳細描述每種方案的預期效果（如皺紋減少百分比、緊致度提升程度），使用量化數據並進行邏輯推導。
                        4. 術後護理建議：
                           - 針對每種方案提供具體護理措施（如保濕、防曬頻率、飲食建議），考慮長期效果。
                        5. 風險提示：
                           - 列出每種方案的潛在風險（如紅腫、過敏）及緩解方法，分析風險可能性。
                        使用專業術語（如「皮下注射」、「色素分解」、「組織提拉」），確保報告詳實且具權威性，展示深入的醫學推理能力。
                    """},
                    {"role": "user", "content": f"""
                        請根據以下面部分析結果生成報告：
                        {analysis_result}
                    """}
                ],
                temperature=0.3,
                max_tokens=2000,
                stream=False
            )
            report = response.choices[0].message.content
            logger.info(f"DeepSeek R1 報告結果生成成功，字數: {len(report)}")
            
            if len(report) < 500:
                logger.warning("報告字數不足 500 字，但仍將使用該報告")
            
            return report + "\n\n**免責聲明**：本報告由 DeepSeek R1 AI 生成，僅供參考，具體治療需諮詢專業醫生。"
        except Exception as e:
            logger.error(f"DeepSeek R1 報告API調用失敗: {str(e)}")
            
            # 根據分析內容直接生成簡單報告（備選方案）
            fallback_report = f"""
            # 面部分析簡易報告

            ## 分析結果
            {analysis_result[:1000]}...

            ## 基本建議
            1. 建議進行專業的皮膚護理療程
            2. 根據分析結果，可考慮光療或其他適合的醫美項目
            3. 日常應加強保濕和防曬
            4. 選擇適合的護膚品，避免刺激成分
            5. 定期複診，追蹤皮膚狀態改善情況

            **注意**：此為系統自動生成的簡易報告，由於API調用失敗，無法提供詳細專業建議。建議諮詢專業醫生獲取更準確的評估和治療方案。
            """
            return fallback_report
            
    except Exception as e:
        logger.error(f"DeepSeek R1 報告生成流程整體失敗: {str(e)}")
        return f"錯誤: 無法生成報告 ({str(e)})\n\n請稍後再試或聯繫技術支持。"

@st.cache_data
def create_visualizations(_image: PILImage.Image, analysis_result: str, report: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """创建可视化图表"""
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    heatmap_path = os.path.join(temp_dir, "face_heatmap.png")
    radar_path = os.path.join(temp_dir, "radar_chart.png")
    priority_path = os.path.join(temp_dir, "treatment_priority.png")

    # 热力图
    try:
        img_array = np.array(_image)
        mask = np.zeros_like(img_array[:, :, 0], dtype=float)
        h, w = mask.shape
        regions = detect_face_regions(_image)
        
        for region, (y1, y2, x1, x2) in regions.items():
            # 确保坐标在图像范围内
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)
            
            # 查找评分
            score_match = re.search(rf"{region}.*?皮膚狀況\s*(\d)/5", analysis_result)
            if score_match:
                score = int(score_match.group(1))
                severity = (5 - score) / 5
                mask[y1:y2, x1:x2] = severity
            else:
                # 默认值
                mask[y1:y2, x1:x2] = 0.5
                
        # 应用高斯模糊使热力图更平滑
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(img_array)
        plt.imshow(mask, cmap='RdYlGn_r', alpha=0.5)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"熱力圖生成成功: {heatmap_path}")
    except Exception as e:
        logger.error(f"熱力圖生成失敗: {str(e)}", exc_info=True)
        heatmap_path = None

    # 雷達圖
    try:
        categories = ['Skin Quality', 'Elasticity', 'Firmness', 'Radiance', 'Evenness']
        current_scores = []
        ideal_scores = [5] * len(categories)
        for category in categories:
            match = re.search(rf"{category}.*?(\d)/5", analysis_result, re.IGNORECASE)
            score = int(match.group(1)) if match else 4
            current_scores.append(score)
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        current_scores += current_scores[:1]
        ideal_scores += ideal_scores[:1]
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, current_scores, color='#4A90E2', alpha=0.5, label='Rating', edgecolor='black')
        ax.fill(angles, ideal_scores, color='#D3E4F5', alpha=0.2, label='Ideal')
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories[:-1])
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.savefig(radar_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"雷達圖生成成功: {radar_path}")
    except Exception as e:
        logger.error(f"雷達圖生成失敗: {str(e)}")
        radar_path = None

    # 優先級圖
    try:
        treatments = []
        priorities = []
        for line in report.split('\n'):
            match = re.search(r'(\d+)\)\s*([^0-5].*?)(?=\s*\d|\n|$)', line)
            if match:
                priority = int(match.group(1))
                treatment = match.group(2).strip()
                treatments.append(treatment)
                priorities.append(6 - priority)
        if not treatments:
            treatments = ["玻尿酸填充", "肉毒素注射", "激光治療"]
            priorities = [5, 4, 3]
        fig = px.bar(
            x=priorities, y=treatments, orientation='h',
            labels={'x': 'Priority', 'y': 'Treatment'},
            title="Treatment Priority",
            color=priorities, color_continuous_scale='Blues',
            text=priorities
        )
        fig.update_traces(textposition='auto')
        fig.update_layout(showlegend=False, width=600, height=400)
        fig.write_image(priority_path, scale=2)
        logger.info(f"優先級圖生成成功: {priority_path}")
    except Exception as e:
        logger.error(f"優先級圖生成失敗: {str(e)}")
        priority_path = None

    return heatmap_path, radar_path, priority_path

def detect_face_regions(image):
    """检测人脸区域，返回额头、脸颊和下巴区域"""
    try:
        # 转换到灰度图
        import cv2
        import dlib
        
        # 加载dlib人脸检测器
        face_detector = dlib.get_frontal_face_detector()
        
        # 转换图像为numpy数组并转为灰度
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 检测人脸
        faces = face_detector(gray)
        
        if not faces:
            # 如果没检测到人脸，使用整个图像
            h, w = img_array.shape[:2]
            regions = {
                'forehead': (0, h//3, 0, w),
                'cheeks': (h//3, 2*h//3, 0, w),
                'chin': (2*h//3, h, 0, w)
            }
            return regions
            
        # 使用第一个检测到的人脸
        face = faces[0]
        face_height = face.bottom() - face.top()
        
        # 定义面部区域
        regions = {
            'forehead': (face.top(), face.top() + face_height // 3, face.left(), face.right()),
            'cheeks': (face.top() + face_height // 3, face.bottom() - face_height // 3, face.left(), face.right()),
            'chin': (face.bottom() - face_height // 3, face.bottom(), face.left(), face.right())
        }
        return regions
    except Exception as e:
        logger.error(f"检测人脸区域失败: {str(e)}")
        # 返回基本分区
        h, w = np.array(image).shape[:2]
        return {
            'forehead': (0, h//3, 0, w),
            'cheeks': (h//3, 2*h//3, 0, w),
            'chin': (2*h//3, h, 0, w)
        }

def generate_better_pdf(report_text, images):
    """生成PDF报告，确保支持中文"""
    try:
        # 注册中文字体
        font_path = os.path.join(os.path.dirname(__file__), 'fonts')
        os.makedirs(font_path, exist_ok=True)
        
        # 下载中文字体
        font_file = os.path.join(font_path, 'simsun.ttf')
        
        # 检查字体文件是否存在
        if not os.path.exists(font_file):
            # 使用临时内置字体
            logger.info("使用内置中文字体")
            temp_font_path = os.path.join(tempfile.gettempdir(), "simsun.ttf")
            
            # 如果您有办法将宋体嵌入应用中，可以尝试以下方式
            try:
                from fontTools.ttLib import TTFont as FontToolsTTFont
                # 创建一个简单的字体
                font = FontToolsTTFont()
                font.save(temp_font_path)
                font_file = temp_font_path
            except:
                # 如果无法创建字体，使用reportlab提供的基本字体
                logger.warning("无法创建中文字体，将使用基本字体")
        
        # 使用 session_state 來追踪字體註冊狀態
        if 'app_fonts_registered' not in st.session_state:
            st.session_state.app_fonts_registered = False
            
        # 注册字体，但只在第一次註冊
        if not st.session_state.app_fonts_registered:
            try:
                pdfmetrics.registerFont(TTFont('SimSun', font_file))
                logger.info("成功注册中文字体")
                st.session_state.app_fonts_registered = True
            except Exception as e:
                logger.error(f"注册字体失败: {str(e)}")
        else:
            logger.debug("字體已註冊，跳過註冊過程")
        
        # 创建一个内存中的PDF，而不是直接写入文件
        buffer = BytesIO()
        
        # 创建PDF文档
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        
        # 自定义样式以使用中文字体
        for style_name in styles.byName:
            styles[style_name].fontName = 'SimSun'
        
        story = []
        
        # 添加标题
        title_style = styles['Heading1']
        title_style.alignment = 1  # 居中对齐
        story.append(Paragraph("醫美智能評估報告", title_style))
        story.append(Spacer(1, 12))
        
        # 添加日期
        date_style = styles['Normal']
        date_style.alignment = 1  # 居中对齐
        current_date = datetime.datetime.now().strftime("%Y年%m月%d日")
        story.append(Paragraph(f"生成日期：{current_date}", date_style))
        story.append(Spacer(1, 20))
        
        # 处理报告内容
        normal_style = styles['Normal']
        normal_style.leading = 14  # 行间距
        
        # 确保报告文本不为空
        if not report_text or len(report_text.strip()) == 0:
            report_text = "無法生成報告內容，請重試。"
        
        # 分段处理报告文本
        paragraphs = report_text.split('\n\n')
        for para in paragraphs:
            if para.strip():
                # 检查是否为标题行
                if re.match(r'^[0-9]+\.\s+\w+', para.strip()):
                    heading_style = styles['Heading2']
                    story.append(Paragraph(para, heading_style))
                else:
                    # 处理普通段落，保留换行
                    lines = para.split('\n')
                    for line in lines:
                        if line.strip():
                            story.append(Paragraph(line, normal_style))
                story.append(Spacer(1, 10))
        
        # 添加图片 - 先验证图片是否可用
        valid_images = []
        for img_path in images:
            if img_path and os.path.exists(img_path):
                try:
                    # 测试是否可以打开图片
                    PILImage.open(img_path)
                    valid_images.append(img_path)
                except Exception as e:
                    logger.error(f"无法打开图片 {img_path}: {str(e)}")
            else:
                logger.warning(f"图片路径不存在: {img_path}")
        
        if valid_images:  # 只有当有有效图片时才添加图表标题
            story.append(Spacer(1, 20))
            story.append(Paragraph("分析圖表", styles['Heading2']))
            story.append(Spacer(1, 10))
            
            # 图片处理
            captions = ["面部問題熱力圖", "面部狀況評分", "治療方案優先級"]
            for i, img_path in enumerate(valid_images):
                try:
                    # 添加图片标题
                    if i < len(captions):
                        story.append(Paragraph(captions[i], styles['Heading3']))
                    
                    # 打开并处理图片
                    img = PILImage.open(img_path)
                    img_width, img_height = img.size
                    
                    # 计算适合A4页面的图片尺寸
                    max_width = 450
                    aspect = img_height / img_width
                    new_width = min(max_width, img_width)
                    new_height = new_width * aspect
                    
                    # 添加图片到PDF
                    img = ReportLabImage(img_path, width=new_width, height=new_height)
                    story.append(img)
                    story.append(Spacer(1, 15))
                except Exception as e:
                    logger.error(f"处理图片失败: {str(e)}", exc_info=True)
        
        # 添加免责声明
        story.append(Spacer(1, 30))
        disclaimer_style = styles['Italic']
        disclaimer_style.textColor = colors.gray
        story.append(Paragraph("免責聲明：本報告由AI系統生成，僅供參考，具體治療方案請諮詢專業醫生。", disclaimer_style))
        
        # 构建PDF
        try:
            doc.build(story)
            buffer.seek(0)
            
            # 保存到临时文件
            temp_pdf = os.path.join(tempfile.gettempdir(), "medical_report.pdf")
            with open(temp_pdf, 'wb') as f:
                f.write(buffer.getvalue())
            
            logger.info(f"PDF生成成功: {temp_pdf}")
            return temp_pdf
        except Exception as e:
            logger.error(f"构建PDF失败: {str(e)}", exc_info=True)
            return None
    except Exception as e:
        logger.error(f"PDF生成失败: {str(e)}", exc_info=True)
        return None

def generate_simple_pdf(report_text, images):
    """使用更简单的方法生成PDF，确保支持中文"""
    try:
        # 创建PDF对象
        pdf = FPDF()
        pdf.add_page()
        
        # 使用Arial Unicode MS字体，这是一个通用的Unicode字体
        # 注意：FPDF默认不支持中文，我们需要使用特殊方法
        
        # 添加标题（使用英文避免字体问题）
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Medical Beauty Assessment Report', 0, 1, 'C')
        
        # 添加日期
        pdf.set_font('Arial', '', 12)
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        pdf.cell(0, 10, f'Date: {current_date}', 0, 1, 'C')
        
        # 添加中文报告内容的提示
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, "Due to font limitations in PDF, Chinese characters cannot be displayed properly.")
        pdf.multi_cell(0, 5, "Below is the analysis visualization. For full report, please download the text report.")
        
        # 添加图片（这部分应该正常工作）
        valid_images = []
        for img_path in images:
            if img_path and os.path.exists(img_path):
                try:
                    valid_images.append(img_path)
                except Exception as e:
                    logger.error(f"图片验证失败: {str(e)}")
        
        # 添加图片
        for img_path in valid_images:
            try:
                pdf.add_page()
                # 添加图片标题
                if img_path.endswith("face_heatmap.png"):
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 10, "Face Problem Heat Map", 0, 1, 'C')
                elif img_path.endswith("radar_chart.png"):
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 10, "Facial Condition Score", 0, 1, 'C')
                elif img_path.endswith("treatment_priority.png"):
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 10, "Treatment Priority", 0, 1, 'C')
                
                # 添加图片，确保适合页面
                pdf.image(img_path, x=10, y=30, w=190)
            except Exception as e:
                logger.error(f"添加图片失败: {str(e)}")
        
        # 添加免责声明
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 10, 'Disclaimer: This report is generated by AI for reference only.', 0, 1, 'C')
        
        # 保存PDF
        temp_pdf = os.path.join(tempfile.gettempdir(), "medical_report.pdf")
        pdf.output(temp_pdf)
        
        return temp_pdf
    except Exception as e:
        logger.error(f"简单PDF生成失败: {str(e)}", exc_info=True)
        return None

def generate_premium_report():
    """生成高級評估報告"""
    try:
        # 获取当前分析结果和报告
        if 'analysis_result' not in st.session_state or 'report' not in st.session_state:
            st.error("請先完成面部分析")
            return
            
        analysis_result = st.session_state.analysis_result
        report = st.session_state.report
        
        # 创建可视化图表
        if 'uploaded_image' in st.session_state and st.session_state.uploaded_image:
            try:
                image = PILImage.open(st.session_state.uploaded_image)
                heatmap_path, radar_path, priority_path = create_visualizations(image, analysis_result["grok_analysis"], report)
                images = [img for img in [heatmap_path, radar_path, priority_path] if img and os.path.exists(img)]
                logger.info(f"可視化圖表生成成功，有效圖片數量: {len(images)}")
            except Exception as e:
                logger.error(f"可視化圖表生成失敗: {str(e)}")
                images = []
                st.warning("無法生成視覺化圖表，報告將只包含文字內容")
        else:
            images = []
            st.warning("未找到上傳的圖片，報告將只包含文字內容")
            
        # 生成PDF报告
        try:
            pdf_path = generate_better_pdf(report, images)
            
            if pdf_path and os.path.exists(pdf_path):
                # 保存为高级报告
                premium_pdf_path = "premium_report.pdf"
                shutil.copy(pdf_path, premium_pdf_path)
                logger.info(f"高級報告生成成功: {premium_pdf_path}")
                st.success("高級報告生成成功！")
                return True
            else:
                st.error("PDF報告生成失敗，請重試")
                return False
        except Exception as e:
            logger.error(f"PDF生成過程中出錯: {str(e)}")
            st.error(f"PDF生成過程中出錯: {str(e)}")
            return False
    except Exception as e:
        logger.error(f"高級報告生成失敗: {str(e)}")
        st.error(f"報告生成失敗: {str(e)}")
        return False

def generate_standard_report():
    """生成標準評估報告"""
    try:
        # 获取当前分析结果和报告
        if 'analysis_result' not in st.session_state or 'report' not in st.session_state:
            st.error("請先完成面部分析")
            return False
            
        analysis_result = st.session_state.analysis_result
        report = st.session_state.report
        
        # 创建可视化图表
        if 'uploaded_image' in st.session_state and st.session_state.uploaded_image:
            try:
                image = PILImage.open(st.session_state.uploaded_image)
                heatmap_path, radar_path, priority_path = create_visualizations(image, analysis_result["grok_analysis"], report)
                images = [img for img in [heatmap_path, radar_path, priority_path] if img and os.path.exists(img)]
                logger.info(f"可視化圖表生成成功，有效圖片數量: {len(images)}")
            except Exception as e:
                logger.error(f"可視化圖表生成失敗: {str(e)}")
                images = []
                st.warning("無法生成視覺化圖表，報告將只包含文字內容")
        else:
            images = []
            st.warning("未找到上傳的圖片，報告將只包含文字內容")
            
        # 生成简单PDF报告
        try:
            pdf_path = generate_simple_pdf(report, images)
            
            if pdf_path and os.path.exists(pdf_path):
                # 保存为标准报告
                standard_pdf_path = "standard_report.pdf"
                shutil.copy(pdf_path, standard_pdf_path)
                logger.info(f"標準報告生成成功: {standard_pdf_path}")
                st.success("標準報告生成成功！")
                return True
            else:
                st.error("PDF報告生成失敗，請重試")
                return False
        except Exception as e:
            logger.error(f"PDF生成過程中出錯: {str(e)}")
            st.error(f"PDF生成過程中出錯: {str(e)}")
            return False
    except Exception as e:
        logger.error(f"標準報告生成失敗: {str(e)}")
        st.error(f"報告生成失敗: {str(e)}")
        return False

def save_api_response(response_data: dict, response_type: str):
    """保存 API 响应到数据库"""
    try:
        conn = sqlite3.connect('responses.db')
        cursor = conn.cursor()
        
        # 创建表（如果不存在）
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            response_type TEXT,
            response_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 保存响应数据
        cursor.execute(
            'INSERT INTO api_responses (response_type, response_data) VALUES (?, ?)',
            (response_type, json.dumps(response_data))
        )
        
        conn.commit()
        conn.close()
        logger.info(f"{response_type} API响应保存成功")
    except Exception as e:
        logger.error(f"保存API响应失败: {str(e)}")
        raise

def plot_radar_chart(analysis_data: dict) -> plt.Figure:
    """生成雷达图"""
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 提取数据
        categories = ['皮肤状况', '色斑', '皱纹', '毛孔', '油脂']
        scores = [
            analysis_data.get('skin_condition', 0),
            analysis_data.get('spots', 0),
            analysis_data.get('wrinkles', 0),
            analysis_data.get('pores', 0),
            analysis_data.get('oil', 0)
        ]
        
        # 确保分数在0-100之间
        scores = [max(0, min(100, score)) for score in scores]
        
        # 创建雷达图
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        scores = np.concatenate((scores, [scores[0]]))  # 闭合图形
        angles = np.concatenate((angles, [angles[0]]))  # 闭合图形
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, scores, 'o-', linewidth=2)
        ax.fill(angles, scores, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        
        plt.title('皮肤状况分析雷达图', pad=20)
        
        return fig
    except Exception as e:
        logger.error(f"生成雷达图失败: {str(e)}")
        raise

def plot_skin_analysis(image: np.ndarray, regions: dict) -> plt.Figure:
    """生成皮肤分析图，标注不同区域"""
    try:
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 显示原始图像
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 为不同区域添加标注
        colors = {
            'forehead': 'red',
            'left_cheek': 'green',
            'right_cheek': 'blue',
            'chin': 'yellow'
        }
        
        for region_name, coords in regions.items():
            if coords:
                x, y, w, h = coords
                rect = plt.Rectangle((x, y), w, h, fill=False, 
                                  color=colors.get(region_name, 'white'),
                                  linewidth=2)
                ax.add_patch(rect)
                ax.text(x, y-5, region_name, color=colors.get(region_name, 'white'),
                       fontsize=10, backgroundcolor='black')
        
        ax.axis('off')
        plt.title('皮肤区域分析')
        
        return fig
    except Exception as e:
        logger.error(f"生成皮肤分析图失败: {str(e)}")
        raise

def main():
    # 初始化 session state
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'report_generated' not in st.session_state:
        st.session_state.report_generated = False
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'report' not in st.session_state:
        st.session_state.report = None
    
    # 側邊欄設置
    with st.sidebar:
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.header("系統設置")
        st.write("分析模型：Grok-2-Vision-1212")
        st.write("報告模型：DeepSeek R1")
        
        # 添加重置按鈕
        if st.button("重置應用"):
            # 清除 session state
            for key in list(st.session_state.keys()):
                if key != "page_config":  # 保留任何頁面配置
                    del st.session_state[key]
            st.session_state.current_step = 1
            st.session_state.analysis_complete = False
            st.session_state.report_generated = False
            st.rerun()
            
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="footer">2025 醫美診所智能系統</div>', unsafe_allow_html=True)

    # 添加標題容器
    st.markdown('<div class="title-container">', unsafe_allow_html=True)
    st.title("醫美診所智能評估系統")
    st.markdown('</div>', unsafe_allow_html=True)

    # 步驟指示器
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            f"""
            <div class="step-indicator">
                <div class="step">
                    <div class="step-circle" style="background: {get_step_color(1)}">1</div>
                    <div class="step-title" style="color: {get_step_text_color(1)}">上傳照片</div>
                </div>
                <div class="step-line" style="background: {get_line_color(1, 2)}"></div>
                <div class="step">
                    <div class="step-circle" style="background: {get_step_color(2)}">2</div>
                    <div class="step-title" style="color: {get_step_text_color(2)}">智能分析</div>
                </div>
                <div class="step-line" style="background: {get_line_color(2, 3)}"></div>
                <div class="step">
                    <div class="step-circle" style="background: {get_step_color(3)}">3</div>
                    <div class="step-title" style="color: {get_step_text_color(3)}">生成報告</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 主要內容區域
    with st.container():
        # 步驟1：上傳照片
        if st.session_state.current_step == 1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📸 上傳您的照片")
            st.markdown('<div class="info-box">請上傳正面清晰的照片，確保光線充足且面部完整可見</div>', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("選擇照片", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file is not None:
                try:
                    # 保存上傳的圖片到 session state
                    temp_dir = "temp"
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_file = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_file, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    st.session_state.uploaded_image = temp_file
                    
                    # 顯示圖片預覽
                    image = PILImage.open(uploaded_file)
                    st.image(image, caption="預覽圖片", use_column_width=True)
                    
                    if st.button("開始分析"):
                        st.session_state.current_step = 2
                        st.rerun()
                except Exception as e:
                    st.error(f"圖片處理錯誤: {str(e)}")
                    logger.error(f"圖片處理錯誤: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)

        # 步驟2：智能分析
        elif st.session_state.current_step == 2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🔍 智能分析進行中")
            
            # 分析狀態顯示
            st.markdown(
                """
                <div class="analysis-status">
                    <p>✓ 圖像預處理完成</p>
                </div>
                <div class="analysis-status">
                    <p>✓ 面部特徵識別完成</p>
                </div>
                <div class="analysis-status">
                    <p>✓ 皮膚狀況分析完成</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            try:
                # 真實分析過程
                with st.spinner("分析中..."):
                    # 檢查是否有上傳的圖片
                    if st.session_state.uploaded_image and os.path.exists(st.session_state.uploaded_image):
                        # 讀取圖片
                        with open(st.session_state.uploaded_image, "rb") as f:
                            image_bytes = io.BytesIO(f.read())
                        
                        # 進行分析
                        progress_text = "正在生成分析報告..."
                        progress_bar = st.progress(0)
                        
                        # 分析圖片
                        analysis_result = analyze_image(image_bytes)
                        # 存儲分析結果
                        st.session_state.analysis_result = analysis_result
                        
                        # 更新進度
                        for i in range(50):
                            time.sleep(0.02)
                            progress_bar.progress(i + 1)
                        
                        # 生成報告
                        if analysis_result and "status" in analysis_result and analysis_result["status"] == "success":
                            # 組合分析結果文本
                            combined_text = f"""
                            # Grok-2-Vision-1212 分析結果:
                            {analysis_result['grok_analysis']}
                            
                            # DeepSeek-Vision-V3 分析結果:
                            {analysis_result['deepseek_analysis']}
                            """
                            
                            # 生成報告
                            report = generate_report(combined_text)
                            st.session_state.report = report
                            
                            # 更新進度
                            for i in range(50, 100):
                                time.sleep(0.02)
                                progress_bar.progress(i + 1)
                            
                            st.session_state.analysis_complete = True
                            st.session_state.current_step = 3
                            st.rerun()
                        else:
                            st.error("分析失敗，請重試")
                            logger.error(f"分析失敗: {analysis_result}")
                            # 提供返回按鈕
                            if st.button("返回"):
                                st.session_state.current_step = 1
                                st.rerun()
                    else:
                        st.error("未找到上傳的圖片，請返回上一步重新上傳")
                        if st.button("返回上傳"):
                            st.session_state.current_step = 1
                            st.rerun()
            except Exception as e:
                st.error(f"分析過程出錯: {str(e)}")
                logger.error(f"分析過程出錯: {str(e)}")
                # 提供返回按鈕
                if st.button("返回重試"):
                    st.session_state.current_step = 1
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

        # 步驟3：顯示結果和生成報告
        elif st.session_state.current_step == 3:
            # 檢查是否完成分析
            if not st.session_state.analysis_complete or not st.session_state.report:
                st.error("尚未完成分析，請返回第一步")
                if st.button("返回第一步"):
                    st.session_state.current_step = 1
                    st.rerun()
            else:
                # 分析結果標籤頁
                tab1, tab2 = st.tabs(["面部分析", "治療建議"])
                
                with tab1:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.subheader("🔍 面部分析結果")
                    
                    # 顯示原始分析結果 (可折疊)
                    with st.expander("查看原始分析數據"):
                        st.markdown("### Grok-2-Vision-1212 分析")
                        st.markdown(st.session_state.analysis_result["grok_analysis"])
                        st.markdown("### DeepSeek-Vision-V3 分析")
                        st.markdown(st.session_state.analysis_result["deepseek_analysis"])
                    
                    # 面部特徵分析
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.markdown('<div class="chart-title">面部特徵評分</div>', unsafe_allow_html=True)
                        try:
                            plot_radar_chart()
                        except Exception as e:
                            st.error(f"無法生成雷達圖: {str(e)}")
                            logger.error(f"無法生成雷達圖: {str(e)}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.markdown('<div class="chart-title">皮膚狀況分析</div>', unsafe_allow_html=True)
                        try:
                            plot_skin_analysis()
                        except Exception as e:
                            st.error(f"無法生成皮膚分析圖: {str(e)}")
                            logger.error(f"無法生成皮膚分析圖: {str(e)}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 熱力圖
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.markdown('<div class="chart-title">面部問題區域熱力圖</div>', unsafe_allow_html=True)
                    try:
                        plot_heatmap()
                    except Exception as e:
                        st.error(f"無法生成熱力圖: {str(e)}")
                        logger.error(f"無法生成熱力圖: {str(e)}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab2:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.subheader("💡 個性化治療建議")
                    
                    # 直接顯示報告內容
                    st.markdown(st.session_state.report)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # 報告生成區域
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("📊 評估報告")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(
                        """
                        <div class="report-option">
                            <h4>標準報告</h4>
                            <p>包含基礎分析結果和治療建議</p>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    if not st.session_state.report_generated:
                        if st.button("生成標準報告"):
                            try:
                                generate_standard_report()
                                st.session_state.report_generated = True
                                st.rerun()
                            except Exception as e:
                                st.error(f"標準報告生成失敗: {str(e)}")
                                logger.error(f"標準報告生成失敗: {str(e)}")
                    else:
                        pdf_path = "standard_report.pdf"
                        if os.path.exists(pdf_path):
                            with open(pdf_path, "rb") as f:
                                pdf_bytes = f.read()
                            st.download_button(
                                label="下載標準報告 ",
                                data=pdf_bytes,
                                file_name="醫美診所評估報告.pdf",
                                mime="application/pdf",
                            )
                        else:
                            st.error("報告文件未找到，請重新生成")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown(
                        """
                        <div class="report-option">
                            <h4>高級報告</h4>
                            <p>包含詳細分析和個性化治療方案</p>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    if not st.session_state.report_generated:
                        if st.button("生成高級報告"):
                            try:
                                generate_premium_report()
                                st.session_state.report_generated = True
                                st.rerun()
                            except Exception as e:
                                st.error(f"高級報告生成失敗: {str(e)}")
                                logger.error(f"高級報告生成失敗: {str(e)}")
                    else:
                        premium_pdf_path = "premium_report.pdf"
                        if os.path.exists(premium_pdf_path):
                            with open(premium_pdf_path, "rb") as f:
                                premium_pdf_bytes = f.read()
                            st.download_button(
                                label="下載高級報告 ",
                                data=premium_pdf_bytes,
                                file_name="醫美診所高級評估報告.pdf",
                                mime="application/pdf",
                            )
                        else:
                            st.error("高級報告文件未找到，請重新生成")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

# 步驟指示器樣式函數
def get_step_color(step_num):
    """獲取步驟圓圈的顏色"""
    current_step = st.session_state.current_step
    if step_num < current_step:
        return "var(--success-color)"  # 已完成步驟
    elif step_num == current_step:
        return "var(--primary-color)"  # 當前步驟
    else:
        return "#E0E0E0"  # 未完成步驟

def get_step_text_color(step_num):
    """獲取步驟文字的顏色"""
    current_step = st.session_state.current_step
    if step_num <= current_step:
        return "var(--neutral-dark)"  # 當前或已完成步驟
    else:
        return "#AAAAAA"  # 未完成步驟

def get_line_color(step1, step2):
    """獲取連接線的顏色"""
    current_step = st.session_state.current_step
    if step2 <= current_step:
        return "var(--success-color)"  # 已完成階段
    else:
        return "#E0E0E0"  # 未完成階段

# 執行主程序
if __name__ == "__main__":
    main()