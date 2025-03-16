import streamlit as st
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)

class UIComponents:
    @staticmethod
    def setup_page():
        st.set_page_config(
            page_title="醫美診所智能評估系統",
            page_icon="💉",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        UIComponents._load_custom_css()

    @staticmethod
    def create_step_indicator(current_step: int):
        steps = ["上傳照片", "AI 分析", "生成報告", "查看結果"]
        
        for i, step in enumerate(steps, 1):
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(
                    f'''
                    <div class="step-circle" style="background: {UIComponents._get_step_color(i, current_step)}">
                        <span style="color: {UIComponents._get_step_text_color(i, current_step)}">{i}</span>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f'<p style="color: {UIComponents._get_step_text_color(i, current_step)}">{step}</p>',
                    unsafe_allow_html=True
                )

    @staticmethod
    def create_upload_section(on_upload: Callable[[Any], None]):
        st.header("上傳照片")
        uploaded_file = st.file_uploader(
            "請上傳正面照片",
            type=["jpg", "jpeg", "png"],
            help="支援 JPG、JPEG 和 PNG 格式的圖片"
        )
        
        if uploaded_file is not None:
            try:
                on_upload(uploaded_file)
            except Exception as e:
                logger.error(f"Error processing uploaded file: {str(e)}")
                st.error("處理照片時發生錯誤，請重試")

    @staticmethod
    def create_analysis_section(analysis_result: dict):
        st.header("AI 分析結果")
        
        # 獲取分析結果
        analysis = analysis_result.get('analysis', {})
        model = analysis.get('model', 'Unknown')
        result = analysis.get('result', '等待分析...')
        
        # 顯示模型標籤
        st.markdown(f"**使用模型:** {model}")
        
        # 顯示分析結果
        st.markdown("### 面部分析")
        st.markdown(result)

    @staticmethod
    def create_report_section(report_buffer):
        st.header("評估報告")
        
        if report_buffer:
            st.download_button(
                label="下載 PDF 報告",
                data=report_buffer,
                file_name="醫美診所智能評估報告.pdf",
                mime="application/pdf"
            )

    @staticmethod
    def _get_step_color(step: int, current_step: int) -> str:
        if step < current_step:
            return "var(--success-color)"
        elif step == current_step:
            return "var(--primary-color)"
        return "var(--neutral-light)"

    @staticmethod
    def _get_step_text_color(step: int, current_step: int) -> str:
        if step <= current_step:
            return "var(--neutral-dark)"
        return "var(--neutral-light)"

    @staticmethod
    def _load_custom_css():
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
    h1, h2, h3 { 
        color: var(--neutral-dark);
        margin-bottom: 20px;
        font-weight: 600;
    }
    
    /* 卡片樣式 */
    .card { 
        background: white; 
        padding: 25px; 
        border-radius: 16px; 
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        margin-bottom: 25px; 
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
    }
    
    /* 按鈕樣式 */
    div.stButton > button { 
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)); 
        color: white; 
        border-radius: 12px; 
        padding: 12px 24px; 
        font-weight: 500; 
        width: 100%;
        margin: 10px 0;
    }
    
    /* 步驟指示器 */
    .step-circle {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 10px;
        transition: all 0.3s ease;
    }
    
    /* 上傳區域 */
    .stFileUploader { 
        border: 2px dashed var(--primary-color); 
        border-radius: 16px; 
        padding: 20px; 
        background-color: var(--primary-light);
    }
    
    /* 標籤頁樣式 */
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)
