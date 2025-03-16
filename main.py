import streamlit as st
import logging
from src.image_analyzer import ImageAnalyzer
from src.report_generator import ReportGenerator
from src.ui_components import UIComponents
from utils.helpers import validate_image, save_api_response
from config.settings import DEEPSEEK_API_KEY, XAI_API_KEY, REPLICATE_API_TOKEN, GROK_API_KEY

logger = logging.getLogger(__name__)

# 初始化 session_state 中的關鍵變量
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'image_processed' not in st.session_state:
    st.session_state.image_processed = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "DeepSeek VL2"  # 預設模型

class BeautyClinicApp:
    def __init__(self):
        self.image_analyzer = ImageAnalyzer()
        self.report_generator = ReportGenerator()
        self.analysis_result = None
        self.report_buffer = None

    def run(self):
        # Setup UI
        UIComponents.setup_page()
        
        # 創建側邊欄
        self.create_sidebar()
        
        # Create step indicator
        UIComponents.create_step_indicator(st.session_state.current_step)
        
        # Main workflow
        try:
            # Step 1: Upload photo
            if st.session_state.current_step == 1:
                self.create_upload_section()
            
            # Step 2: AI Analysis
            elif st.session_state.current_step == 2:
                if st.session_state.get('uploaded_image'):
                    # 顯示已上傳的圖片
                    image = validate_image(st.session_state.uploaded_image)
                    st.image(image, caption="上傳的照片", use_container_width=True)
                    
                    # 添加分析按鈕
                    if st.button("開始分析"):
                        with st.spinner("正在進行 AI 分析..."):
                            logger.info(f"開始使用 {st.session_state.selected_model} 進行 AI 分析...")
                            self.analysis_result = self.image_analyzer.analyze_image(
                                st.session_state.uploaded_image,
                                model=st.session_state.selected_model
                            )
                            
                            # Check if there was an error in the analysis
                            if "error" in self.analysis_result:
                                st.error(f"分析過程中發生錯誤: {self.analysis_result['error']}")
                                logger.error(f"分析錯誤: {self.analysis_result['error']}")
                                # Still store the result to allow viewing error details
                                st.session_state.analysis_result = self.analysis_result
                                st.session_state.analysis_complete = False
                            else:
                                st.session_state.analysis_result = self.analysis_result
                                st.session_state.analysis_complete = True
                                st.session_state.current_step = 3
                                logger.info("AI 分析完成，切換到步驟 3")
                                st.rerun()
            
            # Step 3: Generate Report
            elif st.session_state.current_step == 3:
                if st.session_state.analysis_complete:
                    if not hasattr(self, 'analysis_result') or self.analysis_result is None:
                        self.analysis_result = st.session_state.analysis_result
                    
                    UIComponents.create_analysis_section(self.analysis_result)
                    
                    # 添加生成報告按鈕
                    if st.button("生成醫美建議報告"):
                        with st.spinner("正在生成報告..."):
                            logger.info("開始生成報告...")
                            self.report_buffer = self.report_generator.generate_report(
                                self.analysis_result,
                                [st.session_state.uploaded_image]
                            )
                            st.session_state.report_buffer = self.report_buffer
                            st.session_state.report_generated = True
                            st.session_state.current_step = 4
                            logger.info("報告生成完成，切換到步驟 4")
                            st.rerun()
            
            # Step 4: View Results
            elif st.session_state.current_step == 4:
                if st.session_state.report_generated:
                    if not hasattr(self, 'analysis_result') or self.analysis_result is None:
                        self.analysis_result = st.session_state.analysis_result
                    if not hasattr(self, 'report_buffer') or self.report_buffer is None:
                        self.report_buffer = st.session_state.report_buffer
                    
                    UIComponents.create_analysis_section(self.analysis_result)
                    UIComponents.create_report_section(self.report_buffer)
                    
                    # 添加重新開始按鈕
                    if st.button("重新開始"):
                        for key in list(st.session_state.keys()):
                            if key not in ['fonts_registered', 'app_fonts_registered', 'selected_model']:
                                del st.session_state[key]
                        st.session_state.current_step = 1
                        st.session_state.image_processed = False
                        st.session_state.analysis_complete = False
                        st.session_state.report_generated = False
                        st.rerun()

        except Exception as e:
            logger.error(f"Error in main app flow: {str(e)}", exc_info=True)
            st.error("應用程序發生錯誤，請重試")

    def create_sidebar(self):
        """創建側邊欄，顯示應用程式基本操作步驟和模型選擇"""
        with st.sidebar:
            st.title("醫美診所智能評估系統")
            st.divider()
            
            # 模型選擇
            st.subheader("⚙️ 分析模型選擇")
            model_options = ["DeepSeek VL2", "grok-2-vision-1212", "GPT-4o"]
            selected_model = st.selectbox(
                "選擇分析模型", 
                options=model_options, 
                index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
            )
            if selected_model != st.session_state.selected_model:
                st.session_state.selected_model = selected_model
                logger.info(f"用戶選擇了 {selected_model} 模型")
            
            st.divider()
            
            st.subheader("📋 使用步驟")
            st.markdown("""
            1. **上傳照片** - 上傳您想要分析的臉部照片
            2. **AI 分析** - 系統使用 AI 進行皮膚狀況分析
            3. **生成報告** - 根據分析結果生成詳細評估報告
            4. **查看結果** - 查看分析結果和建議
            """)
            
            st.divider()
            st.subheader("ℹ️ 關於本系統")
            st.markdown("""
            本系統利用先進的 AI 技術，為醫美診所提供客戶臉部皮膚狀況的專業評估。
            系統分析照片後，會提供詳細的皮膚狀況報告，包括皮膚類型、問題區域以及改善建議。
            """)
            
            st.divider()
            st.caption("© 2025 醫美診所智能評估系統 - 版權所有")
            
            # 顯示當前步驟
            step_names = ["上傳照片", "AI 分析", "生成報告", "查看結果"]
            st.success(f"當前步驟: {step_names[st.session_state.current_step - 1]}")

    def create_upload_section(self):
        """創建上傳區域"""
        st.header("📸 上傳面部照片")
        
        uploaded_file = st.file_uploader("選擇面部照片進行分析", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            self.handle_upload(uploaded_file)

    def handle_upload(self, uploaded_file):
        """處理照片上傳，確保只執行一次"""
        if uploaded_file is None:
            return
            
        try:
            logger.info("開始驗證上傳的圖片")
            image = validate_image(uploaded_file)
            logger.info(f"圖片驗證成功，尺寸: {image.size}")
            
            # Store image in session state
            st.session_state.uploaded_image = uploaded_file
            st.session_state.image_processed = True
            logger.info("圖片已保存到 session_state")
            
            # Display preview
            st.image(image, caption="上傳的照片", use_container_width=True)
            
            # 切換到下一步
            st.session_state.current_step = 2
            logger.info(f"切換到步驟 {st.session_state.current_step}")
            st.rerun()
            
        except Exception as e:
            logger.error(f"Error handling upload: {str(e)}", exc_info=True)
            st.error(f"處理照片時發生錯誤: {str(e)}")

if __name__ == "__main__":
    # Verify API keys
    missing_keys = []
    if not DEEPSEEK_API_KEY:
        missing_keys.append("DEEPSEEK_API_KEY")
    if not XAI_API_KEY:
        missing_keys.append("XAI_API_KEY")
    if not REPLICATE_API_TOKEN:
        missing_keys.append("REPLICATE_API_TOKEN")
    
    if missing_keys:
        st.error(f"以下環境變量未設置: {', '.join(missing_keys)}，請檢查 .env 文件")
        st.stop()
    
    # Run app
    app = BeautyClinicApp()
    app.run()
