import io
from typing import List, Dict, Any
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import streamlit as st

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._register_fonts()

    def _register_fonts(self):
        try:
            # 使用 Streamlit 的 session_state 來追踪字體是否已註冊
            if 'fonts_registered' not in st.session_state:
                st.session_state.fonts_registered = False
                
            if not st.session_state.fonts_registered:
                # 尝试使用系统中文字体
                chinese_fonts = [
                    ('SimSun', 'simsun.ttc'),
                    ('SimHei', 'simhei.ttf'),
                    ('Microsoft YaHei', 'msyh.ttc')
                ]
                
                font_found = False
                for font_name, font_file in chinese_fonts:
                    try:
                        font_path = f"C:\\Windows\\Fonts\\{font_file}"
                        if os.path.exists(font_path):
                            pdfmetrics.registerFont(TTFont(font_name, font_path))
                            # 设置为默认字体
                            for style in self.styles.byName.values():
                                style.fontName = font_name
                            font_found = True
                            logger.info(f"成功加载系统字体: {font_name}")
                            st.session_state.fonts_registered = True
                            break
                    except Exception as e:
                        logger.warning(f"加载字体 {font_name} 失败: {str(e)}")
                        continue
                
                if not font_found:
                    # 如果没有找到中文字体，使用默认字体
                    default_font = 'Helvetica'
                    for style in self.styles.byName.values():
                        style.fontName = default_font
                    logger.warning("未找到中文字体，使用默认字体 Helvetica")
                    st.session_state.fonts_registered = True
            else:
                logger.debug("字體已註冊，跳過註冊過程")
                
        except Exception as e:
            logger.error(f"字体配置错误: {str(e)}")
            raise

    def generate_report(self, analysis_result: Dict[str, Any], images: List[io.BytesIO]) -> io.BytesIO:
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            story = []

            # Add title
            title_style = self.styles['Title']
            story.append(Paragraph("医美诊所智能评估报告", title_style))
            story.append(Spacer(1, 30))

            # Add analysis results
            self._add_analysis_section(story, analysis_result)
            
            # Add visualizations
            self._add_visualizations(story, analysis_result, images)

            # Build PDF
            doc.build(story)
            buffer.seek(0)
            return buffer

        except Exception as e:
            logger.error(f"生成报告错误: {str(e)}")
            raise

    def _add_analysis_section(self, story: List, analysis_result: Dict[str, Any]):
        heading_style = self.styles['Heading1']
        normal_style = self.styles['Normal']

        try:
            # Check if there's an error at the top level
            if "error" in analysis_result:
                story.append(Paragraph("分析错误", heading_style))
                story.append(Spacer(1, 12))
                story.append(Paragraph(f"错误信息: {analysis_result['error']}", normal_style))
                story.append(Spacer(1, 20))
                return

            # 從新的分析結果結構中獲取數據
            analysis = analysis_result.get('analysis', {})
            
            # Handle error in analysis
            if isinstance(analysis, dict) and "error" in analysis:
                story.append(Paragraph("分析错误", heading_style))
                story.append(Spacer(1, 12))
                story.append(Paragraph(f"错误信息: {analysis['error']}", normal_style))
                story.append(Spacer(1, 20))
                return
                
            # Extract model and result based on structure
            if isinstance(analysis, dict):
                model = analysis.get('model', 'Unknown')
                
                # Handle different result formats
                if 'result' in analysis:
                    result = analysis['result']
                else:
                    # If no 'result' key, try to use the whole analysis as result
                    result = str(analysis)
            else:
                # Fallback for unexpected cases
                model = 'Unknown'
                result = str(analysis) if analysis else '暂无分析结果'
            
            # Ensure result is a string
            if not isinstance(result, str):
                result = str(result)

            # 添加模型信息和分析結果
            story.append(Paragraph(f"{model} 分析结果", heading_style))
            story.append(Spacer(1, 12))
            story.append(Paragraph(result, normal_style))
            story.append(Spacer(1, 20))
            
        except Exception as e:
            logger.error(f"添加分析部分时出错: {str(e)}")
            story.append(Paragraph("分析结果处理错误", heading_style))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"处理分析结果时发生错误: {str(e)}", normal_style))
            story.append(Spacer(1, 20))

    def _add_visualizations(self, story: List, analysis_result: Dict[str, Any], images: List[io.BytesIO]):
        try:
            # Add original image
            if images:
                img = Image(images[0], width=400, height=300)
                story.append(img)
                story.append(Spacer(1, 20))

            # Add radar chart (with error handling)
            try:
                radar_chart = self._create_radar_chart()
                story.append(radar_chart)
                story.append(Spacer(1, 20))
            except Exception as e:
                logger.error(f"無法生成雷達圖: {str(e)}")
                
            # Add skin analysis chart (with error handling)
            try:
                if images:
                    skin_chart = self._create_skin_analysis_chart()
                    story.append(skin_chart)
            except Exception as e:
                logger.error(f"無法生成皮膚分析圖: {str(e)}")
        except Exception as e:
            logger.error(f"添加可視化圖表時出錯: {str(e)}")

    def _create_radar_chart(self) -> Image:
        try:
            # Create radar chart visualization
            categories = ['膚質', '皺紋', '色斑', '毛孔', '彈性']
            values = [0.8, 0.6, 0.7, 0.5, 0.9]  # Example values

            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
            values = np.concatenate((values, [values[0]]))
            angles = np.concatenate((angles, [angles[0]]))

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            ax.plot(angles, values)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plt.close(fig)  # 關閉圖形，避免內存泄漏
            
            return Image(buffer, width=300, height=300)
        except Exception as e:
            logger.error(f"無法生成雷達圖: {str(e)}")
            raise

    def _create_skin_analysis_chart(self) -> Image:
        try:
            # Create skin analysis visualization
            categories = ['油性', '乾性', '敏感', '色素沉澱']
            values = [70, 30, 50, 40]  # Example values

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(categories, values)
            ax.set_ylabel('程度 (%)')
            ax.set_title('皮膚狀況分析')

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plt.close(fig)  # 關閉圖形，避免內存泄漏
            
            return Image(buffer, width=400, height=300)
        except Exception as e:
            logger.error(f"無法生成皮膚分析圖: {str(e)}")
            raise
