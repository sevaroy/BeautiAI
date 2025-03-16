# AI醫美智能評估系統 - MVP版本

這是一個使用多模態AI模型進行面部分析並生成醫美建議報告的應用程式。

## 功能特點

- 支援上傳面部照片進行分析
- 可選擇使用GPT-4o或DeepSeek VL2進行面部特徵分析
- 使用DeepSeek-R1生成專業醫美建議報告
- 提供報告下載功能

## 安裝與使用

### 環境要求

- Python 3.8+
- 相關API密鑰（OpenAI、Replicate、DeepSeek）

### 環境設置

本專案使用 `.env` 文件來管理環境變量。為了保護敏感信息，我們提供了一個 `.env.example` 模板文件。

### 設置步驟

1. 複製 `.env.example` 文件並重命名為 `.env`：
```bash
cp .env.example .env
```

2. 在 `.env` 文件中填入你的實際 API 密鑰：
```env
DEEPSEEK_API_KEY=你的_deepseek_api_密鑰
XAI_API_KEY=你的_xai_api_密鑰
```

注意：`.env` 文件包含敏感信息，已被添加到 `.gitignore` 中，不會被提交到版本控制系統。

### 安裝步驟

1. 克隆倉庫或下載源代碼

2. 安裝依賴包
   ```bash
   pip install -r requirements.txt
   ```

3. 運行應用
   ```bash
   streamlit run app.py
   ```

4. 在瀏覽器中訪問應用（默認地址：http://localhost:8501）

## 使用說明

1. 在應用介面上傳您的面部照片（支援JPG、JPEG、PNG格式）
2. 在側邊欄選擇分析模型（GPT-4o或DeepSeek VL2）
3. 點擊"開始分析"按鈕
4. 等待系統分析面部特徵並生成醫美建議報告
5. 查看分析結果和建議報告
6. 點擊"下載完整報告"保存報告

## 免責聲明

本系統生成的醫美建議僅供參考，在進行任何醫美治療前，請務必諮詢專業醫生的意見。

## 技術架構

- 前端介面：Streamlit
- 圖像分析：OpenAI GPT-4o / DeepSeek VL2
- 報告生成：DeepSeek-R1
- 資料處理：Python 