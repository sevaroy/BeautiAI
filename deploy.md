# EC2 部署指南

## 前置準備
1. 確保您有一個運行中的 EC2 實例（建議使用 Ubuntu Server）
2. 已配置好安全組，開放 8501 端口（或需要的其他端口）
3. 有 SSH 訪問權限
4. 準備好所需的 API Keys:
   - OPENAI_API_KEY
   - REPLICATE_API_KEY
   - DEEPSEEK_API_KEY
   - XAI_API_KEY
   - REPLICATE_API_TOKEN

## 部署步驟

### 1. 連接到 EC2
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

### 2. 安裝 Docker
```bash
# 更新包列表
sudo apt update

# 安裝 Docker
sudo apt install -y docker.io

# 啟動 Docker 服務
sudo systemctl start docker
sudo systemctl enable docker

# 將當前用戶加入 docker 組（免去每次都要 sudo）
sudo usermod -aG docker $USER

# 重新登錄以使權限生效
exit
# 重新 SSH 登錄
```

### 3. 準備環境變量
```bash
# 克隆代碼
git clone https://github.com/[您的用戶名]/BeautiAI.git
cd BeautiAI

# 創建並編輯 .env 文件
cp .env.example .env
nano .env

# 在 .env 文件中填入您的 API Keys:
# OPENAI_API_KEY=your_openai_api_key_here
# REPLICATE_API_KEY=your_replicate_api_key_here
# DEEPSEEK_API_KEY=your_deepseek_api_key_here
# XAI_API_KEY=your_xai_api_key_here
# REPLICATE_API_TOKEN=your_replicate_api_token_here
```

### 4. 部署應用
```bash
# 構建 Docker 映像
docker build -t beautiai .

# 運行容器（使用 --env-file 載入環境變量）
docker run -d \
  --name beautiai \
  -p 8501:8501 \
  --env-file .env \
  --restart unless-stopped \
  beautiai
```

### 5. 確認部署
- 訪問 `http://[您的EC2公網IP]:8501` 查看應用是否正常運行
- 檢查容器狀態：`docker ps`
- 查看容器日誌：`docker logs beautiai`

### 6. 更新應用
```bash
# 拉取最新代碼
git pull

# 重新構建映像
docker build -t beautiai .

# 停止並刪除舊容器
docker stop beautiai
docker rm beautiai

# 運行新容器（記得加上 --env-file）
docker run -d \
  --name beautiai \
  -p 8501:8501 \
  --env-file .env \
  --restart unless-stopped \
  beautiai
```

## 故障排查
- 如果應用無法訪問，檢查 EC2 安全組設置
- 查看容器日誌：`docker logs beautiai`
- 檢查容器狀態：`docker ps -a`
- 進入容器檢查：`docker exec -it beautiai bash`
- 確認環境變量是否正確載入：`docker exec beautiai env`

## 注意事項
1. `.env` 文件包含敏感信息，確保不要上傳到 Git
2. 建議配置 HTTPS 以提高安全性
3. 定期備份重要數據
4. 監控服務器資源使用情況
