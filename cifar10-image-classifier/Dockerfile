# 1. Python 3.9 (軽量版) をベースにする
FROM python:3.9-slim

# 2. 作業ディレクトリを作成
WORKDIR /app

# 3. 必要なパッケージリストをコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. アプリケーションコードとモデルファイルをコピー
# (注意: ここでローカルにある .pth ファイルもコンテナ内にコピーされます)
COPY . .

# 5. ポート8000を開放
EXPOSE 7860

# 6. アプリケーション起動
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]