## 專案介紹

本專案是一個 **基於 DeepFM 模型的電商點擊率預測與推薦模擬**，使用 Flipkart 手機數據集訓練 DeepFM 模型來預測用戶的點擊行為，並根據預測結果為用戶推薦最可能感興趣的商品。

## 功能特點

- **數據預處理**：清理並轉換 Flipkart 手機數據，填補缺失值、特徵編碼與標準化處理。
- **DeepFM 模型訓練**：使用 deepctr-torch 設計 DeepFM 模型，進行 CTR (Click-Through Rate) 預測。
- **模型評估**：計算混淆矩陣與準確率，分析模型效能。
- **推薦系統**：基於預測點擊率 (CTR)，為每位用戶推薦 5 款最可能點擊的手機商品。
- **結果可解釋性**：透過反向標籤編碼與標準化還原，提供直觀的推薦結果。

## 安裝與使用

### 1. 環境準備

請確保已安裝 Python 3.10 以上版本，並安裝必要的 Python 套件。

```bash
pip install kaggle deepctr-torch torch scikit-learn tensorflow pandas numpy
```

### 2. 設置 Kaggle API 金鑰

請至 [Kaggle](https://www.kaggle.com/) 取得 API Key，並設置環境變數。

```python
import os
os.environ['KAGGLE_USERNAME'] = "your_kaggle_username"
os.environ['KAGGLE_KEY'] = "your_kaggle_key"
```

### 3. 下載與解壓數據集

```bash
!kaggle datasets download -d devsubhash/flipkart-mobiles-dataset
!unzip flipkart-mobiles-dataset.zip -d data/
```

### 4. 執行程式

```python
python deepfm_recommendation.py
```

程式將自動進行數據預處理、模型訓練、預測與推薦結果輸出。

## 技術細節

### 數據預處理

- 處理缺失值並填補數據。
- 創建新特徵，如 **價格差異 (Original Price - Selling Price)**。
- 進行 **Label Encoding、One-Hot Encoding**。
- 數值特徵進行 **標準化 (StandardScaler) 與 Min-Max 轉換**。

### DeepFM 模型訓練

- 使用 **deepctr-torch** 設置 DeepFM 模型。
- 類別特徵 (Sparse Features) 使用嵌入層處理。
- 數值特徵 (Dense Features) 直接輸入模型。
- 優化器使用 **Adam**，損失函數為 **binary cross-entropy**。
- 訓練 30 個 Epoch，批量大小 256。

### 模型評估

- 計算 **混淆矩陣 (Confusion Matrix)**。
- 估算 **準確率 (Accuracy Score)**。

### 推薦系統

- 根據預測點擊率 (CTR) 進行排序。
- 為每位用戶推薦 **5 個點擊率最高的手機商品**。

## 預測結果示例

| Customer ID | Brand   | Model                  | Color           | Memory(GB/MB) | Storage(GB) | Selling Price | pred_CTR |
|------------|--------|-----------------------|----------------|---------------|------------|--------------|----------|
| 2          | vivo   | Y83 Pro               | Nebula Purple  | 4.0           | 64         | 14500.0      | 5.457096e-05 |
| 4          | Apple  | iPhone 7 Plus         | Rose Gold      | 3.0           | 128        | 42900.0      | 9.964652e-01 |
| 7          | Infinix| Hot 7 Pro             | Mocha Brown    | 6.0           | 64         | 10999.0      | 1.652500e-01 |
| 8          | realme | 9i                     | Prism Black    | 6.0           | 128        | 15999.0      | 9.256600e-04 |
| 9          | Xiaomi | 11 Lite NE            | Tuscany Coral  | 8.0           | 128        | 23999.0      | 9.982635e-01 |
| 12         | Apple  | iPhone 12 Mini        | Green          | 4.0           | 128        | 61999.0      | 3.009395e-04 |
| 12         | SAMSUNG| Galaxy J2 Core        | Blue           | 1.0           | 8          | 6500.0       | 6.155065e-06 |
| 13         | OPPO   | A15                    | Dynamic Black  | 2.0           | 32         | 8990.0       | 8.491672e-01 |
| 13         | Nokia  | XPlus                  | Bright Green   | 768.0         | 4          | 7990.0       | 5.580077e-07 |
| 14         | realme | Narzo 30 Pro 5G       | Blade Silver   | 8.0           | 128        | 19999.0      | 8.148954e-10 |

## 未來改進
DeepFM 模型學習了用戶與商品之間的特徵交互，但在驗證集上的 AUC 表現較低，產生過擬合現象，具體原因有待商榷，可能為資料集內容、資料前處理或特徵選擇。
- 資料集改善：目前重要內容（用戶、CTR）為額外隨機生成，可改為真實數據進行訓練。
- 模型正規化與調參：調整 Dropout、L2 正則化、學習率等參數，以改善模型泛化能力。 
- 特徵工程優化：重新評估數值與類別特徵的選擇與處理方式，以提升模型效果。 
- 替換模型： DeepFM 為針對 CTR 之單一預測模型，若資料集中數據關係複雜，或需同時預測多項輸出，可考慮其他推薦模型如 MMOE 多任務學習模型、GNN 圖神經網路模型等。 

## 授權條款

本專案採用 **MIT License**，歡迎自由使用與修改。
