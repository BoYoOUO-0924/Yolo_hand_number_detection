# 🖐️ YOLO11 手部數字手勢辨識系統 (Hand Digit Recognition)

基於 Ultralytics YOLO11 與 Roboflow 手勢資料集打造的高效能手勢數字辨識系統。支援 NVIDIA RTX 3060 GPU 加速、即時 Webcam 視訊推論、多手數字加總與抗抖動平滑濾波。

---

## 📁 專案目錄結構

```text
Yolo_hand_detection/
├── dataset/                  # 資料集目錄 (存放 train/valid/test 與 data.yaml)
├── weights/                  # 模型權重 (存放 best.pt 與 last.pt)
├── snapshots/                # 即時推論按 's' 儲存的截圖
├── runs/                     # YOLO 訓練與驗證日誌、曲線圖、混淆矩陣
├── src/
│   ├── config.py             # 全域設定 (路徑、色彩、標籤映射)
│   ├── dataset_manager.py    # 資料集解壓、路徑修正與分佈檢查
│   ├── trainer.py            # YOLO11 訓練封裝
│   ├── evaluator.py          # 模型評估與指標計算
│   ├── hand_detector.py      # 手勢辨識與時序平滑濾波核心
│   └── visualizer.py         # OpenCV HUD 與邊界框繪製工具
├── main_infer.py             # 即時 Webcam / 影片 / 照片推論主程式
├── train.py                  # 訓練入口腳本
├── evaluate.py               # 評估入口腳本
├── requirements.txt          # 依賴套件清單
└── README.md                 # 專案說明文件
```

---

## 🚀 快速開始

### 1. 啟用虛擬環境
本專案已建立專屬 `.venv` 虛擬環境（支援 PyTorch CUDA 12.4）：

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

---

### 2. 資料集解壓與健康度檢查

本系統支援自動解壓 Roboflow 下載的 zip 壓縮檔（例如 `Hand Detection.v1i.yolov11.zip`）：

```powershell
# 自動解壓指定 zip 檔並校正 data.yaml
.\.venv\Scripts\python.exe src/dataset_manager.py "C:\Users\tung0\Downloads\Hand Detection.v1i.yolov11.zip"
```

---

### 3. 開始訓練模型

使用本地 RTX 3060 進行訓練，預設使用 `yolo11s.pt`（可更換為 `yolo11n.pt`）：

```powershell
# 預設訓練 100 個 Epochs，Batch Size 16
.\.venv\Scripts\python.exe train.py --epochs 100 --batch 16 --model yolo11s.pt --device 0
```

> **訓練提示**：
> - 訓練完成後，最佳權重會自動儲存於 `weights/best.pt`。
> - 訓練過程各項曲線（Loss、mAP、PR-Curve、混淆矩陣）會保存在 `runs/hand_detection/` 目錄中。

---

### 4. 評估模型效能

評估訓練後的模型在驗證集（`val`）或測試集（`test`）上的 mAP@50 與 Precision/Recall：

```powershell
# 評估驗證集
.\.venv\Scripts\python.exe evaluate.py --weights weights/best.pt --split val

# 評估測試集
.\.venv\Scripts\python.exe evaluate.py --weights weights/best.pt --split test
```

---

### 5. 啟動即時攝影機辨識 (Real-Time Webcam HUD)

使用 OpenCV 啟動即時手勢數字辨識視窗：

```powershell
# 預設啟動 0 號鏡頭
.\.venv\Scripts\python.exe main_infer.py --weights weights/best.pt --source 0

# 指定信心度閥值 (例如 0.60)
.\.venv\Scripts\python.exe main_infer.py --weights weights/best.pt --source 0 --conf 0.60
```

#### 🎮 即時視窗快捷鍵說明：
| 快捷鍵 | 功能說明 |
| :--- | :--- |
| `q` 或 `ESC` | 退出程式 |
| `s` | 擷取當前辨識畫面並儲存至 `snapshots/` |
| `c` | 循環切換攝影機 (0, 1, 2...) |
| `Space` | 暫停 / 繼續推論 |
| `+` / `-` | 即時調整信心度閥值 (+/- 0.05) |

---

### 6. 單張圖片或影片辨識測試

您也可以將任何圖片或影片傳入測試：

```powershell
# 測試單張圖片
.\.venv\Scripts\python.exe main_infer.py --weights weights/best.pt --source dataset/test/images/sample.jpg

# 測試影片檔案
.\.venv\Scripts\python.exe main_infer.py --weights weights/best.pt --source test_video.mp4
```

---

## 🎯 核心功能特色

1. **數字自動映射**：支援將標籤（如 `Hand-one` ~ `Hand-five` / `Hand-zero`）轉換為標準整數數值。
2. **多手加總 HUD**：畫面中同時出現雙手或多隻手時，頂部資訊列會自動即時顯示各手數字與總和（如 `[5] + [3] = 8`）。
3. **時序平滑濾波 (Temporal Smoothing)**：內建移動平均平滑器，消除因手部晃動或光影變化引起的邊界框抖動與閃爍。
4. **極致即時效能**：在 RTX 3060 上推論延遲約 5~10ms，輕鬆達到 60+ FPS。
