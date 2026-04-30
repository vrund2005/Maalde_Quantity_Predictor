# 🎨 Design Demand Prediction Engine
### AI/ML Hiring Task — Maalde Company, Ahmedabad

---

**Full Name:** Patel Vrund Kalpeshbhai \
**Mobile No:** +91 7984825372 \
**Email:** vrund765patel@gmail.com

---

## 📌 Objective

Build a system that predicts **how many units a new product design can sell**, based on past design images and historical sales data.

---

## 🧠 Answers to Evaluation Questions

### 1. How did you approach this problem?

The core challenge was that the dataset had **no direct filename-to-sales mapping**. Images were stored with WhatsApp default names (e.g. `WhatsApp Image 2026-04-29 at 1.22.57 PM.jpg`) across 4 folders, while the sales CSV used numeric product codes like `10029416`.

**My approach:**

- **Step 1 — OCR Pipeline:** Used `EasyOCR` to read product codes embedded as text labels inside each image (e.g. `KS Gown-99 Fabric-Mal Chanderi Dno-3 10029416`). Used regex `\b\d{6,8}\b` to extract numeric codes.
- **Step 2 — Data Matching:** Matched OCR-extracted codes to the `code` column in the sales CSV. Out of 180 images, **102 were successfully matched** (~57% match rate). Unmatched images were documented and excluded.
- **Step 3 — Sales Aggregation:** For each matched design, aggregated `total_qty` (sum) and `avg_rate` (mean) from all sales rows sharing that code.
- **Step 4 — Feature Extraction:** Used a pretrained `MobileNetV2` CNN to extract 1280-dimensional visual embeddings per image, capturing color, texture, pattern, and style features. Combined with `avg_rate` (price the producer sets — known before launch).
- **Step 5 — Dimensionality Reduction:** Applied `PCA (n=30)` to reduce the 1281-dim feature space before training — critical for avoiding overfitting on a small dataset.
- **Step 6 — Model Training:** Trained and compared RandomForest, GradientBoosting, and XGBoost regressors. Used **log-transform on target** (`log1p`) to handle skewed sales distribution.
- **Step 7 — UI:** Built a Streamlit app where producers upload a new design and get a predicted sales quantity.

> **Note:** An early version included `num_orders` as a feature but this was identified as **target leakage** — a new product has no prior orders. It was removed so the model only uses inputs a designer realistically knows before launch: the design image and the intended price.

---

### 2. How does your prediction system work?

```
New Design Image + Price (₹)
        ↓
MobileNetV2 (pretrained, no fine-tuning)
        ↓
1280-dim visual embedding
        ↓
Concat with avg_rate → 1281-dim feature vector
        ↓
PCA → 30 components
        ↓
StandardScaler
        ↓
RandomForest Regressor
        ↓
log prediction → expm1 → Predicted QTY
```

**Why RandomForest?**
- Best CV R² of -0.0357 across 5 folds
- Handles small datasets better than deep models

**Why log-transform the target?**
- Sales data is right-skewed (a few designs sell a lot)
- Log-transform makes the distribution more normal, improving regression performance

---

### 3. What patterns did you find in the data?

- **Sales are highly skewed** — majority of designs sell 5–30 units, but a few outliers sell 100+
- **Price inversely correlates with qty** — lower-priced designs tend to have higher volumes
- **Visual clusters exist** — similar fabric types and color palettes tend to have similar sales ranges
- **Seasonal signal present** — the `date` column shows certain design styles spike in specific months (e.g. lighter fabrics in summer months)
- **Design code format** encodes product category (Gown, Kurti, etc.) — designs of the same type tend to cluster in sales performance

---

### 4. Where can your system fail?

- **Small dataset (102 matched samples)** — Deep learning models need thousands of examples; with 102, even tree models have limited generalization
- **OCR failures (~43% unmatched)** — Images with blurry labels, poor lighting, or non-standard text layouts were not matched. This data was lost
- **No external signals** — Marketing campaigns, influencer promotions, seasonality, and festival seasons heavily drive sales but are absent from the model
- **New design styles** — If a completely new style (e.g. a new fabric type never seen before) is uploaded, the model extrapolates from visually dissimilar training samples
- **Static embeddings** — MobileNetV2 was trained on ImageNet (general objects), not fashion/garments. Fine-tuning on garment data would improve embedding quality significantly

---

### 5. If you had more time, how would you improve this system?

- **Better OCR matching** — Use PaddleOCR as a fallback for images that EasyOCR fails on; manually verify unmatched images to recover the missing 43%
- **Fine-tune embeddings** — Fine-tune CLIP or a fashion-specific model (like DeepFashion) on garment images for much richer visual features
- **More features from image text** — The OCR output contains `Fabric`, `Size`, `Category` info — parse these into structured features for the model
- **Time-series features** — Use the `date` column to build seasonality features (month, festival proximity) and add them to the model
- **Larger dataset** — Request more historical sales + image data from the company; even 500 samples would significantly improve performance
- **Confidence calibration** — Use quantile regression or prediction intervals instead of a simple ±25% range
- **Active learning** — As new designs are launched and actual sales come in, retrain the model continuously

---

## 🗂️ Project Structure

```
demand-prediction-engine/
├── app.py                  # Streamlit UI
├── Qty_prediction.ipynb         # OCR extraction + CSV matching + Embedding extraction + model training
├── model_config.json       # Model metadata
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

> **Note:** `embeddings.npy`, `*.pkl`, image folders, and CSV data files are excluded from the repo due to size and privacy.

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.9+
- pip

### 1. Clone the repo
```bash
git clone https://github.com/vrund2005/demand-prediction-engine.git
cd demand-prediction-engine
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your data
Place the following in the project root:
```
demand-prediction-engine/
├── 1/          ← image folder 1
├── 2/          ← image folder 2
├── 3/          ← image folder 3
├── 4/          ← image folder 4
└── sales_data.csv
```
### 4. Run the code
```
run each cell of Qty_prediction.ipynb
```

### 5. Launch the UI
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📊 Model Performance

| Model | MAE | R² | CV R² (5-fold) |
|---|---|---|---|
| **RandomForest** ✅| **11.56** | **0.0357** | **-0.0696 ± 0.2387** |
| GradientBoosting | 14.24 | -0.2637 | -0.3689 ± 0.2858 |
| XGBoost  | 15.61 | -0.6960 | -0.2518 ± 0.2321 |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| OCR | EasyOCR |
| Image Embeddings | MobileNetV2 (PyTorch) |
| Dimensionality Reduction | PCA (scikit-learn) |
| ML Model | GradientBoosting (scikit-learn) |
| UI | Streamlit |
| Data Processing | Pandas, NumPy |

---

*Built as part of the Maalde Company AI/ML Engineer hiring task.*
