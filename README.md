# 🚆 Deutsche Bahn Train Delay Predictor

This project uses historical Deutsche Bahn (DB) train data to build a machine learning model  
that predicts train delays in Germany.  
The final solution includes a full pipeline:
```markdown
- ✅ Data Preprocessing  
- ✅ PyTorch Model Training  
- ✅ Flask API to Serve Predictions  
- ✅ Web App Interface for Users  
- ✅ Monitoring and Logging System  
- ✅ Cloud Deployment (e.g. Render, AWS EC2, Azure)  
- ✅ Two-Page PDF Project Report  
```
---

## 📁 Dataset

The dataset is from [piebro/deutsche-bahn-data](https://github.com/piebro/deutsche-bahn-data),  
which contains raw arrival and departure times for DB trains across Germany.  

I used these records to calculate and predict **train delay in minutes**.

**Credit:** All data used in this project belongs to and is made publicly available  
by [piebro](https://github.com/piebro).

---

## 🧠 Project Goal

Predict whether a train will be delayed (and by how many minutes) based on features such as:

- Train type  
- Station  
- Planned arrival time  
- Day of the week and hour  
- (Optional) Weather and holiday data  

---

## ⚙️ Project Structure

```markdown
├── data_processing.py         # Clean & transform raw DB data  
├── train_model.py             # PyTorch model training script  
├── model.pth                  # Saved trained model  
├── app/  
│   ├── app.py                 # Flask API + Web app backend  
│   ├── templates/  
│   │   └── form.html          # Web form UI  
│   └── static/                # CSS, JS, images  
├── monitor.py                 # Request logging and drift detection  
├── requirements.txt           # Python dependencies  
├── save_feature_columns.py    # Save one-hot feature column names after preprocessing  
├── deploy.sh                  # Cloud deployment script  
├── DB_Delay_Predictor.pdf     # 2-page project documentation  
└── README.md                  # You're here  
```

---

## 🏗️ Step-by-Step Overview

### 1. 🧹 Data Preprocessing

* Clean raw DB train data from piebro’s repo
* Parse planned vs actual arrival times and compute delay in minutes
* Extract and one-hot encode features: train type, station, hour, weekday

```bash
python data_processing.py
```

This generates cleaned and processed data files (CSV/Parquet) used for training.

---

### 2. 🧪 Model Training (PyTorch)

Train a **Multilayer Perceptron (MLP)** regression model on the processed tabular features.

* Loss: MSE (Mean Squared Error)
* Optimizer: Adam
* Metrics: MAE, R² Score

```bash
python train_model.py
```

This creates the trained model file `model.pth`.

---

### 3. 🌐 API with Flask

Start the Flask API serving a `/predict` endpoint that accepts JSON input and returns predicted delay in minutes.

```bash
cd app
python app.py
```

Example request JSON:

```json
{
  "station_id": "8000105",
  "train_type": "ICE",
  "hour": 16,
  "weekday": 2
}
```

---

### 4. 🖥️ Web App

A simple form interface served via Flask lets users enter details (station, time, train type) and shows prediction results directly on the page.

Access at: `http://localhost:5000/`

---

### 5. 📊 Monitoring Pipeline

The monitoring script logs:

* Timestamp of prediction
* Inference time
* Input/output data
* Input distribution monitoring for data drift alerts

Logs are saved in a local SQLite database `logs.db`.

```bash
python monitor.py
```

---

### 6. ☁️ Cloud Hosting

Deployment script (`deploy.sh`) automates:

* Running Flask API with Gunicorn
* Hosting on Render or an Ubuntu VM with Nginx reverse proxy

---

### 7. 📄 Documentation (PDF)

Complete project documentation including architecture diagrams, usage, and results is available in `DB_Delay_Predictor.pdf`.

---

## 📈 Example Results

| Metric         | Value    |
| -------------- | -------- |
| MAE            | 3.2 mins |
| R² Score       | 0.68     |
| Inference Time | <100ms   |

*Note: Example metrics from a sample test split.*

---

## 🧠 Future Improvements

* Add weather and holiday data features
* Experiment with XGBoost or LightGBM models
* Add interactive map visualizations of delay hotspots
* Dockerize the entire application for portability

---

## 📜 License & Credits

* Data source: [piebro/deutsche-bahn-data](https://github.com/piebro/deutsche-bahn-data)
* Code & project: © 2025 GNU AFFERO GENERAL PUBLIC LICENSE

---

## 🙌 Contributing

Feel free to open issues or submit pull requests to improve the model, UI, or monitoring.

---

## 🛠️ How to Run

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Preprocess data
python data_processing.py

# Step 3: Train model
python train_model.py

# Step 4: Run API and Web App
cd app
python app.py
```

---
