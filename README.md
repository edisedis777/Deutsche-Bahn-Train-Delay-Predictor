# Deutsche-Bahn-Train-Delay-Predictor
A machine learning pipeline to predict train delays in Germany using the Deutsche Bahn dataset

---

```markdown
# 🚆 Deutsche Bahn Train Delay Predictor

This project uses historical Deutsche Bahn (DB) train data to build a machine learning model
that predicts train delays in Germany.
The final solution includes a full pipeline:

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

We use these records to calculate and predict **train delay in minutes**.

**Credit:** All data used in this project belongs to and is made publicly available 
by [piebro](https://github.com/piebro).

---

## 🧠 Project Goal

Predict whether a train will be delayed (and by how many minutes) based on features 
such as:

- Train type
- Station
- Planned arrival time
- Day of the week and hour
- (Optional) Weather and holiday data

---

## ⚙️ Project Structure

```markdown

├── preprocessing.py       # Clean & transform data
├── train\_model.py              # PyTorch model training
├── model.pth                   # Saved trained model
├── app/
│   ├── app.py                  # Flask API
│   ├── templates/
│   │   └── form.html           # Web form UI
│   └── static/
├── monitor.py                  # Request logging and drift detection
├── requirements.txt
├── deploy.sh                   # Cloud deployment script
├── DB_Delay_Predictor.pdf      # 2-page project documentation
└── README.md                   # You're here
```

---

## 🏗️ Step-by-Step Overview

### 1. 🧹 Data Preprocessing

We cleaned and joined CSV files from piebro’s repo, 
parsed planned vs actual times, 
and computed delay in minutes. 

Then we extracted:
- Hour and day of the week
- One-hot encoded train type and station ID

```bash
python data_preprocessing.py
````

This generates a cleaned dataset in `./processed_data/`.

---

### 2. 🧪 Model Training (PyTorch)

We trained a **Multilayer Perceptron (MLP)** regression model on tabular features using PyTorch.

* Loss: MSE (Mean Squared Error)
* Optimizer: Adam
* Metrics: MAE, R² Score

```bash
python train_model.py
```

This creates `model.pth`.

---

### 3. 🌐 API with Flask

We expose a `/predict` endpoint that accepts JSON input 
and returns the predicted delay in minutes.

```bash
cd app
python app.py
```

Example request:

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

A simple form interface is served via Flask using HTML/CSS. 
Users input details like station, time, and train type. 
Output is shown directly on the page.

Visit: `http://localhost:5000/`

---

### 5. 📊 Monitoring Pipeline

Logs:

* Timestamp of prediction
* Inference time
* Input/output data
* Input distribution monitoring (drift alert)

Saved to local `logs.db` using SQLite.

```bash
python monitor.py
```

---

### 6. ☁️ Cloud Hosting

Deployment script is provided (`deploy.sh`) to run:

* Flask API served via Gunicorn
* On a Render server or an Ubuntu VM with Nginx proxy

---

### 7. 📄 Documentation (PDF)

All project components, architecture diagrams, usage instructions, 
and results are summarized in `DB_Delay_Predictor.pdf`.

---

## 📈 Example Results

| Metric         | Value    |
| -------------- | -------- |
| MAE            | 3.2 mins |
| R² Score       | 0.68     |
| Inference Time | <100ms   |

Note: These are example metrics from a sample test split.

---

## 🧠 Future Improvements

* Add weather and holiday features
* Use XGBoost or LightGBM for comparison
* Add map visualizations of delay hotspots
* Dockerize the app

---

## 📜 License & Credits

* Data: [piebro/deutsche-bahn-data](https://github.com/piebro/deutsche-bahn-data)
* Code: © 2025 

---

## 🙌 Contributing

Feel free to open issues or submit pull requests to improve the model, UI, or monitoring features.

---

## 🛠️ How to Run

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Preprocess data
python data_preprocessing.py

# Step 3: Train model
python train_model.py

# Step 4: Run API + Web App
cd app
python app.py
```

---



