# 🧪 Chemical Yield Predictor

A Streamlit-powered machine learning dashboard that predicts the **chemical reaction yield (%)** from experimental parameters like temperature, pressure, and catalyst concentration. Built with Python, MySQL, scikit-learn, and a user-friendly web interface.

---

## 🚀 Features

- 📊 Predict **reaction yield (%)** from:
  - Temperature (°C)
  - Pressure (atm)
  - Catalyst Concentration (mol/L)
- 🤖 Supports 3 trained ML models:
  - K-Nearest Neighbors (KNN)
  - Support Vector Regressor (SVR)
  - Neural Network (MLP)
- 📁 Fetches real-time data from **MySQL** database
- 📉 Visualizes how yield is affected by each parameter
- 🌐 Interactive web interface via **Streamlit**

---

## 🧰 Tech Stack

| Layer              | Tools Used                                |
|--------------------|--------------------------------------------|
| 🧠 ML Models        | scikit-learn (KNN, SVR, MLPRegressor)       |
| 📊 Frontend         | Streamlit                                  |
| 🧪 Data Storage     | MySQL + Python connector                   |
| 📁 Others           | Pandas, Matplotlib, Seaborn, Joblib       |

---

## 📂 Project Structure

```
chemical-yield-predictor/
│
├── main.py              # Trains ML models, saves them
├── predict.py           # Streamlit web app
├── knn_model.pkl        # Saved KNN model
├── svr_model.pkl        # Saved SVR model
├── neural_net_model.pkl # Saved Neural Network model
├── scaler.pkl           # Feature scaler
├── init_experiments.sql # MySQL table schema + sample data
├── requirements.txt     # Python dependencies
├── chemical_data.csv    # Dataset (if not loaded from SQL)
└── README.md            # You're reading this
```

---

## ⚙️ How It Works

1. MySQL stores reaction data in a table named `data` under DB `experiments`
2. `main.py` fetches data, scales features, trains KNN/SVR/NN models
3. Trained models and the scaler are saved as `.pkl` files
4. `predict.py` runs a Streamlit interface:
   - Accepts experimental inputs
   - Lets you choose a model
   - Returns the predicted reaction yield in %

---

## 📉 Model Performance (MSE)

| Model      | MSE (on test data) |
|------------|--------------------|
| KNN        | *(printed during training)* |
| SVR        | *(printed during training)* |
| Neural Net | *(printed during training)* |

---

## 📈 Data Visualizations

### 🔥 Effect of Temperature on Yield
![Temperature vs Yield 1](image/temp%20vs%20yield.png)
![Temperature vs Yield 2](image/temp%20vs%20yield%202.png)

### 💨 Effect of Pressure on Yield
![Pressure vs Yield 1](image/pressure%20vs%20yield.png)
![Pressure vs Yield 2](image/pressure%20vs%20yield2.png)

### 🧪 Effect of Catalyst Concentration on Yield
![Catalyst vs Yield 1](image/catalyst%20con%20vs%20yield.png)
![Catalyst vs Yield 2](image/catalyst%20con%20vs%20yield2.png)

---

## 🖥️ Web App Preview

### 🧪 Input Interface
<img src="image/Screenshot 2025-07-29 151529.png" alt="Input Form" width="60%">

### 🔍 Model Selection
<img src="image/Screenshot 2025-07-29 151556.png" alt="Model Dropdown" width="50%">

### 🎯 Prediction Result
<img src="image/Screenshot 2025-07-29 151612.png" alt="Prediction Output" width="48%">


---

## 🛠️ Run Locally

### 1. Clone this repo
```bash
git clone https://github.com/hhemantpatel/chemical-yield-predictor.git
cd chemical-yield-predictor
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up MySQL

- Create DB `experiments`
- Import SQL schema and data:
```sql
source init_experiments.sql;
```

### 4. Train models
```bash
python main.py
```

### 5. Launch Streamlit app
```bash
streamlit run predict.py
```

---

## 📥 Dataset Source

- **Kaggle**: [Chemical Yield Dataset](https://www.kaggle.com/datasets/ayushbarnawal/chemical-yield)

---

## 👨‍🔬 Author

**Hemant Patel**  
B.Tech, Chemical Engineering  
IIT Jodhpur  
📧 patelhemant.7509@gmail.com  
🔗 [GitHub @hhemantpatel](https://github.com/hhemantpatel)

---

> ⭐ If you find this project useful, consider starring the repo!
