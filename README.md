# 🧪 Chemical Yield Predictor

A web-based machine learning project that predicts the yield of a chemical reaction based on experimental conditions such as temperature, pressure, and catalyst concentration. This project combines SQL database handling, Python data science tools, and a user-friendly Streamlit interface.

---

## 🚀 Features

- 📊 Predicts **reaction yield (%)** based on:
  - Temperature (°C)
  - Pressure (atm)
  - Catalyst Concentration (mol/L)
- 🧠 Trained a regression model using real experimental data
- 💾 Uses **MySQL** to store and query chemical data
- ⚙️ Built with Python, scikit-learn, Streamlit, and Pandas
- 🌐 Interactive interface using **Streamlit**

---

## 🧰 Tech Stack

| Layer              | Tools Used                                |
|--------------------|--------------------------------------------|
| 🧠 Machine Learning | scikit-learn, pandas, numpy               |
| 🧪 Data Storage     | MySQL (with Python connector)             |
| 📊 Frontend         | Streamlit (Python-based UI framework)     |
| 📁 Others           | Git, VS Code, Matplotlib, Seaborn         |

---

## 📂 Project Structure

chemical-yield-predictor/
│
├── main.py # Streamlit app
├── model_training.py # Model training and evaluation
├── yield_model.pkl # Saved regression model
├── init_experiments.sql # SQL file to create and populate the table
├── requirements.txt # Python dependencies
├── README.md # This file
└── chemical_data.csv # Dataset used

yaml
Copy code

---

## ⚙️ How It Works

1. Experimental data is stored in a MySQL table called `experiments`
2. Python reads the data using `mysql-connector-python` and `pandas`
3. A regression model is trained using:
   - Features: temperature, pressure, catalyst concentration
   - Target: yield_percent
4. The trained model is saved using `joblib`
5. The Streamlit app loads the model and provides sliders for live prediction

---

## 📥 Dataset Source

- Real experimental dataset from: [Kaggle - Chemical Yield Dataset](https://www.kaggle.com/datasets/ayushbarnawal/chemical-yield)

---


## 🖥️ Run Locally

### 1. Clone this repository

```bash
git clone https://github.com/hhemantpatel/chemical-yield-predictor.git
cd chemical-yield-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up MySQL

- Create a database (e.g., `chemical_data`)
- Run the SQL file to create and populate the table:

```sql
source init_experiments.sql;
```

### 4. Train the model

```bash
python model_training.py
```

### 5. Run the Streamlit app

```bash
streamlit run main.py
```

---

👨‍💻 Author
Hemant Patel  
B.Tech Chemical Engineering, IIT Jodhpur  
GitHub: @hhemantpatel  
Email: patelhemant.7509@gmail.com


