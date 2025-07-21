from matplotlib import pyplot as plt
import mysql.connector
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn as sk
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import joblib

# Step 1: Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",        
    user="root",             
    password="Patel.7509",
    database="experiments" 
)

# Step 2: Query data into a Pandas DataFrame
query = "SELECT * FROM data"
df = pd.read_sql(query, conn)

# Features and Target
x_raw = df[['temperature', 'pressure', 'catalyst_concentration']]
y = df["yeild"]

# Scaling
scaler = StandardScaler()
x = scaler.fit_transform(x_raw)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ====================== KNN Regressor ======================
knn = KNeighborsRegressor(n_neighbors=20)
knn.fit(x_train, y_train)
p1 = knn.predict(x_test)
mse_knn = mean_squared_error(y_test, p1)
print("\n📘 KNN Regressor MSE:", mse_knn)

# ====================== Best SVR Model ======================
svr = SVR(kernel='rbf', C=10, gamma='scale', epsilon=1)
svr.fit(x_train, y_train)
pred1 = svr.predict(x_test)
mse_svr = mean_squared_error(y_test, pred1)
print("\n📗 SVR (Best Params) MSE:", mse_svr)

# ====================== Neural Network with GridSearchCV ======================
param_grid = {
    'hidden_layer_sizes': [(64, 32), (100, 50), (128, 64)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001],
    'learning_rate_init': [0.001, 0.005],
    'max_iter': [2000]
}

mlp = MLPRegressor(activation='tanh',alpha=0.001,hidden_layer_sizes=(128,64),learning_rate_init=0.001,max_iter=2000,solver='adam',random_state=42)
mlp.fit(x_train, y_train)

# Evaluate best neural net
pred_mlp = mlp.predict(x_test)
mse_mlp = mean_squared_error(y_test, pred_mlp)
print("\n📕 Neural Network MSE (Best Grid Search):", mse_mlp)



# Save KNN model
joblib.dump(knn, "knn_model.pkl")

# Save SVR model (with best params)
joblib.dump(svr, "svr_model.pkl")

# Save Neural Network (with best params)
joblib.dump(mlp, "neural_net_model.pkl")

print("✅ Models saved successfully!")


# Close DB connection
conn.close()
