# ===== main.py =====
from matplotlib import pyplot as plt
import mysql.connector
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import joblib

# Connect to MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Patel.7509",
    database="experiments"
)

# Fetch data
df = pd.read_sql("SELECT * FROM data", db)
x_raw = df[['temperature', 'pressure', 'catalyst_concentration']]
y = df['yeild']

# Scale and split
scaler = StandardScaler()
x = scaler.fit_transform(x_raw)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ----- KNN -----
knn = KNeighborsRegressor(n_neighbors=20)
knn.fit(x_train, y_train)
print("KNN MSE:", mean_squared_error(y_test, knn.predict(x_test)))

# ----- SVR -----
svr = SVR(kernel='rbf', C=10, gamma='scale', epsilon=1)
svr.fit(x_train, y_train)
print("SVR MSE:", mean_squared_error(y_test, svr.predict(x_test)))

# ----- Neural Net -----
nn = MLPRegressor(
    hidden_layer_sizes=(128, 64), activation='tanh', solver='adam',
    alpha=0.001, learning_rate_init=0.001, max_iter=2000, random_state=42)
nn.fit(x_train, y_train)
print("Neural Net MSE:", mean_squared_error(y_test, nn.predict(x_test)))

# Save models and scaler
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(svr, 'svr_model.pkl')
joblib.dump(nn, 'neural_net_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ All models and scaler saved!")
print("✅ Models saved at:", pd.Timestamp.now())

db.close()
