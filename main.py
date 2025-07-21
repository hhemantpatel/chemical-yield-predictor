from matplotlib import pyplot as plt
import mysql.connector
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn as sk
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
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


x_raw = df[['temperature', 'pressure', 'catalyst_concentration']]
y=df["yeild"]
#z=df["Yield_Level"]
scaler=StandardScaler()
x=scaler.fit_transform(x_raw)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#x_train,x_test,z_train,z_test=train_test_split(x,z,test_size=0.2,random_state=42)
# Optional: Close the connection
knn=KNeighborsRegressor(n_neighbors=20)
#knn1=KNeighborsClassifier(n_neighbors=44)

knn.fit(x_train,y_train)
#knn1.fit(x_train,y_train)

p1=knn.predict(x_test)
#p2=knn1.predict(x_test)

#a=accuracy_score(z_test,p2)
b=mean_squared_error(y_test,p1)

#print(a)
print(b)



conn.close()
