import mysql.connector
import pandas as pd

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

print(df.head())

# Optional: Close the connection
conn.close()
