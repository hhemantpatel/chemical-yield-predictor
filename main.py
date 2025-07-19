import mysql.connector
import pandas as pd

# Step 1: Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",        # or "127.0.0.1"
    user="root",             # your MySQL username
    password="Patel.7509",# your MySQL password
    database="class" # the database you created
)

# Step 2: Query data into a Pandas DataFrame
query = "SELECT * FROM dat"
df = pd.read_sql(query, conn)

print(df.head())

# Optional: Close the connection
conn.close()
