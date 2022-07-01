import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="karan",
  password=""
)

print(mydb)

mycursor = mydb.cursor()

mycursor.execute("CREATE DATABASE karan")