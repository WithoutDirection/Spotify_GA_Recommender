import mysql.connector
# 建立連線
conn = mysql.connector.connect(
    host="172.20.10.3",  # 本機 MySQL 伺服器
    user="Da",
    password="abc",
    port=3306,
    database="final_project"
)

# 建立游標
cursor = conn.cursor()

# 執行 SQL 查詢
cursor.execute("SELECT * FROM music_pool WHERE track_id LIKE '2tolmRzbUfgL5KRplIqHlu';")

# 獲取資料
result = cursor.fetchall()[0][0]
print(result)

# 關閉連線
cursor.close()
conn.close()