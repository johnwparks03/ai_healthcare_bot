import psycopg2

conn = None
cur = None

AmtOfQuestions = int(input("How many questions do you want? "))

try:
    conn = psycopg2.connect(
        host="database-1.chsyuesiimuq.us-east-2.rds.amazonaws.com",
        database="postgres",
        user="postgres",
        password="abc12345",
        port="5432"
    )

    cur = conn.cursor()

    cur.execute(
        'SELECT * FROM public."QuestionAnswer" ORDER BY RANDOM() LIMIT %s',
        (AmtOfQuestions,)
    )
    questions = cur.fetchall()

    print()
    for i, row in enumerate(questions, 1):
        print(f"Q{i}: {row[0]}")
        print(f"A{i}: {row[1]}\n")

except psycopg2.Error as e:
    print(f"Error connecting to PostgreSQL: {e}")

finally:
    if cur is not None:
        cur.close()
    if conn is not None:
        conn.close()