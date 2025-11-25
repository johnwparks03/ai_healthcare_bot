'''This script:
Gets questions from database, lives on AWS
Feeds them into the LLM
Uses x test to see how close the LLM's (medalpaca-7b or our model) answer is to the actual answer
Prints out the results'''

"""1- Get questions from database, lives on AWS"""

import psycopg2

conn = None
cur = None

AmtOfQuestions = int(input("How many questions do you want? ")) #I think we can trust ourselves not to need error handeling here

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
    QA = cur.fetchall()

    print()
    for i, row in enumerate(QA, 1):
        print(f"Q{i}: {row[0]}")
        print(f"A{i}: {row[1]}\n")
    print(QA)

except psycopg2.Error as e:
    print(f"Error connecting to PostgreSQL: {e}")

finally:
    if cur is not None:
        cur.close()
    if conn is not None:
        conn.close()

"""2- Feed questions into LLM"""

