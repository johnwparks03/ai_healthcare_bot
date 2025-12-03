from sagemaker.huggingface import HuggingFacePredictor

predictor = HuggingFacePredictor(endpoint_name="RedoneMedBot")
conversation = []

def model_predict(user_input):

    # Build context string from history
    context = ""
    for msg in conversation:
        context += f"Q: {msg['q']}\nA: {msg['a']}\n\n"
    # Combine history + new question
    full_question = f"Q: {user_input}" if context else user_input
    print(full_question)

    response = predictor.predict({
        "question": full_question,
        "max_length": 256,
        "temperature": 0.7
    })

    answer = response.get("answer", response)
    print(answer)
    answer = answer.rstrip()[:-3]
    print(f"\nAssistant: {answer}")


    conversation.append({"q": user_input, "a": answer})

    return answer