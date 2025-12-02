from sagemaker.huggingface import HuggingFacePredictor

predictor = HuggingFacePredictor(endpoint_name="RedoneMedBot")
conversation = []

while True:
    UserPrompt = input("\nYou: ").strip()

    if not UserPrompt or UserPrompt.lower() == 'quit':
        break

    # Build context string from history
    context = ""
    for msg in conversation:
        context += f"Q: {msg['q']}\nA: {msg['a']}\n\n"

    # Combine history + new question
    full_question = f"{context}Q: {UserPrompt}" if context else UserPrompt

    response = predictor.predict({
        "question": full_question,
        "max_length": 256,
        "temperature": 0.7
    })

    answer = response.get("answer", response)
    print(f"\nAssistant: {answer}")

    conversation.append({"q": UserPrompt, "a": answer})