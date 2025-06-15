from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv
import os


load_dotenv()

model = ChatGroq(
    model_name="llama3-8b-8192",  # You can use llama3-70b-8192 or mixtral-8x7b too
    temperature=0.4
)


# 2. Define the prompt with system, chat history, and user input
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant in health insurance domain. If any question is asked outside of this domain don't say anything but 'I can answer only from health or hospital domain.Nothing else."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

# 3. Start chat history
chat_history = []

# 4. Run the interactive loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # 5. Build the full prompt dynamically using history + current input
    full_prompt = prompt_template.invoke({
        "chat_history": chat_history,
        "user_input": user_input
    })

    # 6. Send prompt to model
    response = model.invoke(full_prompt)

    # 7. Update chat history with user and AI messages
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response.content))

    print("AI:", response.content)

# 8. Print final chat history (optional)
print("\nConversation History:")
for msg in chat_history:
    print(f"{msg.type.capitalize()}: {msg.content}")

