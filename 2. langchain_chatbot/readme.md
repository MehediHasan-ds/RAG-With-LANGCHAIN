# Building a Domain-Specific Chatbot Using LangChain & Groq

*By Mehedi Hasan*

---

## **Project Overview & Setup**

### Objective

The goal of this project was to build a **conversational AI chatbot** that specializes in **health insurance**. I wanted it to:

* Understand user queries in natural language
* Respond only within the health domain
* Maintain conversation context
* Be fast and lightweight

---

### Tools I Used

| Tool                 | Why I Used It                                 |
| -------------------- | --------------------------------------------- |
| `LangChain`          | Gives modular structure and prompt management |
| `Groq` (LLaMA-3)     | Fast inference, cheaper than OpenAI           |
| `dotenv`             | Load API keys securely                        |
| `ChatPromptTemplate` | Create dynamic prompts with memory            |

---

### Challenge 1: Create a smart, focused chatbot

🔹 **Problem:** LLMs tend to answer everything — not ideal for domain-specific support.
🔹 **Solution:** I defined a clear **system message** that sets the assistant’s boundaries.

```python
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant in health insurance domain. If any question is asked outside of this domain don't say anything but 'I can answer only from health or hospital domain. Nothing else."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])
```

**How this helps:**

* Prevents the model from going off-topic
* Creates domain focus from the first message
* Reduces hallucinations from general knowledge

---

### Challenge 2: Track conversation state

🔹 **Problem:** A real chatbot needs memory — it should respond in context.
🔹 **Solution:** I use `chat_history` to store message objects (user + AI).

```python
chat_history = []
```

**How this helps:**

* Stores ongoing dialogue as message objects
* Feeds this history back into the prompt
* Makes the assistant feel more human and coherent

---

## **Real-Time Interaction with the Model**

### Challenge 3: Handle input and build prompts dynamically

🔹 **Problem:** Every new message needs to include the **chat history** so the model responds appropriately.
🔹 **Solution:** I use `ChatPromptTemplate.invoke()` to merge the history and input.

```python
full_prompt = prompt_template.invoke({
    "chat_history": chat_history,
    "user_input": user_input
})
```

**How this helps:**

* Combines past messages + current query into a full prompt
* Gives the model context to avoid repeated explanations
* Makes replies more relevant and grounded

---

### Challenge 4: Connect with a model that's fast and cost-effective

🔹 **Problem:** OpenAI models are powerful but expensive and rate-limited.
🔹 **Solution:** I switched to `ChatGroq` with LLaMA 3 for local-like speed and zero cost per token.

```python
model = ChatGroq(
    model_name="llama3-8b-8192",
    temperature=0.4
)
```

**How this helps:**

* Groq inference is fast and scalable
* Lower temperature improves factuality
* No billing surprises like with OpenAI or Anthropic

---

### Challenge 5: Generate and save the AI’s reply

🔹 **Problem:** The model returns a response, but I also need to track and reuse it.
🔹 **Solution:** I append both the user message and AI reply to the chat history.

```python
chat_history.append(HumanMessage(content=user_input))
chat_history.append(AIMessage(content=response.content))
```

**How this helps:**

* Maintains full chat trace
* Enables long conversations
* Could be saved to file later for persistent memory

---

### Full Loop Summary

```python
while True:
    user_input = input("You: ")
    ...
    response = model.invoke(full_prompt)
    print("AI:", response.content)
```

This loop enables live, intelligent conversation — just like a customer service chatbot — but with memory and guardrails.

---

## ❌📄 **Limitations & What I Learned**

### Limitation 1: Weak Domain Control

Even though I set a clear system message, LLaMA sometimes still replies to unrelated queries.
Why? Because LLMs **aren’t rule-based** — they follow probability, not hard logic.

*Future Fix:* Add a domain intent classifier (like `scikit-learn` or `fasttext`) to block off-topic inputs before invoking the LLM.

---

### Limitation 2: Memory Lost After Exit

Currently, `chat_history` is stored in memory only. When I quit the program, it’s gone.
*Future Fix:* Save `chat_history` to a file (`chat_history.json`) and load it back.

---

### Limitation 3: Token Limits

LLaMA 3 8B has an 8,192-token context limit. Long chats will eventually break or lose older messages.

*Future Fix:* Add summarization for old history (e.g., "Summarize last 10 messages").

---

### Limitation 4: No Personalization

All users talk to the same AI. It doesn’t know names, plans, or past questions.
*Future Fix:* Connect a simple user login + database (e.g., SQLite) and pass user data into the prompt.

---

### Limitation 5: No Web UI or Streaming

Right now, it’s terminal-only and responses are printed after full generation.
*Future Fix:* Add a **Streamlit interface** and `model.stream()` for a smoother chat experience.

---

## Conclusion

This project gave me hands-on experience with:

* LLM prompt management using LangChain
* Fast and efficient deployment using Groq models
* Chat memory and message handling
* Structuring AI assistants to be task-specific and restricted

**Next steps:**

* Add RAG (retrieval from insurance PDFs)
* Save chat history to disk
* Integrate with claim data for real-time answers

---
