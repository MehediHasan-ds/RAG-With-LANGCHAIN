# 📄 **Understanding Embeddings and Semantic Similarity in LangChain**

### Overview

This report explains how **text embeddings** are used to find semantically similar documents based on user queries. Using **LangChain** with **Hugging Face Embeddings** (`sentence-transformers/all-MiniLM-L6-v2`), we compute vector representations of health insurance-related documents and compare them with vectorized queries using **cosine similarity**.

---

## Step-by-Step Code Walkthrough

### 🔹 1. **Setup and Imports**

```python
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()
```

* **`HuggingFaceEmbeddings`**: Loads a local embedding model that converts text into numerical vectors.
* **`cosine_similarity`**: Computes similarity between the query and each document vector.
* **`load_dotenv()`**: Loads environment variables (useful for storing API keys securely if needed).

---

### 🔹 2. **Initialize Embedding Model**

```python
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

* We use **MiniLM-L6-v2**, a small but high-performing model trained to capture semantic meaning.
* This model converts text into **384-dimensional embeddings**.

---

### 🔹 3. **Define Document Corpus**

```python
health_insurance_docs = [
    "Health insurance helps cover the cost of medical services...",
    ...
]
```

* A list of 8 short health insurance-related texts.
* These are the documents we want to search against.

---

### 🔹 4. **Define User Queries**

```python
queries = [
    "What is health insurance and why is it important?",
    ...
]
```

* 15 realistic questions a user might ask.
* We want to retrieve the most relevant document for each.

---

### 🔹 5. **Generate Document Embeddings**

```python
document_embeddings = embedding_model.embed_documents(health_insurance_docs)
```

* Converts all documents into embeddings (vectors).
* Each document becomes a 384-dimensional numeric representation capturing its **semantic meaning**.

---

## Query Processing and Semantic Matching

### 🔹 6. **Embed Query and Compare to Documents**

```python
for query in queries:
    query_embedding = embedding_model.embed_query(query)
    score = cosine_similarity([query_embedding], document_embeddings)[0]
```

* Each query is converted to an embedding vector.
* Then, **cosine similarity** is computed between this query vector and all document vectors.

---

### 🔹 7. **Find the Most Relevant Document**

```python
    idx, score = sorted(enumerate(score), key=lambda x: x[1], reverse=True)[0]
```

* `enumerate(score)` gives each document's index and similarity score.
* We sort them in descending order and select the one with the highest score — the most relevant match.

---

### 🔹 8. **Print the Match**

```python
    print(f"\nQuery: {query}")
    print(f"  -> Doc {idx}: {health_insurance_docs[idx]} (Score: {score:.3f})")
```

* The code prints the query and its top-matched document along with the similarity score.

---

## Summary

* **Embeddings** convert both documents and queries into dense numeric vectors.
* **Cosine similarity** identifies which document is **semantically closest** to the user's query — even if the words differ.
* This setup enables **intelligent information retrieval**, which is a foundation for search engines, chatbots, and RAG (retrieval-augmented generation) systems.

---

## Final Notes

* The `sentence-transformers/all-MiniLM-L6-v2` model is a great default — fast and effective.
* This pipeline will be scaled with vector databases (e.g., FAISS, Chroma) for larger corpora.
* Then I will also integrate this with an LLM (like LLaMA via Groq) to generate answers from matched content.

---


## Project Setup & Run Instructions

This section explains how to set up and run this project locally.

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/MehediHasan-ds/RAG-With-LANGCHAIN.git
cd RAG-With-LANGCHAIN
```

---

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install manually:

```bash
pip install langchain langchain-huggingface sentence-transformers scikit-learn python-dotenv
```

---

### Step 3: Create `.env` File

In the project root, create a file named `.env` and include any necessary variables:

```
# .env
OPENAI_API_KEY = "openai-api-key"
HUGGINGFACE_API_KEY = "huggingface-api-key"
```

If you later add other APIs, you'll store their keys here.

---

### Step 4: Run the Script

```bash
python embedding_test.py
```

