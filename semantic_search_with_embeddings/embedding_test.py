from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# print(embedding_model)

health_insurance_docs = [
    "Health insurance helps cover the cost of medical services, including doctor visits, hospital stays, and prescription medications.",
    "Most health insurance plans have a deductible, which is the amount you pay before your insurance starts to cover costs.",
    "A copayment is a fixed fee you pay for a covered service, usually due at the time of service.",
    "Preventive services like annual checkups and vaccines are typically covered without a copay.",
    "HMO plans require members to choose a primary care provider and get referrals to see specialists.",
    "PPO plans offer more flexibility in choosing doctors and do not require referrals for specialists.",
    "Pre-existing conditions are typically covered under ACA-compliant plans.",
    "Open enrollment is the period during which you can sign up for or change your health insurance plan."
]

queries = [
    "What is health insurance and why is it important?",
    "How does a health insurance plan help pay for medical expenses?",
    "What is a deductible in health insurance?",
    "How are copayments different from deductibles?",
    "Do I have to pay anything for preventive care services?",
    "What’s the difference between HMO and PPO health plans?",
    "Do PPO plans require referrals to see specialists?",
    "Can I choose my own doctor with an HMO?",
    "Are pre-existing conditions covered by all health insurance plans?",
    "What does preventive care typically include?",
    "What is the open enrollment period?",
    "Can I apply for health insurance outside of open enrollment?",
    "If I visit a specialist without a referral, will my insurance still pay?",
    "How do I know if a specific treatment is covered by my plan?",
    "What happens if I don’t meet my deductible in a year?"
]

document_embeddings =  embedding_model.embed_documents(health_insurance_docs)


for query in queries:
    query_embedding = embedding_model.embed_query(query)
    score = cosine_similarity([query_embedding], document_embeddings)[0]
    idx, score = sorted(enumerate(score), key=lambda x: x[1], reverse=True)[0]
    print(f"\nQuery: {query}")
    print(f"  -> Doc {idx}: {health_insurance_docs[idx]} (Score: {score:.3f})")


