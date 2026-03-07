# scripts/fordham_rag_app.py

import os
from dotenv import load_dotenv
import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ----------------------------------------------------------------------------
# USED LLM TO GENERATE MOST OF THIS CODE
# ----------------------------------------------------------------------------
# Load .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY not found in environment. Check your .env file.")

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Load saved artifacts
# -----------------------------
embeddings = np.load("temp/embeddings.npy")  # precomputed embeddings
df_chunks = pd.read_json("temp/chunks_metadata.json")  # chunk text + filename
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI()  # make sure OPENAI_API_KEY is set in env

# -----------------------------
# Helper functions
# -----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(question, k=5):
    """Return top-k relevant chunks for a question"""
    q_embedding = embedding_model.encode(question)
    similarities = [cosine_similarity(q_embedding, emb) for emb in embeddings]
    similarities = np.array(similarities)
    top_k_idx = similarities.argsort()[-k:][::-1]
    return df_chunks.iloc[top_k_idx]

def generate_answer(question, retrieved_chunks):
    """Call LLM to generate answer from retrieved chunks"""
    context = "\n\n".join(retrieved_chunks["chunk"].tolist())
    prompt = f"""
You are a helpful assistant answering questions about Fordham University.

Use ONLY the information in the context below.
If the answer is not in the context, say:
"I don't have enough information from the provided sources."

Context:
{context}

Question:
{question}

Answer:
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def rag(question, k=5):
    """Full RAG pipeline: retrieve + generate"""
    retrieved_chunks = retrieve(question, k=k)
    answer = generate_answer(question, retrieved_chunks)
    return answer, retrieved_chunks

# -----------------------------
# Streamlit app
# -----------------------------
st.title("Fordham University RAG Assistant")
st.write("Ask a question about Fordham University and get a grounded answer.")

question = st.text_input("Type your question here:")

if question:
    answer, retrieved_chunks = rag(question)
    st.write("### Answer")
    st.write(answer)
    
    st.write("### Sources")
    for url in retrieved_chunks["filename"].unique():
        st.write(url)