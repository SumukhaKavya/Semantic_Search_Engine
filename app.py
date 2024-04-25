import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load the embeddings DataFrame
df = pd.read_csv('E:\\search_engine\\data\\final.csv')

def preprocess(query):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(query)
    return embeddings


# Function to calculate similarity scores between query embedding and document embeddings
def calculate_similarity(query_embedding, document_embeddings):
    similarities = cosine_similarity(query_embedding, document_embeddings)
    return similarities.flatten()

# Function to display top 10 file names based on similarity scores
def display_top_files(df, similarity_scores):
    top_indices = np.argsort(similarity_scores)[::-1][:10]  
    st.subheader("Top 10 Matching Files:")
    for idx in top_indices:
        st.write(df.loc[idx, 'name'])

# Streamlit UI
st.title("Document Search Engine")

# User input for search query
user_query = st.text_input("Enter your search query:")

if user_query:
    # Preprocess the user query
    query_embedding = preprocess(user_query)

    # Extract document embeddings
    document_embeddings = np.array([np.fromstring(embedding[1:-1], dtype=np.float, sep=',') 
                                     for embedding in df['embedding_chunk']])

    # Calculate similarity scores
    similarity_scores = calculate_similarity(query_embedding, document_embeddings)

    # Display top 10 matching file names
    display_top_files(df, similarity_scores)
