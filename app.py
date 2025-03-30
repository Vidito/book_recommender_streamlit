import streamlit as st
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Load FAISS index
faiss_index = faiss.read_index("faiss_books.index")

# Load book metadata
with open("book_metadata.pkl", "rb") as f:
    book_metadata = pickle.load(f)  # List of dicts with "title", "description", etc.

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def recommend_books(user_query, top_n=5):
    # Convert query to embedding
    query_embedding = model.encode(user_query)
    query_embedding = np.array([query_embedding]).astype("float32")
    
    # Search FAISS index
    distances, indices = faiss_index.search(query_embedding, top_n)
    
    # Get book details
    recommendations = []
    for idx in indices[0]:  
        book = book_metadata[idx]
        recommendations.append({
            "title": book["title"],
            "description": book["description"],
            "author": book.get("author", "Unknown")
        })
    
    return recommendations

# Streamlit UI
st.title("Book Recommender System")
st.write("Describe a story you want to read, and we'll suggest the best books!")

user_input = st.text_area("Enter a story description:")

if st.button("Find Books") and user_input:
    results = recommend_books(user_input)
    
    if results:
        st.subheader("Recommended Books:")
        for book in results:
            st.markdown(f"### {book['title']}")
            st.write(f"**Author:** {book['author']}")
            st.write(f"{book['description']}")
            st.write("---")
    else:
        st.write("No matching books found. Try a different description!")
