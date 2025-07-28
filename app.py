import streamlit as st
import pandas as pd
import torch
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

from model import HybridRecommender
from sklearn.metrics.pairwise import cosine_similarity

# Load and deduplicate metadata (keep only one row per asin for dropdown)
metadata = pd.read_pickle("product_feature.pkl")

# Convert features to tensors
img_tensor = torch.tensor(np.array(metadata['image_feature'].tolist())).float()
text_tensor = torch.tensor(np.array(metadata['text_feature'].tolist())).float()

# Load model
model = HybridRecommender(fusion_type='attention')
model.eval()

# Get fused features
with torch.no_grad():
    fused_features = model(img_tensor, text_tensor).numpy()

# Helper to fetch image from URL
def fetch_image(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except:
        return None

# Recommendation logic
def recommend_products(product_index, fused_features, metadata, top_k=5):
    query_asin = metadata.iloc[product_index]['asin']
    
    # Compute cosine similarity
    similarities = cosine_similarity([fused_features[product_index]], fused_features)[0] #type: ignore
    
    # Attach similarity scores
    metadata_with_scores = metadata.copy()
    metadata_with_scores['similarity'] = similarities

    # Exclude query product
    metadata_with_scores = metadata_with_scores[metadata_with_scores['asin'] != query_asin]

    # Keep highest similarity per ASIN
    idx = metadata_with_scores.groupby('asin')['similarity'].idxmax()
    unique_recs = metadata_with_scores.loc[idx]

    # Return top-k
    top_recs = unique_recs.sort_values(by='similarity', ascending=False).head(top_k)

    return top_recs[['asin', 'title', 'image_url', 'review', 'similarity']].reset_index(drop=True)

# Set Streamlit layout
st.set_page_config(page_title="Product Recommender", layout="wide")
st.title("🛍️ Multi-Modal Product Recommender")

# Get unique products for dropdown
unique_products = metadata.drop_duplicates(subset='asin')[['asin', 'title']].reset_index(drop=True)
dropdown_titles = unique_products['title'].tolist()

# User selection
selected_title = st.selectbox("Choose a product:", dropdown_titles)

# Get selected ASIN and index
selected_asin = unique_products[unique_products['title'] == selected_title]['asin'].values[0]
selected_index = metadata[metadata['asin'] == selected_asin].index[0]

# Display selected product
st.subheader("🎯 Selected Product")
col1, col2 = st.columns([1, 4])
with col1:
    img = fetch_image(metadata.loc[selected_index, 'image_url'])
    if img:
        st.image(img, width=150)
with col2:
    st.markdown(f"**{selected_title}**")
    st.markdown("**Reviews:**")
    reviews = metadata[metadata['asin'] == selected_asin]['review'].dropna().unique()
    for review in reviews:
        st.markdown(f"- {review}")

# Show recommendations
st.subheader("🔎 Top 5 Similar Products")
recommendations = recommend_products(selected_index, fused_features, metadata)

for _, row in recommendations.iterrows():
    col1, col2 = st.columns([1, 4])
    with col1:
        rec_img = fetch_image(row['image_url'])
        if rec_img:
            st.image(rec_img, width=150)
    with col2:
        st.markdown(f"**{row['title']}**")
        st.markdown(f"Similarity: {row['similarity']:.2f}")
        st.markdown("**Reviews:**")
        st.markdown(f"- {row['review']}")
    st.markdown("---")
