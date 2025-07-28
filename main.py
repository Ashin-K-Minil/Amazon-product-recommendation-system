import torch
import numpy as np
import pandas as pd
from model import HybridRecommender
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

from captum.attr import IntegratedGradients
from lime.lime_text import LimeTextExplainer 
from transformers import DistilBertTokenizer, DistilBertModel

metadata = pd.read_pickle("product_feature.pkl")

print(f"Review @ index 100:\n{metadata.loc[100, 'review']}")
print(f"Type: {type(metadata.loc[100, 'review'])}")

model = HybridRecommender(fusion_type='attention')
model.eval()

# Convert features to tensors
img_tensor = torch.tensor(metadata['image_feature']).float()
text_tensor = torch.tensor(metadata['text_feature']).float()

# Get fused features
with torch.no_grad():
    fused_features = model(img_tensor, text_tensor).numpy()

def recommend_products(product_index, fused_features, metadata, top_k=5):
    query_asin = metadata.iloc[product_index]['asin']
    
    # Compute cosine similarity scores
    similarities = cosine_similarity([fused_features[product_index]], fused_features)[0] # type: ignore
    
    # Attach similarity scores to metadata
    metadata_with_scores = metadata.copy()
    metadata_with_scores['similarity'] = similarities

    # Exclude the query product itself
    metadata_with_scores = metadata_with_scores[metadata_with_scores['asin'] != query_asin]

    # Keep only one row per ASIN (highest similarity)
    idx = metadata_with_scores.groupby('asin')['similarity'].idxmax()
    unique_recs = metadata_with_scores.loc[idx]

    # Sort and return top_k recommendations
    top_recs = unique_recs.sort_values(by='similarity', ascending=False).head(top_k)

    return top_recs[['asin', 'title', 'image_url', 'similarity']].reset_index(drop=True)

def explain_image(index, model, img_tensor, text_tensor):
    input_img = img_tensor[index].unsqueeze(0)
    input_text = text_tensor[index].unsqueeze(0)

    def forward_func(input_img):
        return model(input_img, input_text)
    
    ig = IntegratedGradients(forward_func)
    attributions, _ = ig.attribute(input_img, target=0, return_convergence_delta=True)
    return attributions.squeeze().detach().numpy()

def explain_text(index, metadata, tokenizer, model):
    review_text = metadata.iloc[index]['review']

    class_names = ['irrelevant','relevant']
    explainer = LimeTextExplainer(class_names=class_names)

    def predict_prob(texts):
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            output = model(**inputs)
            cls_embed = output.last_hidden_state[:, 0, :]  # [batch_size, 768]

        # Cosine similarity to the first sample
        scores = F.cosine_similarity(cls_embed, cls_embed[0].unsqueeze(0), dim=1)  # [batch_size]

        # Stack relevance and irrelevance scores: [batch_size, 2]
        score_matrix = torch.stack([1 - scores, scores], dim=1)

        # Normalize to valid probabilities using softmax
        probs = F.softmax(score_matrix, dim=1)

        return probs.numpy()
    
    exp = explainer.explain_instance(review_text, predict_prob, num_features=10)
    return exp.as_list()

product_index = 100
print("Query Product:")
print(metadata.iloc[product_index][['title', 'image_url']])

print("Recommended Products:")
recommendations = recommend_products(product_index, fused_features, metadata, top_k=5)
print(recommendations)

top_rec_index = recommendations.index[0]  # relative index
true_index = recommendations.iloc[0].name  # absolute index in metadata

# Explain Image
image_attr = explain_image(true_index, model, img_tensor, text_tensor)
print("Captum Image Attribution Shape:", image_attr.shape)

# Explain Text
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
text_explanation = explain_text(product_index, metadata, tokenizer, bert_model)
print("LIME Text Explanation:")
print(text_explanation)

def plot_image_attributions(attributions, top_k=20):
    top_indices = np.argsort(attributions)[-top_k:][::-1]
    plt.figure(figsize=(12, 4))
    plt.bar(range(top_k), attributions[top_indices])
    plt.xticks(range(top_k), top_indices, rotation=45) # type: ignore
    plt.title(f"Top {top_k} Image Feature Attributions")
    plt.xlabel("Feature Index")
    plt.ylabel("Attribution")
    plt.tight_layout()
    plt.show()

plot_image_attributions(image_attr)