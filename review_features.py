import json
import gzip
import torch
import requests
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from torchvision import models, transforms
from transformers import DistilBertTokenizer, DistilBertModel

def parse_gz(path):
    with gzip.open(path, 'rb') as f:
        for line in f:
            yield json.loads(line)

def load_metadata(meta_path):
    data = []
    for item in parse_gz(meta_path):
        if 'imageURLHighRes' in item and item['imageURLHighRes']:
            data.append({
                'asin': item['asin'],
                'title': item.get('title', ''),
                'image_url': item['imageURLHighRes'][0]
            })
    return pd.DataFrame(data)

def load_reviews(review_path, valid_asins):
    data = []
    for r in parse_gz(review_path):
        if r['asin'] in valid_asins and 'reviewText' in r:
            data.append({
                'asin': r['asin'],
                'review': r['reviewText']
            })
    return pd.DataFrame(data)

def fetch_image(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except:
        return None

# Load metadata and reviews
metadata = load_metadata("meta_Video_Games.json.gz")
reviews = load_reviews("Video_Games_5.json.gz", set(metadata['asin']))

metadata = metadata.sample(n=5000, random_state=42)

resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Identity() # type: ignore
resnet.eval()

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@torch.no_grad()
def extract_img_feature(url, model, preprocess):
    try:
        response = requests.get(url,timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        tensor = preprocess(img).unsqueeze(0)
        feature = model(tensor)
        return feature.squeeze(0).numpy()
    except:
        return None
    
metadata['image_feature'] = None

for i, row in tqdm(metadata.iterrows(), total=len(metadata)):
    feature = extract_img_feature(row['image_url'], resnet, preprocess)
    metadata.at[i, 'image_feature'] = feature

metadata = metadata[metadata['image_feature'].notnull()]

merged = pd.merge(metadata, reviews, on='asin', how='left')

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
bert_model.eval() # Non-training mode

# Feature extraction with zero vector fallback
zero_review_feat = np.zeros(768)

@torch.no_grad()
def extract_review_features(text, tokenizer, model, max_length=128):
    if isinstance(text, str) and text.strip():
        input = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_length)
        output = model(**input)
        cls_embedding = output.last_hidden_state[:, 0, :]
        return cls_embedding.squeeze(0).numpy()
    else:
        return zero_review_feat

# Apply feature extraction to all items, even those with no review (cold start)
merged["text_feature"] = None
for i, row in tqdm(merged.iterrows(), total=len(merged)):
    feature = extract_review_features(row["review"], tokenizer, bert_model)
    merged.at[i, "text_feature"] = feature

merged.to_pickle("product_feature.pkl")

# Save review features
# np.save("review_features.npy", np.stack(merged["text_feature"].values)) # type: ignore