import json
import gzip
import torch
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
from io import BytesIO
from torchvision import models, transforms

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
    
metadata = load_metadata("meta_Video_Games.json.gz")
reviews = load_reviews("Video_Games_5.json.gz",set(metadata['asin']))

metadata = metadata.sample(n=5000, random_state=42)
merged = pd.merge(metadata, reviews, on='asin', how='left')
reviews = merged[['asin','review']].copy()

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

np.save('image_features.npy', np.stack(metadata['image_feature'].values)) # type: ignore