import argparse
import os
from pathlib import Path
from io import BytesIO
import requests
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack, csr_matrix

# Import simple heuristics
from diet_classifiers import is_keto, is_vegan

IMAGE_DIR = Path('data/images')
EMBED_PATH = Path('data/image_embeddings.npy')


def download_images(df: pd.DataFrame) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for idx, url in df['photo_url'].items():
        file = IMAGE_DIR / f"{idx}.jpg"
        if file.exists() or not isinstance(url, str) or not url:
            continue
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            with open(file, 'wb') as f:
                f.write(resp.content)
        except Exception:
            # Skip failed downloads
            continue


def build_image_embeddings(df: pd.DataFrame, force: bool = False) -> np.ndarray:
    if EMBED_PATH.exists() and not force:
        return np.load(EMBED_PATH)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    embeddings = []
    for idx in df.index:
        img_file = IMAGE_DIR / f"{idx}.jpg"
        if not img_file.exists():
            embeddings.append(np.zeros(2048, dtype=np.float32))
            continue
        try:
            img = Image.open(img_file).convert('RGB')
            with torch.no_grad():
                tensor = preprocess(img).unsqueeze(0).to(device)
                emb = model(tensor).squeeze().cpu().numpy()
        except Exception:
            emb = np.zeros(2048, dtype=np.float32)
        embeddings.append(emb)

    arr = np.vstack(embeddings)
    np.save(EMBED_PATH, arr)
    return arr


def label_with_rules(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['label_keto'] = df['ingredients'].apply(is_keto).astype(int)
    df['label_vegan'] = df['ingredients'].apply(is_vegan).astype(int)
    return df


def vectorize_text(df: pd.DataFrame) -> tuple:
    texts = df['ingredients'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
    vec = TfidfVectorizer(min_df=2)
    X = vec.fit_transform(texts)
    return X, vec


def combine_features(X_text, X_image) -> csr_matrix:
    img_sparse = csr_matrix(X_image)
    return hstack([X_text, img_sparse])


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_res, y_res)
    pred = clf.predict(X_test)
    print(classification_report(y_test, pred))


def main():
    parser = argparse.ArgumentParser(description='Hybrid text+image diet classifier')
    parser.add_argument('--data', default='silver.csv', help='CSV with ingredients, photo_url columns')
    # Default to hybrid mode (text + image) =======
    parser.add_argument('--mode', choices=['text', 'image', 'both'], default='both')
    parser.add_argument('--force', action='store_true', help='Force recompute image embeddings')
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = label_with_rules(df)

    if args.mode in {'both', 'image'}:
        download_images(df)
        img_embs = build_image_embeddings(df, force=args.force)
    else:
        img_embs = None

    if args.mode in {'both', 'text'}:
        X_text, _ = vectorize_text(df)
    else:
        X_text = None

    if args.mode == 'text':
        X = X_text
    elif args.mode == 'image':
        X = csr_matrix(img_embs)
    else:
        X = combine_features(X_text, img_embs)

    print('=== Keto ===')
    train_model(X, df['label_keto'])
    print('=== Vegan ===')
    train_model(X, df['label_vegan'])


if __name__ == '__main__':
    main()
