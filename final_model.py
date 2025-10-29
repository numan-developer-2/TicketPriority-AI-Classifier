"""
train_and_test_priority.py
Interactive, beginner-friendly pipeline for Ticket Priority classification.
- Supports TF-IDF or Sentence-Transformer embeddings (recommended).
- Balances classes (SMOTE for embeddings, RandomOverSampler for TF-IDF).
- Trains DecisionTree and RandomForest.
- Evaluates and saves models.
- Interactive CLI to test new tickets.
"""

import os
import glob
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Balancing methods
from imblearn.over_sampling import SMOTE, RandomOverSampler

# Optional embeddings (fast & small)
USE_EMBEDDINGS = True           # << set to True to use SentenceTransformer embeddings (recommended)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # sentence-transformers small model

# Config: change these to fit your environment
DATA_DIR = r"D:/AI(Python)/Unsupervisdmodel"   # <- set your folder with CSVs
OUTPUT_DIR = "./models"
SAMPLE_MAX_ROWS = 12000   # reduce for speed (keeps stratified proportions)
RANDOM_STATE = 42

# RandomForest hyperparams
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 20

# ------------------------------
# Helper: safe import for embeddings
# ------------------------------
embedder = None
if USE_EMBEDDINGS:
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        print("Loading embedding model (this may take a few seconds)...")
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded:", EMBEDDING_MODEL_NAME)
    except Exception as e:
        print("ERROR: sentence-transformers not available or failed to load.")
        print("Install with: pip install sentence-transformers")
        print("Falling back to TF-IDF (set USE_EMBEDDINGS=False if you can't install).")
        USE_EMBEDDINGS = False

# ------------------------------
# Step A: Load and combine CSV files
# ------------------------------
def load_csvs(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if len(files) == 0:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    frames = []
    for f in files:
        print("Loading:", f)
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception:
            df = pd.read_csv(f, low_memory=False, encoding="latin1")
        df["source_file"] = os.path.basename(f)
        # ensure text columns exist
        if "subject" not in df.columns:
            df["subject"] = ""
        if "body" not in df.columns:
            alt = next((c for c in ['message','content','text','description','answer'] if c in df.columns), None)
            df["body"] = df[alt].astype(str) if alt else ""
        if "priority" not in df.columns:
            df["priority"] = np.nan
        frames.append(df)
    master = pd.concat(frames, ignore_index=True, sort=False)
    print("Combined master shape:", master.shape)
    return master

# ------------------------------
# Step B: Preprocess text & labels
# ------------------------------
def preprocess(master):
    # create text
    master["subject"] = master["subject"].fillna("").astype(str)
    master["body"] = master["body"].fillna("").astype(str)
    master["text"] = (master["subject"] + " " + master["body"]).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    # mask simple PII
    master["text"] = master["text"].str.replace(r"\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b", " [email] ", regex=True)
    master["text"] = master["text"].str.replace(r"\b\d{7,}\b", " [number] ", regex=True)
    # keep rows with text and priority
    usable = master[(master["text"].str.strip() != "") & (master["priority"].notna())].copy()
    print("Usable rows:", len(usable))
    # normalize raw priority to high/medium/low
    def map_priority(p):
        p = str(p).lower()
        if "high" in p or "urgent" in p or "critical" in p:
            return "high"
        if "med" in p or "normal" in p or "medium" in p:
            return "medium"
        if "low" in p or "minor" in p:
            return "low"
        if p.startswith("p1"):
            return "high"
        if p.startswith("p2"):
            return "medium"
        if p.startswith("p3") or p.startswith("p4"):
            return "low"
        return "other"
    usable["priority_mapped"] = usable["priority"].apply(map_priority)
    print("Mapped counts (before drop):")
    print(usable["priority_mapped"].value_counts())
    # drop tiny 'other' if present
    other_count = (usable["priority_mapped"] == "other").sum()
    if other_count > 0 and other_count < 0.03 * len(usable):
        usable = usable[usable["priority_mapped"] != "other"]
        print("Dropped 'other' labels (very small).")
    return usable

# ------------------------------
# Step C: Optional sampling for speed
# ------------------------------
def stratified_sample(df, max_rows=SAMPLE_MAX_ROWS):
    if len(df) <= max_rows:
        return df
    print(f"Sampling ~{max_rows} rows stratified by label for speed...")
    sampled = df.groupby("priority_mapped", group_keys=False).apply(
        lambda g: g.sample(int(max_rows * len(g) / len(df)), random_state=RANDOM_STATE)
    ).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print("After sampling:", len(sampled))
    return sampled

# ------------------------------
# Step D: Vectorize (TF-IDF or embeddings)
# ------------------------------
def vectorize_text(X_train_texts, X_test_texts, use_embeddings=USE_EMBEDDINGS):
    if use_embeddings and embedder is not None:
        print("Computing SentenceTransformer embeddings (train)...")
        X_train_emb = embedder.encode(list(X_train_texts), show_progress_bar=True)
        X_test_emb = embedder.encode(list(X_test_texts), show_progress_bar=True)
        vectorizer_obj = {"type": "embeddings", "model_name": EMBEDDING_MODEL_NAME}
        return np.array(X_train_emb), np.array(X_test_emb), vectorizer_obj
    else:
        print("Computing TF-IDF vectors (train)...")
        tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1,2))
        X_train_tfidf = tfidf.fit_transform(X_train_texts)
        X_test_tfidf = tfidf.transform(X_test_texts)
        vectorizer_obj = {"type": "tfidf", "object": tfidf}
        return X_train_tfidf, X_test_tfidf, vectorizer_obj

# ------------------------------
# Step E: Balance training data
# ------------------------------
def balance_training(X_train_vec, y_train, vectorizer_meta):
    print("Balancing training set...")
    # If using sparse TF-IDF -> avoid SMOTE (memory/time). Use RandomOverSampler.
    if vectorizer_meta["type"] == "tfidf":
        print("Using RandomOverSampler for TF-IDF (safe for high-dim).")
        ros = RandomOverSampler(random_state=RANDOM_STATE)
        X_dense = X_train_vec if not hasattr(X_train_vec, "toarray") else X_train_vec.toarray()
        X_res, y_res = ros.fit_resample(X_dense, y_train)
        return X_res, y_res
    else:
        # embeddings (dense) -> OK to use SMOTE
        print("Using SMOTE for embeddings.")
        sm = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = sm.fit_resample(X_train_vec, y_train)
        return X_res, y_res

# ------------------------------
# Step F: Train models
# ------------------------------
def train_models(X_train_vec_bal, y_train_bal):
    print("Training Decision Tree (baseline)...")
    dt = DecisionTreeClassifier(max_depth=20, random_state=RANDOM_STATE)
    dt.fit(X_train_vec_bal, y_train_bal)
    print("Decision Tree done.")
    print("Training Random Forest (recommended)...")
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train_vec_bal, y_train_bal)
    print("Random Forest done.")
    return dt, rf

# ------------------------------
# Step G: Evaluate
# ------------------------------
def evaluate_and_print(model, X_test_vec, y_test, label="Model"):
    print(f"\n=== Evaluation: {label} ===")
    # If X_test_vec is sparse with .toarray available -> convert for predict if model expects dense
    X_for_pred = X_test_vec if not hasattr(X_test_vec, "toarray") else X_test_vec.toarray()
    preds = model.predict(X_for_pred)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))
    cm = confusion_matrix(y_test, preds, labels=["high","medium","low"])
    print("Confusion matrix (actual rows, predicted cols):")
    print(pd.DataFrame(cm, index=["actual_high","actual_medium","actual_low"],
                       columns=["pred_high","pred_medium","pred_low"]))

# ------------------------------
# Step H: Save models + metadata
# ------------------------------
def save_models(rf_model, dt_model, vectorizer_meta, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    # save RF and DT
    joblib.dump(rf_model, os.path.join(out_dir, "priority_rf.joblib"))
    joblib.dump(dt_model, os.path.join(out_dir, "priority_dt.joblib"))
    # save vectorizer metadata and object if TF-IDF
    meta = {"vectorizer": vectorizer_meta}
    if vectorizer_meta["type"] == "tfidf":
        joblib.dump(vectorizer_meta["object"], os.path.join(out_dir, "tfidf_vectorizer.joblib"))
        meta["vectorizer"] = {"type": "tfidf", "saved": True}
    # Save metadata json
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Saved models and metadata to", out_dir)

# ------------------------------
# Step I: Interactive testing function
# ------------------------------
def interactive_test(rf_model, vectorizer_meta):
    print("\nInteractive test mode. Type a support ticket text and press Enter.")
    print("Type 'exit' to stop.")
    while True:
        text = input("\nTicket text> ").strip()
        if text.lower() in ("exit","quit"):
            print("Exiting interactive mode.")
            break
        if vectorizer_meta["type"] == "tfidf":
            tfidf = joblib.load(os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib"))
            x = tfidf.transform([text])
            X_for_pred = x if not hasattr(x, "toarray") else x.toarray()
        else:
            emb = embedder.encode([text])
            X_for_pred = np.array(emb)
        probs = rf_model.predict_proba(X_for_pred)[0]
        classes = rf_model.classes_
        idx = probs.argmax()
        print("Prediction:", classes[idx], f"(confidence {probs[idx]*100:.1f}%)")
        # also show full probs
        prob_str = ", ".join([f"{c}:{p*100:.1f}%" for c,p in zip(classes, probs)])
        print("All class probabilities ->", prob_str)

# ------------------------------
# Main: Run the pipeline
# ------------------------------
def main():
    print("STEP 1: Loading data...")
    master = load_csvs(DATA_DIR)
    print("STEP 2: Preprocessing...")
    usable = preprocess(master)
    usable = stratified_sample(usable, SAMPLE_MAX_ROWS)
    # prepare X, y
    X = usable["text"].values
    y = usable["priority_mapped"].values
    print("Label distribution (sampled):")
    print(pd.Series(y).value_counts())
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    print("Train/test split:", len(X_train), "/", len(X_test))
    # vectorize
    X_train_vec, X_test_vec, vectorizer_meta = vectorize_text(X_train, X_test, use_embeddings=USE_EMBEDDINGS)
    # balance training
    X_train_bal, y_train_bal = balance_training(X_train_vec, y_train, vectorizer_meta)
    print("Balanced training shape:", X_train_bal.shape, len(y_train_bal))
    # Train
    dt_model, rf_model = train_models(X_train_bal, y_train_bal)
    # Evaluate - for evaluation convert X_test_vec to dense if needed
    evaluate_and_print(dt_model, X_test_vec, y_test, "Decision Tree")
    evaluate_and_print(rf_model, X_test_vec, y_test, "Random Forest")
    # Save models
    save_models(rf_model, dt_model, vectorizer_meta, OUTPUT_DIR)
    # Interactive test
    interactive_test(rf_model, vectorizer_meta)

if __name__ == "__main__":
    main()
