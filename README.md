 # 🎫 TicketPriority AI Classifier

**An intelligent machine learning system for automated support ticket priority classification using ensemble learning and advanced NLP techniques.**

---

## 📋 Overview

TicketPriority AI is an end-to-end machine learning pipeline that automatically classifies support tickets into priority levels (High, Medium, Low). The system leverages state-of-the-art natural language processing and ensemble learning methods to deliver accurate, real-time predictions with confidence scores.

### **Key Capabilities**

- ✅ Multi-source CSV data integration
- ✅ Advanced text preprocessing with PII protection
- ✅ Dual vectorization strategies (TF-IDF / Transformer embeddings)
- ✅ Intelligent class balancing for imbalanced datasets
- ✅ Ensemble learning (Decision Tree + Random Forest)
- ✅ Interactive CLI for real-time predictions
- ✅ Comprehensive model evaluation metrics

---

## 🚀 Features

### **1. Flexible Data Processing**
- Loads and combines multiple CSV files automatically
- Handles various ticket formats (subject, body, message, content)
- Stratified sampling for large datasets (configurable up to 12,000 rows)
- Intelligent priority mapping from diverse formats (P1-P4, urgent, critical, etc.)

### **2. Advanced Text Preprocessing**
- Automatic PII masking (emails, phone numbers)
- Text normalization and cleaning
- Handles missing data gracefully
- Combines subject and body for comprehensive context

### **3. Dual Vectorization Engine**
- **TF-IDF**: Fast, lightweight (1500 features, 1-2 grams)
- **Sentence Transformers**: Semantic embeddings using `all-MiniLM-L6-v2`
- Configurable via single flag (`USE_EMBEDDINGS`)

### **4. Class Imbalance Handling**
- **SMOTE** for dense embeddings (synthetic minority oversampling)
- **RandomOverSampler** for sparse TF-IDF vectors
- Maintains class distribution integrity

### **5. Ensemble Learning Models**
- **Decision Tree**: Fast baseline classifier (max_depth=20)
- **Random Forest**: Production-ready model (300 estimators, balanced weights)
- Parallel processing support (`n_jobs=-1`)

### **6. Interactive Testing Interface**
- Real-time ticket classification via CLI
- Confidence scores for predictions
- Full probability distribution across all classes
- Easy model validation

---

## 📦 Installation

### **Prerequisites**
- Python 3.7+
- pip package manager

### **Required Dependencies**

```bash
pip install numpy pandas scikit-learn imbalanced-learn joblib
```

### **Optional (for Transformer Embeddings)**

```bash
pip install sentence-transformers
```

---

## ⚙️ Configuration

Edit the configuration variables in `final_model.py`:

```python
# Data Configuration
DATA_DIR = r"D:/AI(Python)/Unsupervisdmodel"  # Path to CSV files
OUTPUT_DIR = "./models"                        # Model save directory
SAMPLE_MAX_ROWS = 12000                        # Max rows for training

# Model Configuration
USE_EMBEDDINGS = True                          # True: Transformers, False: TF-IDF
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"     # Sentence-Transformer model
RF_N_ESTIMATORS = 300                          # Random Forest trees
RF_MAX_DEPTH = 20                              # Maximum tree depth
RANDOM_STATE = 42                              # Reproducibility seed
```

---

## 🎯 Usage

### **1. Training the Model**

```bash
python final_model.py
```

**Pipeline Steps:**
1. Loads all CSV files from `DATA_DIR`
2. Preprocesses and cleans text data
3. Performs stratified sampling
4. Generates feature vectors (TF-IDF or embeddings)
5. Balances training data
6. Trains Decision Tree and Random Forest models
7. Evaluates performance with metrics
8. Saves models to `OUTPUT_DIR`
9. Launches interactive testing mode

### **2. Interactive Testing**

After training, the system enters interactive mode:

```
Ticket text> Server is down, urgent fix needed
Prediction: high (confidence 87.3%)
All class probabilities -> high:87.3%, medium:9.2%, low:3.5%

Ticket text> Update documentation for API
Prediction: low (confidence 72.1%)
All class probabilities -> high:5.4%, medium:22.5%, low:72.1%
```

Type `exit` to quit interactive mode.

---

## 📊 Model Performance

### **Evaluation Metrics**
- **Accuracy Score**: Overall classification accuracy
- **Classification Report**: Precision, Recall, F1-Score per class
- **Confusion Matrix**: Detailed prediction breakdown

### **Example Output**
```
=== Evaluation: Random Forest ===
Accuracy: 0.8542

              precision    recall  f1-score   support
        high       0.84      0.87      0.85      1200
      medium       0.82      0.81      0.81      1500
         low       0.89      0.88      0.88      1800

Confusion matrix (actual rows, predicted cols):
                pred_high  pred_medium  pred_low
actual_high          1044          120        36
actual_medium         156         1215       129
actual_low             72          144      1584
```

---

## 💾 Saved Artifacts

After training, the following files are saved in `OUTPUT_DIR`:

| File | Description |
|------|-------------|
| `priority_rf.joblib` | Trained Random Forest model |
| `priority_dt.joblib` | Trained Decision Tree model |
| `tfidf_vectorizer.joblib` | TF-IDF vectorizer (if used) |
| `metadata.json` | Model configuration metadata |

---

## 📁 Input Data Format

### **Required CSV Columns**
Your CSV files should contain at least:
- `subject` or `title`: Ticket subject line
- `body`, `message`, `content`, or `description`: Ticket details
- `priority`: Priority label (various formats supported)

### **Supported Priority Formats**
- **Text**: high, medium, low, urgent, critical, normal, minor
- **P-notation**: P1 (high), P2 (medium), P3/P4 (low)
- **Custom**: Automatically mapped to high/medium/low

---

## 🔧 Technical Architecture

```
┌─────────────────┐
│  CSV Files      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Loading   │ ← Multi-file aggregation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │ ← PII masking, text cleaning
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vectorization  │ ← TF-IDF or Transformers
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Class Balance  │ ← SMOTE / Oversampling
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Training │ ← Decision Tree + Random Forest
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Evaluation     │ ← Metrics & Confusion Matrix
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Export   │ ← Joblib serialization
└─────────────────┘
```

---

## 🛠️ Customization

### **Switching Between TF-IDF and Embeddings**

```python
# For faster training with less memory
USE_EMBEDDINGS = False

# For better semantic understanding
USE_EMBEDDINGS = True
```

### **Adjusting Random Forest Parameters**

```python
# More trees = better accuracy, slower training
RF_N_ESTIMATORS = 500

# Deeper trees = more complex patterns, risk of overfitting
RF_MAX_DEPTH = 30
```

### **Handling Large Datasets**

```python
# Reduce sample size for faster iteration
SAMPLE_MAX_ROWS = 5000

# Increase for more comprehensive training
SAMPLE_MAX_ROWS = 20000
```

---

## 📈 Performance Optimization

### **Speed Improvements**
- Use TF-IDF instead of embeddings (10x faster)
- Reduce `SAMPLE_MAX_ROWS` for initial testing
- Lower `RF_N_ESTIMATORS` for faster training

### **Accuracy Improvements**
- Use Sentence-Transformer embeddings
- Increase `SAMPLE_MAX_ROWS` for more training data
- Tune `RF_N_ESTIMATORS` and `RF_MAX_DEPTH`
- Ensure balanced class distribution

---

## 🐛 Troubleshooting

### **Issue: "No CSV files found"**
- Verify `DATA_DIR` path is correct
- Ensure CSV files exist in the directory

### **Issue: "sentence-transformers not available"**
- Install: `pip install sentence-transformers`
- Or set `USE_EMBEDDINGS = False`

### **Issue: Memory errors with SMOTE**
- Reduce `SAMPLE_MAX_ROWS`
- Switch to TF-IDF with `USE_EMBEDDINGS = False`

### **Issue: Low accuracy**
- Check data quality and label consistency
- Increase training data size
- Try embeddings instead of TF-IDF
- Verify priority mapping logic

---

## 📝 License

This project is open-source and available for educational and commercial use.

---

## 👨‍💻 Author

Developed as part of an AI/ML initiative for intelligent ticket management systems.

---

## 🔮 Future Enhancements

- [ ] Multi-label classification support
- [ ] Real-time API endpoint deployment
- [ ] Deep learning models (BERT, RoBERTa)
- [ ] Automated hyperparameter tuning
- [ ] Web-based dashboard for predictions
- [ ] Integration with ticketing systems (Jira, ServiceNow)
- [ ] Model versioning and A/B testing

---

## 📞 Support

For issues, questions, or contributions, please refer to the project documentation or contact the development team.

---

**Built with ❤️ using Python, scikit-learn, and Sentence-Transformers**
