# 🍽️ Restaurant Recommendation System

An end-to-end machine learning project that predicts which restaurants a customer is most likely to order from, based on location, vendor attributes, and historical behavior.

The solution uses a supervised learning approach with temporal validation to avoid data leakage, and incorporates geospatial, popularity, and user–vendor interaction features.

Achieved a realistic validation AUC of ~0.74, demonstrating effective pattern learning beyond simple memorization.

> ⚠️ Key Insight: Initial model performance (~0.99 AUC) revealed data leakage. Fixing this using a time-based split led to a more reliable and realistic model.

---

## 📌 Problem Statement

Predict which vendors a customer is likely to order from given:
- Customer location  
- Vendor information  
- Historical order data  

Output format: `CID X LOC_NUM X VENDOR → 0 or 1`

---

## 🧠 Approach

- Framed as a **binary classification problem for ranking**
- Predicted probability of order for each customer–vendor pair
- Converted probabilities into **ranked recommendations (Top-K)**

---

## 🔧 Feature Engineering

Features were grouped into four categories:

### 1. Geospatial Features
- Haversine distance between customer and vendor  
- Distance buckets and serving range checks  

### 2. Vendor Attributes
- Rating, delivery charge, preparation time, category, rank, availability  

### 3. Vendor Popularity
- Total orders, average spend, ratings, and favorite rate  

### 4. Customer–Vendor Interaction
- Order frequency, repeat behavior, average rating, favorite flag  

---

## 🏗️ Model

- **HistGradientBoostingClassifier**
  - Histogram-based gradient boosting (similar to LightGBM)  
  - Handles non-linear relationships and missing values effectively  

---

## 📊 Evaluation

- Temporal train-validation split to avoid leakage  
- Metrics used:
  - **Validation AUC:** ~0.74  
  - **Average Precision:** ~0.66  

---

## 🔮 Prediction Strategy

- Scored all vendors for each customer-location pair  
- Ranked vendors based on predicted probability  
- Final prediction:
  - `1` if probability ≥ 0.5  
  - OR within **Top-10 vendors** (cold-start handling)  

---

## 📁 Project Structure

- `Recommendation_Engine.ipynb` — complete ML pipeline  
- `submission.csv` — final predictions  
- `README.md` — project overview  

---

## ▶️ How to Run

1. Clone the repository  
2. Install required libraries (`pandas`, `numpy`, `scikit-learn`, `matplotlib`)  
3. Open the notebook  
4. Run all cells sequentially  

> Note: Dataset is not included due to size constraints.

---

## 💡 Key Learnings

- Data leakage can severely inflate performance  
- Temporal validation is critical for recommendation systems  
- Feature engineering plays a bigger role than model choice  
- Cold-start users require fallback strategies like popularity and distance  

---

## 🚀 Future Improvements

- Use ranking metrics like Recall@K or NDCG  
- Improve cold-start handling with hybrid methods  
- Optimize negative sampling strategy  
- Replace `.apply()` with vectorized distance computation for scalability  
