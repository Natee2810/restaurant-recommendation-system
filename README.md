# Restaurant Recommendation System

## Problem
Predict which restaurants a customer is most likely to order from based on location, vendor data, and order history.

## Approach
- Framed as binary classification for ranking
- Used temporal train-validation split to avoid data leakage
- Feature engineering:
  - Geospatial features (Haversine distance)
  - Vendor attributes
  - Vendor popularity
  - Customer-vendor interactions

## Model
- HistGradientBoostingClassifier (histogram-based gradient boosting)

## Results
- Validation AUC: ~0.74

## Key Insight
Identified and fixed data leakage using temporal validation, leading to realistic model performance.

## Output
Predictions generated for all customer-location-vendor combinations using ranking-based selection.
