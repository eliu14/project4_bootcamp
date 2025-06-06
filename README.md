Online Shoppers Purchase Prediction
A machine learning project that predicts whether online shoppers will complete a purchase based on their browsing behavior and session characteristics.

📊 Project Overview
This project analyzes online shopping behavior data to predict purchase likelihood using a Random Forest classifier. The model processes various features including page views, session duration, bounce rates, and visitor characteristics to determine if a shopping session will result in revenue generation.

## Members
* Sarah Burnap
* Jordan Dreyer
* Ellen Liu
* Renz Supnet
* Pratishtha Theeng

##  Data Source
[Data Source](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

🎯 Objective

* Primary Goal: Build a machine learning model to predict online purchase behavior
* Target Accuracy: 75% or higher
* Business Impact: Help e-commerce businesses identify high-value visitors and optimize conversion strategies

📁 Dataset Features
Numerical Features

* Administrative, Informational, ProductRelated: Number of pages visited in each category
* Administrative_Duration, Informational_Duration, ProductRelated_Duration: Time spent on each page type
* BounceRates: Percentage of visitors who enter and leave from the same page
* ExitRates: Percentage of pageviews that were the last in the session
* PageValues: Average value of pages visited before completing an e-commerce transaction
* OperatingSystems, Region, TrafficType: Technical and demographic identifiers

Categorical Features

* Month: Month of the year (converted to dummy variables)
* VisitorType: New Visitor, Returning Visitor, or Other
* Weekend: Boolean indicating if the session occurred on weekend

Target Variable

* Revenue: Boolean indicating whether the session resulted in a purchase

🔧 Data Preprocessing Pipeline

1. Data Loading and Exploration

# Load dataset (assumed to be stored in 'peoplewatcher' variable)
rockwall = peoplewatcher.copy()

2. Feature Engineering

* One-Hot Encoding: Convert categorical variables (Month, VisitorType) to dummy variables
* Boolean Conversion: Convert Revenue and Weekend to integer format
* Label Encoding: Handle any remaining categorical columns
* Data Type Standardization: Ensure all features are numeric (int64/float64)

3. Data Scaling

* StandardScaler: Normalize features to have mean=0 and std=1
* Train/Test Split: 80/20 split with stratification to maintain class balance

🤖 Model Implementation
Why Random Forest Was Chosen
After testing multiple algorithms including K-Nearest Neighbors (KNN) and TensorFlow Keras neural networks, Random Forest emerged as the clear winner for this project due to several key advantages:

🏆 Superior Performance for Tabular Data

* Optimized for structured data: Random Forest excels with tabular datasets like ours, unlike deep learning models that shine with unstructured data (images, text)
* Built-in feature selection: Automatically handles feature importance without extensive preprocessing
* Robust to outliers: Less sensitive to extreme values in bounce rates, page values, and session durations

🎯 Versatility & Flexibility

* Dual capability: Handles both regression and classification tasks seamlessly (our project could extend to predicting purchase amounts)
* Mixed data types: Naturally processes both categorical and numerical features without complex encoding requirements
* No assumptions: Doesn't require assumptions about data distribution (unlike KNN's distance-based approach)

📊 Practical Advantages Over Alternatives
vs. K-Nearest Neighbors (KNN):

* Scalability: Better performance with larger datasets
* Feature handling: Less sensitive to irrelevant features and different scales
* Training efficiency: Faster training time and better memory usage
* Interpretability: Provides feature importance rankings

vs. TensorFlow Keras Neural Networks:

* Complexity: Simpler architecture without need for extensive hyperparameter tuning
* Training time: Much faster training and inference
* Overfitting resistance: Built-in regularization through ensemble averaging
* Data requirements: Performs well with smaller datasets (neural networks typically need more data)
* Interpretability: Clear feature importance vs. black-box neural networks

🛡️ Robustness Features

* Ensemble strength: Combines multiple decision trees to reduce overfitting
* Bootstrap aggregating: Uses different data samples for each tree, improving generalization
* Class imbalance handling: Built-in class weighting options for our imbalanced revenue data
* Missing data tolerance: Can handle missing values naturally

Random Forest Classifier Configuration
pythonRandomForestClassifier(
    n_estimators=100,      # Number of trees in the forest
    max_depth=25,          # Maximum depth of trees
    min_samples_split=5,   # Minimum samples required to split
    random_state=50,       # For reproducibility
    class_weight='balanced' # Handle class imbalance
)

Key Model Features

* Ensemble Method: Uses multiple decision trees for robust predictions
* Class Balancing: Handles imbalanced revenue data
* Feature Scaling: Preprocessed with StandardScaler for optimal performance
* Feature Importance: Provides insights into which variables most influence purchase decisions

📈 Model Evaluation
Performance Metrics

* Accuracy Score: Primary evaluation metric
* Classification Report: Precision, recall, and F1-score for both classes
* Target Achievement: Success measured against 75% accuracy threshold

Visualization

* Bar chart comparing model accuracy against target performance
* Color-coded results (green for success, orange for improvement needed)
* Detailed performance statistics display

🚀 Usage Instructions
Prerequisites
pythonimport pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

Running the Model

1. Load your data into a variable called peoplewatcher
2. Execute preprocessing steps to clean and prepare the data
3. Train the model using the Random Forest classifier
4. Evaluate performance against the target accuracy
5. Visualize results with the provided plotting code

Code Structure<br>
├── Data Loading & Exploration<br>
├── Preprocessing Pipeline<br>
│   ├── Categorical Encoding<br>
│   ├── Data Type Conversion<br>
│   └── Feature Scaling<br>
├── Model Training<br>
│   ├── Train/Test Split<br>
│   ├── Random Forest Configuration<br>
│   └── Model Fitting<br>
└── Evaluation & Visualization<br>
    ├── Accuracy Calculation<br>
    ├── Classification Report<br>
   └── Performance Visualization<br>

📊 Expected Outcomes

Model Performance

* Target Accuracy: ≥75%
* Class Balance: Handled through balanced class weights
* Feature Importance: Random Forest provides insight into most predictive features

Business Insights

* Identify key factors influencing purchase decisions
* Understand visitor behavior patterns
* Optimize marketing and UX strategies based on predictive features

🔍 Key Preprocessing Steps

1. Handle Mixed Data Types: Convert booleans and categories to numeric
2. Feature Engineering: Create dummy variables for categorical data
3. Data Validation: Ensure all columns are properly converted before scaling
4. Stratified Splitting: Maintain class distribution in train/test sets

📋 Notes and Considerations

* Class Imbalance: The dataset likely has more non-purchasing sessions than purchasing ones
* Feature Scaling: Essential for optimal Random Forest performance
* Reproducibility: Random states set for consistent results across runs
* Model Interpretability: Random Forest provides feature importance rankings

🎯 Success Criteria
✅ Model achieves ≥75% accuracy
✅ All data types properly converted to numeric
✅ Successful handling of categorical variables
✅ Balanced approach to class imbalance
✅ Clear visualization of model performance

🔄 Future Improvements

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
* Feature selection techniques to identify most important predictors
* Ensemble methods combining multiple algorithms
* Cross-validation for more robust performance estimation
* ROC-AUC analysis for threshold optimization


This project demonstrates end-to-end machine learning workflow from data preprocessing through model evaluation, with a focus on practical e-commerce applications.
