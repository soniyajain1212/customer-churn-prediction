"""
Customer Churn Prediction Analysis
Author: Soniya Jain
Description: Predicting telecom customer churn using Machine Learning

Requirements:
pip install pandas numpy matplotlib seaborn scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ==============================================
# 1. DATA GENERATION (Simulating Telecom Data)
# ==============================================

def generate_telecom_data(n_records=7000):
    """Generate realistic telecom customer data"""
    
    np.random.seed(42)
    
    data = []
    
    for i in range(n_records):
        customer_id = f'CUST{i+1:05d}'
        
        # Customer demographics
        gender = np.random.choice(['Male', 'Female'])
        senior_citizen = np.random.choice([0, 1], p=[0.85, 0.15])
        partner = np.random.choice(['Yes', 'No'], p=[0.48, 0.52])
        dependents = np.random.choice(['Yes', 'No'], p=[0.30, 0.70])
        
        # Account info
        tenure = np.random.randint(1, 73)  # months
        
        # Services
        phone_service = np.random.choice(['Yes', 'No'], p=[0.90, 0.10])
        multiple_lines = np.random.choice(['Yes', 'No', 'No phone service'], 
                                         p=[0.42, 0.48, 0.10])
        internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], 
                                           p=[0.34, 0.44, 0.22])
        
        # Additional services
        online_security = np.random.choice(['Yes', 'No', 'No internet'], p=[0.29, 0.49, 0.22])
        online_backup = np.random.choice(['Yes', 'No', 'No internet'], p=[0.34, 0.44, 0.22])
        device_protection = np.random.choice(['Yes', 'No', 'No internet'], p=[0.34, 0.44, 0.22])
        tech_support = np.random.choice(['Yes', 'No', 'No internet'], p=[0.29, 0.49, 0.22])
        streaming_tv = np.random.choice(['Yes', 'No', 'No internet'], p=[0.38, 0.40, 0.22])
        streaming_movies = np.random.choice(['Yes', 'No', 'No internet'], p=[0.39, 0.39, 0.22])
        
        # Contract and billing
        contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                   p=[0.55, 0.21, 0.24])
        paperless_billing = np.random.choice(['Yes', 'No'], p=[0.59, 0.41])
        payment_method = np.random.choice([
            'Electronic check', 'Mailed check', 
            'Bank transfer', 'Credit card'
        ], p=[0.34, 0.23, 0.22, 0.21])
        
        # Charges
        if internet_service == 'No':
            monthly_charges = np.random.uniform(18, 30)
        elif internet_service == 'DSL':
            monthly_charges = np.random.uniform(40, 70)
        else:  # Fiber optic
            monthly_charges = np.random.uniform(70, 120)
        
        total_charges = monthly_charges * tenure + np.random.uniform(-50, 50)
        total_charges = max(total_charges, 0)
        
        # Churn prediction logic (realistic factors)
        churn_probability = 0.1  # base probability
        
        # Contract type strongly affects churn
        if contract == 'Month-to-month':
            churn_probability += 0.35
        elif contract == 'One year':
            churn_probability += 0.10
        
        # Tenure affects churn (longer tenure = less churn)
        if tenure < 6:
            churn_probability += 0.25
        elif tenure < 12:
            churn_probability += 0.15
        elif tenure > 48:
            churn_probability -= 0.15
        
        # High monthly charges increase churn
        if monthly_charges > 80:
            churn_probability += 0.20
        
        # Fiber optic with high charges
        if internet_service == 'Fiber optic' and monthly_charges > 90:
            churn_probability += 0.15
        
        # No tech support increases churn
        if tech_support == 'No':
            churn_probability += 0.12
        
        # Electronic check payment increases churn
        if payment_method == 'Electronic check':
            churn_probability += 0.10
        
        # Senior citizens have slightly higher churn
        if senior_citizen == 1:
            churn_probability += 0.08
        
        # Determine churn
        churn = 'Yes' if np.random.random() < min(churn_probability, 0.95) else 'No'
        
        data.append({
            'customerID': customer_id,
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': round(monthly_charges, 2),
            'TotalCharges': round(total_charges, 2),
            'Churn': churn
        })
    
    return pd.DataFrame(data)

# ==============================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================

def perform_eda(df):
    """Comprehensive exploratory data analysis"""
    
    print("=== DATASET OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print(f"\nData Types:\n{df.dtypes.value_counts()}")
    print(f"\nMissing Values:\n{df.isnull().sum().sum()}")
    
    # Churn distribution
    print("\n=== CHURN DISTRIBUTION ===")
    churn_counts = df['Churn'].value_counts()
    churn_rate = (churn_counts['Yes'] / len(df)) * 100
    print(f"Churn Rate: {churn_rate:.2f}%")
    print(churn_counts)
    
    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Customer Churn Analysis - Key Factors', fontsize=16, fontweight='bold')
    
    # 1. Churn by Contract Type
    contract_churn = df.groupby(['Contract', 'Churn']).size().unstack()
    contract_churn_pct = contract_churn.div(contract_churn.sum(axis=1), axis=0) * 100
    contract_churn_pct.plot(kind='bar', ax=axes[0, 0], color=['#2ecc71', '#e74c3c'])
    axes[0, 0].set_title('Churn Rate by Contract Type')
    axes[0, 0].set_ylabel('Percentage')
    axes[0, 0].legend(title='Churn')
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)
    
    # 2. Churn by Tenure
    tenure_bins = [0, 12, 24, 48, 72]
    tenure_labels = ['0-12', '13-24', '25-48', '49-72']
    df['TenureGroup'] = pd.cut(df['tenure'], bins=tenure_bins, labels=tenure_labels)
    tenure_churn = df.groupby(['TenureGroup', 'Churn']).size().unstack()
    tenure_churn_pct = tenure_churn.div(tenure_churn.sum(axis=1), axis=0) * 100
    tenure_churn_pct.plot(kind='bar', ax=axes[0, 1], color=['#2ecc71', '#e74c3c'])
    axes[0, 1].set_title('Churn Rate by Tenure (months)')
    axes[0, 1].set_ylabel('Percentage')
    axes[0, 1].legend(title='Churn')
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=0)
    
    # 3. Monthly Charges Distribution
    df[df['Churn'] == 'No']['MonthlyCharges'].hist(ax=axes[0, 2], bins=30, 
                                                     alpha=0.6, label='No Churn', color='#2ecc71')
    df[df['Churn'] == 'Yes']['MonthlyCharges'].hist(ax=axes[0, 2], bins=30, 
                                                      alpha=0.6, label='Churned', color='#e74c3c')
    axes[0, 2].set_title('Monthly Charges Distribution')
    axes[0, 2].set_xlabel('Monthly Charges (₹)')
    axes[0, 2].legend()
    
    # 4. Churn by Internet Service
    internet_churn = df.groupby(['InternetService', 'Churn']).size().unstack()
    internet_churn_pct = internet_churn.div(internet_churn.sum(axis=1), axis=0) * 100
    internet_churn_pct.plot(kind='bar', ax=axes[1, 0], color=['#2ecc71', '#e74c3c'])
    axes[1, 0].set_title('Churn Rate by Internet Service')
    axes[1, 0].set_ylabel('Percentage')
    axes[1, 0].legend(title='Churn')
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)
    
    # 5. Churn by Tech Support
    tech_churn = df.groupby(['TechSupport', 'Churn']).size().unstack()
    tech_churn_pct = tech_churn.div(tech_churn.sum(axis=1), axis=0) * 100
    tech_churn_pct.plot(kind='bar', ax=axes[1, 1], color=['#2ecc71', '#e74c3c'])
    axes[1, 1].set_title('Churn Rate by Tech Support')
    axes[1, 1].set_ylabel('Percentage')
    axes[1, 1].legend(title='Churn')
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)
    
    # 6. Churn by Payment Method
    payment_churn = df.groupby(['PaymentMethod', 'Churn']).size().unstack()
    payment_churn_pct = payment_churn.div(payment_churn.sum(axis=1), axis=0) * 100
    payment_churn_pct.plot(kind='bar', ax=axes[1, 2], color=['#2ecc71', '#e74c3c'])
    axes[1, 2].set_title('Churn Rate by Payment Method')
    axes[1, 2].set_ylabel('Percentage')
    axes[1, 2].legend(title='Churn')
    axes[1, 2].set_xticklabels(axes[1, 2].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    return df

# ==============================================
# 3. DATA PREPROCESSING FOR ML
# ==============================================

def preprocess_data(df):
    """Prepare data for machine learning"""
    
    # Create a copy
    df_ml = df.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    
    categorical_cols = df_ml.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('customerID')  # Don't encode ID
    
    for col in categorical_cols:
        df_ml[col] = le.fit_transform(df_ml[col])
    
    # Separate features and target
    X = df_ml.drop(['customerID', 'Churn', 'TenureGroup'], axis=1, errors='ignore')
    y = df_ml['Churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

# ==============================================
# 4. MODEL TRAINING & EVALUATION
# ==============================================

def train_and_evaluate_model(X_train, X_test, y_train, y_test, feature_names):
    """Train Logistic Regression model and evaluate"""
    
    print("\n=== MODEL TRAINING ===")
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Accuracy: {accuracy:.2%}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': abs(model.coef_[0])
    }).sort_values('Coefficient', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Coefficient', y='Feature', palette='viridis')
    plt.title('Top 10 Most Important Features for Churn Prediction')
    plt.xlabel('Absolute Coefficient Value')
    plt.tight_layout()
    plt.show()
    
    return model, accuracy

# ==============================================
# 5. BUSINESS INSIGHTS & RECOMMENDATIONS
# ==============================================

def generate_insights(df):
    """Generate actionable business insights"""
    
    print("\n=== KEY FINDINGS ===")
    
    # Contract analysis
    contract_churn_rate = df.groupby('Contract')['Churn'].apply(
        lambda x: (x == 'Yes').sum() / len(x) * 100
    )
    print("\n1. Churn Rate by Contract Type:")
    for contract, rate in contract_churn_rate.items():
        print(f"   - {contract}: {rate:.1f}%")
    
    # Monthly charges analysis
    churned = df[df['Churn'] == 'Yes']['MonthlyCharges'].mean()
    retained = df[df['Churn'] == 'No']['MonthlyCharges'].mean()
    print(f"\n2. Average Monthly Charges:")
    print(f"   - Churned customers: ₹{churned:.2f}")
    print(f"   - Retained customers: ₹{retained:.2f}")
    print(f"   - Difference: ₹{churned - retained:.2f} higher for churned")
    
    # Tenure analysis
    churned_tenure = df[df['Churn'] == 'Yes']['tenure'].mean()
    retained_tenure = df[df['Churn'] == 'No']['tenure'].mean()
    print(f"\n3. Average Tenure:")
    print(f"   - Churned customers: {churned_tenure:.1f} months")
    print(f"   - Retained customers: {retained_tenure:.1f} months")
    
    # Tech support impact
    tech_yes_churn = (df[df['TechSupport'] == 'Yes']['Churn'] == 'Yes').mean() * 100
    tech_no_churn = (df[df['TechSupport'] == 'No']['Churn'] == 'Yes').mean() * 100
    print(f"\n4. Tech Support Impact:")
    print(f"   - With tech support: {tech_yes_churn:.1f}% churn")
    print(f"   - Without tech support: {tech_no_churn:.1f}% churn")
    
    print("\n=== RECOMMENDATIONS ===")
    print("1. 🎯 Target month-to-month customers with long-term contract incentives")
    print("2. 💰 Review pricing strategy for high monthly charge customers (>₹80)")
    print("3. 🛠️  Promote tech support services - customers with tech support have lower churn")
    print("4. 📅 Implement retention campaigns for customers with <12 months tenure")
    print("5. 💳 Encourage customers to switch from electronic check to more stable payment methods")
    print("6. 🌐 Investigate Fiber optic service quality - higher churn than DSL")

# ==============================================
# 6. MAIN EXECUTION
# ==============================================

if __name__ == "__main__":
    print("Starting Customer Churn Prediction Analysis...\n")
    
    # Generate data
    df = generate_telecom_data(7000)
    print(f"Generated {len(df)} customer records")
    
    # Save to CSV
    df.to_csv('telecom_customer_churn.csv', index=False)
    print("Data saved to 'telecom_customer_churn.csv'")
    
    # EDA
    df = perform_eda(df)
    
    # Preprocess
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    
    # Train model
    model, accuracy = train_and_evaluate_model(X_train, X_test, y_train, y_test, feature_names)
    
    # Generate insights
    generate_insights(df)
    
    print("\n✅ Analysis Complete!")
    print(f"📊 Model Accuracy: {accuracy:.2%}")
