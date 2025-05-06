import streamlit as st
from faker import Faker
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_train_test_split
from collections import defaultdict
import scipy.stats as stats

warnings.filterwarnings('ignore')

# Initialize Faker
fake = Faker()

# --- Helper Functions from Notebook ---
def ndcg_at_k(predictions_df, k=10):
    """
    Computes Normalized Discounted Cumulative Gain (NDCG) at k.
    Assumes predictions_df has 'uid', 'iid', 'rating', 'prediction' columns.
    'rating' is actual relevance, 'prediction' is predicted score.
    """
    def single_user_ndcg(user_data):
        # Sort items by predicted score for the user
        top_k_recs = user_data.sort_values(by='prediction', ascending=False).head(k)
        actual_relevance = top_k_recs['rating'].values

        # DCG for actual relevance
        actual_dcg = np.sum(actual_relevance / np.log2(np.arange(2, len(actual_relevance) + 2)))

        # Ideal DCG (sorted by actual relevance)
        ideal_relevance = np.sort(user_data['rating'].values)[::-1][:k] # Sort all user ratings and take top k
        ideal_dcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))

        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0

    if predictions_df.empty or 'uid' not in predictions_df.columns:
        return 0.0 # Or handle as an error/warning

    # Compute NDCG for each user and return the average
    # Ensure 'uid' is suitable for groupby (e.g., not all NaNs)
    if predictions_df['uid'].isnull().all():
        st.warning("All 'uid' values are NaN in predictions_df for NDCG calculation.")
        return 0.0

    ndcg_scores = predictions_df.groupby('uid').apply(single_user_ndcg)
    return ndcg_scores.mean()


# --- App Layout ---
st.set_page_config(layout="wide")
st.title("Predictive Analytics and Recommendation Systems in Banking")

# --- Sidebar for Navigation (Optional, but good for large apps) ---
# section = st.sidebar.radio("Go to",
#                            ["Introduction", "Data Generation & EDA",
#                             "Loan Default Prediction", "Customer Segmentation",
#                             "Product Recommendation"])

# --- Introduction ---
st.header("Project Objective")
st.markdown("""
This project showcases:
1.  **Loan Default Prediction** using Supervised Learning.
2.  **Customer Segmentation** using Unsupervised Learning.
3.  **Bank Product Recommendation** using a Recommendation Engine.
""")

# --- Data Collection / Generation ---
st.header("1. Data Collection and Generation")
st.markdown("Synthetic data is generated for customers, transactions, loans, and products.")

# Cache data generation to speed up reruns
@st.cache_data
def generate_data():
    # Customer Data
    num_customers = 5000
    customer_data = {
        'CustomerID': [f'C{1000+i}' for i in range(num_customers)],
        'Age': [random.randint(18, 70) for _ in range(num_customers)],
        'Gender': [random.choice(['Male', 'Female']) for _ in range(num_customers)],
        'Income': [random.randint(20000, 200000) for _ in range(num_customers)],
        'Education': [random.choice(['High School', 'Bachelors', 'Masters', 'PhD']) for _ in range(num_customers)],
        'EmploymentType': [random.choice(['Salaried', 'Self-Employed', 'Unemployed', 'Student']) for _ in range(num_customers)],
        'MaritalStatus': [random.choice(['Single', 'Married', 'Divorced']) for _ in range(num_customers)],
        'CreditScore': [random.randint(300, 850) for _ in range(num_customers)]
    }
    customer_df = pd.DataFrame(customer_data)

    # Transaction Data
    num_transactions = 20000
    transaction_data = {
        'TransactionID': [f'T{10000+i}' for i in range(num_transactions)],
        'CustomerID': [random.choice(customer_df['CustomerID']) for _ in range(num_transactions)],
        'TransactionDate': [fake.date_between(start_date='-2y', end_date='today') for _ in range(num_transactions)],
        'TransactionAmount': [round(random.uniform(10, 5000), 2) for _ in range(num_transactions)],
        'TransactionType': [random.choice(['Deposit', 'Withdrawal', 'Transfer', 'Payment']) for _ in range(num_transactions)],
        'Channel': [random.choice(['Online', 'Mobile', 'ATM', 'Branch']) for _ in range(num_transactions)]
    }
    transaction_df = pd.DataFrame(transaction_data)

    # Loan Data
    num_loans = 1000
    loan_data = {
        'LoanID': [f'L{5000+i}' for i in range(num_loans)],
        'CustomerID': [random.choice(customer_df['CustomerID']) for _ in range(num_loans)],
        'LoanAmount': [random.randint(1000, 100000) for _ in range(num_loans)],
        'LoanTerm': [random.choice([12, 24, 36, 48, 60]) for _ in range(num_loans)], # months
        'InterestRate': [round(random.uniform(3, 15), 2) for _ in range(num_loans)],
        'LoanPurpose': [random.choice(['Home', 'Car', 'Education', 'Personal', 'Business']) for _ in range(num_loans)],
        'LoanStatus': [random.choices(['Current', 'Paid Off', 'Default'], weights=[0.7, 0.2, 0.1])[0] for _ in range(num_loans)]
    }
    loan_df = pd.DataFrame(loan_data)

    # Product Data
    bank_products = ['Savings Account', 'Checking Account', 'Credit Card', 'Mortgage', 'Personal Loan', 'Investment Account', 'Insurance']
    num_product_holdings = 7000
    product_data = {
        'CustomerID': [random.choice(customer_df['CustomerID']) for _ in range(num_product_holdings)],
        'Product': [random.choice(bank_products) for _ in range(num_product_holdings)],
        'OpenDate': [fake.date_between(start_date='-5y', end_date='today') for _ in range(num_product_holdings)]
    }
    product_df = pd.DataFrame(product_data)
    product_df = product_df.drop_duplicates(subset=['CustomerID', 'Product']) # Ensure one customer doesn't have the same product multiple times

    return customer_df, transaction_df, loan_df, product_df

customer_df, transaction_df, loan_df, product_df = generate_data()

st.subheader("Sample Generated Data")
if st.checkbox("Show Customer Data Sample"):
    st.dataframe(customer_df.head())
if st.checkbox("Show Transaction Data Sample"):
    st.dataframe(transaction_df.head())
if st.checkbox("Show Loan Data Sample"):
    st.dataframe(loan_df.head())
if st.checkbox("Show Product Holdings Data Sample"):
    st.dataframe(product_df.head())

# --- Data Preprocessing and EDA ---
st.header("2. Data Preprocessing and Exploratory Data Analysis (EDA)")

# Merge dataframes
@st.cache_data
def merge_data(_customer_df, _loan_df):
    # Merge customer and loan data
    # We'll use a left merge to keep all customers, even if they don't have a loan
    # For loan default prediction, we are interested in customers who have taken loans.
    # So, an inner merge on CustomerID might be more appropriate for that specific task.
    # However, for general EDA, a left merge can be useful.
    # Let's use the loan_df as the base and merge customer info to it, as loan status is key.
    merged_df = pd.merge(_loan_df, _customer_df, on='CustomerID', how='left')
    return merged_df

merged_df_loan_focus = merge_data(customer_df.copy(), loan_df.copy())

st.subheader("Merged Data (Loan Focus)")
st.markdown("Loan data merged with customer details.")
if st.checkbox("Show Merged Data (Loan Focus) Sample"):
    st.dataframe(merged_df_loan_focus.head())

st.subheader("Data Overview")
if st.checkbox("Show Data Info (Merged Loan Focus)"):
    st.text(merged_df_loan_focus.info(verbose=True))

if st.checkbox("Show Descriptive Statistics (Merged Loan Focus)"):
    st.dataframe(merged_df_loan_focus.describe(include='all'))

if st.checkbox("Show Missing Values Count (Merged Loan Focus)"):
    st.dataframe(merged_df_loan_focus.isnull().sum())


st.subheader("Exploratory Data Analysis (EDA) - Visualizations")

# Plot 1: Distribution of Age
st.markdown("#### Distribution of Customer Age (Loan Applicants)")
fig_age, ax_age = plt.subplots()
sns.histplot(merged_df_loan_focus['Age'].dropna(), kde=True, ax=ax_age)
ax_age.set_title('Distribution of Customer Age')
st.pyplot(fig_age)

# Plot 2: Loan Status Distribution
st.markdown("#### Loan Status Distribution")
fig_loan_status, ax_loan_status = plt.subplots()
merged_df_loan_focus['LoanStatus'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax_loan_status)
ax_loan_status.set_title('Loan Status Distribution')
ax_loan_status.set_ylabel('') # Hide the default ylabel
st.pyplot(fig_loan_status)

# Plot 3: Credit Score Distribution
st.markdown("#### Credit Score Distribution (Loan Applicants)")
fig_credit_score, ax_credit_score = plt.subplots()
sns.histplot(merged_df_loan_focus['CreditScore'].dropna(), kde=True, color='skyblue', ax=ax_credit_score)
ax_credit_score.set_title('Credit Score Distribution of Loan Applicants')
st.pyplot(fig_credit_score)

# Plot 4: Correlation Heatmap (Numerical Features for Loan Prediction)
st.markdown("#### Correlation Heatmap (Numerical Features)")
numerical_features_loan = merged_df_loan_focus.select_dtypes(include=np.number)
# Drop LoanID if it's just an identifier and not a feature
if 'LoanID' in numerical_features_loan.columns and not pd.api.types.is_numeric_dtype(merged_df_loan_focus['LoanID']):
     # If LoanID was string based and converted to numeric by describe, exclude it
    pass # It won't be in select_dtypes(include=np.number) anyway
elif 'LoanID' in numerical_features_loan.columns: # If it is numeric but just an ID
    numerical_features_loan = numerical_features_loan.drop(columns=['LoanID'], errors='ignore')


fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
sns.heatmap(numerical_features_loan.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
ax_corr.set_title('Correlation Heatmap of Numerical Features')
st.pyplot(fig_corr)

# Plot 5: Average Income by Education Level
st.markdown("#### Average Income by Education Level (Loan Applicants)")
fig_income_edu, ax_income_edu = plt.subplots()
avg_income_edu = merged_df_loan_focus.groupby('Education')['Income'].mean().sort_values(ascending=False)
avg_income_edu.plot(kind='bar', color='lightgreen', ax=ax_income_edu)
ax_income_edu.set_title('Average Income by Education Level')
ax_income_edu.set_ylabel('Average Income')
ax_income_edu.set_xlabel('Education Level')
plt.xticks(rotation=45)
st.pyplot(fig_income_edu)

# Plot 6: Loan Default Rate by Employment Type
st.markdown("#### Loan Default Rate by Employment Type")
default_counts = merged_df_loan_focus[merged_df_loan_focus['LoanStatus'] == 'Default'].groupby('EmploymentType').size()
total_counts = merged_df_loan_focus.groupby('EmploymentType').size()
default_rate = (default_counts / total_counts).fillna(0).sort_values(ascending=False)

fig_default_emp, ax_default_emp = plt.subplots()
default_rate.plot(kind='bar', color='salmon', ax=ax_default_emp)
ax_default_emp.set_title('Loan Default Rate by Employment Type')
ax_default_emp.set_ylabel('Default Rate')
ax_default_emp.set_xlabel('Employment Type')
plt.xticks(rotation=45)
st.pyplot(fig_default_emp)


# --- Feature Engineering ---
st.header("3. Feature Engineering")
st.markdown("Creating new features from existing data to improve model performance.")

@st.cache_data
def feature_engineer(_df):
    df = _df.copy()
    # Example: Debt-to-Income Ratio (DTI)
    # Assuming 'LoanAmount' is the total loan and 'Income' is annual.
    # For a more accurate DTI, we'd need monthly income and total monthly debt payments.
    # This is a simplified version.
    df['DTI'] = df['LoanAmount'] / (df['Income'] + 1e-6) # Add small constant to avoid division by zero

    # Example: Loan Amount to Credit Score Ratio
    df['LoanAmount_CreditScore_Ratio'] = df['LoanAmount'] / (df['CreditScore'] + 1e-6)

    # Example: Age Group
    bins = [18, 30, 40, 50, 60, 100]
    labels = ['18-30', '31-40', '41-50', '51-60', '60+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    return df

merged_df_featured = feature_engineer(merged_df_loan_focus.copy())
st.subheader("Data with New Features")
if st.checkbox("Show Data with Engineered Features Sample"):
    st.dataframe(merged_df_featured.head())


# --- Predictive Analytics: Loan Default Prediction ---
st.header("4. Predictive Analytics: Loan Default Prediction (Supervised Learning)")
st.markdown("Predicting whether a customer is likely to default on their loan.")

# Prepare data for modeling
@st.cache_data
def prepare_loan_data(_df):
    df_model = _df.copy()

    # Handle missing values (simple imputation for demonstration)
    for col in ['Age', 'Income', 'CreditScore', 'DTI', 'LoanAmount_CreditScore_Ratio']:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(df_model[col].median())

    # For categorical features, fill with mode or a specific category like 'Unknown'
    for col in ['Gender', 'Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose', 'AgeGroup']:
         if col in df_model.columns:
            df_model[col] = df_model[col].fillna(df_model[col].mode()[0] if not df_model[col].mode().empty else 'Unknown')


    # Target variable: 'LoanStatus' -> 'Default' (1) or not (0)
    df_model['Default'] = df_model['LoanStatus'].apply(lambda x: 1 if x == 'Default' else 0)

    # Select features
    # 'CustomerID', 'LoanID', 'LoanStatus' are dropped as they are identifiers or the original target
    features = [col for col in df_model.columns if col not in ['CustomerID', 'LoanID', 'LoanStatus', 'Default']]
    
    # Identify categorical and numerical features
    categorical_cols = df_model[features].select_dtypes(include=['object', 'category']).columns
    numerical_cols = df_model[features].select_dtypes(include=np.number).columns

    # Label Encoding for categorical features
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le # Store for potential inverse transform or understanding

    # Scaling numerical features
    scaler = StandardScaler()
    df_model[numerical_cols] = scaler.fit_transform(df_model[numerical_cols])
    
    X = df_model[features]
    y = df_model['Default']
    
    return X, y, df_model # Return df_model to show preprocessed data

X, y, preprocessed_df_loan = prepare_loan_data(merged_df_featured.copy())

st.subheader("Preprocessed Data for Loan Default Modeling")
if st.checkbox("Show Preprocessed Data Sample (Loan Default)"):
    st.dataframe(preprocessed_df_loan.head())
    st.write("Features (X):")
    st.dataframe(X.head())
    st.write("Target (y):")
    st.dataframe(y.head())


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

st.markdown(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

# Model Training
st.subheader("Model Training and Evaluation")
models = {
    "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = {}
trained_models = {}

for model_name, model in models.items():
    st.markdown(f"#### Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # For ROC AUC

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    results[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC AUC": roc_auc,
        "Confusion Matrix": cm
    }
    trained_models[model_name] = model # Store trained model

    st.write(f"**{model_name} Results:**")
    st.json({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in results[model_name].items()})


# Plot ROC Curves
st.markdown("#### ROC Curves")
fig_roc, ax_roc = plt.subplots()
for model_name, model in trained_models.items():
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        ax_roc.plot(fpr, tpr, label=f'{model_name} (AUC = {results[model_name]["ROC AUC"]:.2f})')
ax_roc.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curve for Loan Default Prediction Models')
ax_roc.legend()
st.pyplot(fig_roc)

# Feature Importance (for Random Forest)
st.markdown("#### Feature Importance (Random Forest)")
if "Random Forest" in trained_models:
    rf_model = trained_models["Random Forest"]
    importances = rf_model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(10), ax=ax_fi) # Top 10
    ax_fi.set_title('Top 10 Feature Importances from Random Forest')
    st.pyplot(fig_fi)


# --- Customer Segmentation (Unsupervised Learning) ---
st.header("5. Customer Segmentation (Unsupervised Learning)")
st.markdown("Grouping customers based on their characteristics using K-Means clustering.")

# Prepare data for segmentation (using customer_df and some aggregated transaction/loan info if desired)
# For simplicity, let's use features from customer_df.
@st.cache_data
def prepare_segmentation_data(_customer_df):
    seg_df = _customer_df.copy()
    # Select features for segmentation
    # For example: Age, Income, CreditScore
    features_for_segmentation = ['Age', 'Income', 'CreditScore']
    seg_df_selected = seg_df[features_for_segmentation].copy()

    # Handle missing values (if any, though Faker data usually doesn't have them)
    seg_df_selected.fillna(seg_df_selected.median(), inplace=True)

    # Scale features
    scaler_seg = StandardScaler()
    scaled_features = scaler_seg.fit_transform(seg_df_selected)
    scaled_df = pd.DataFrame(scaled_features, columns=features_for_segmentation)
    return scaled_df, seg_df_selected # Return original selected for attaching labels

scaled_segmentation_df, original_segmentation_df = prepare_segmentation_data(customer_df)

st.subheader("Data for Segmentation (Scaled)")
if st.checkbox("Show Scaled Data Sample (Segmentation)"):
    st.dataframe(scaled_segmentation_df.head())

# K-Means Clustering
st.subheader("K-Means Clustering")

# Elbow Method to find optimal K
st.markdown("#### Elbow Method for Optimal K")
inertia = []
k_range = range(1, 11)
for k_val in k_range:
    kmeans = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
    kmeans.fit(scaled_segmentation_df)
    inertia.append(kmeans.inertia_)

fig_elbow, ax_elbow = plt.subplots()
ax_elbow.plot(k_range, inertia, marker='o')
ax_elbow.set_xlabel('Number of clusters (K)')
ax_elbow.set_ylabel('Inertia')
ax_elbow.set_title('Elbow Method for Optimal K')
st.pyplot(fig_elbow)

st.markdown("Based on the elbow plot, choose an optimal K (e.g., 3 or 4). Let's proceed with K=4 for demonstration.")
chosen_k = st.slider("Select K for K-Means Clustering", min_value=2, max_value=10, value=4)

kmeans_final = KMeans(n_clusters=chosen_k, random_state=42, n_init='auto')
original_segmentation_df['Cluster'] = kmeans_final.fit_predict(scaled_segmentation_df)

st.subheader(f"Customer Segments (K={chosen_k})")
if st.checkbox("Show Customer Data with Cluster Labels"):
    st.dataframe(original_segmentation_df.head())

# Visualize Clusters (using PCA for 2D if more than 2 features were used)
st.markdown("#### Cluster Visualization (PCA)")
if scaled_segmentation_df.shape[1] > 2:
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(scaled_segmentation_df)
    pca_df = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'])
    pca_df['Cluster'] = original_segmentation_df['Cluster']

    fig_pca, ax_pca = plt.subplots(figsize=(10,7))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_df, palette='viridis', ax=ax_pca)
    ax_pca.set_title(f'Customer Segments Visualized with PCA (K={chosen_k})')
    st.pyplot(fig_pca)
else: # if 2 features were used directly
    fig_scatter, ax_scatter = plt.subplots(figsize=(10,7))
    sns.scatterplot(x=scaled_segmentation_df.columns[0], y=scaled_segmentation_df.columns[1], hue=original_segmentation_df['Cluster'], data=scaled_segmentation_df.assign(Cluster=original_segmentation_df['Cluster']), palette='viridis', ax=ax_scatter)
    ax_scatter.set_title(f'Customer Segments (K={chosen_k})')
    st.pyplot(fig_scatter)


st.markdown("#### Cluster Profiles")
cluster_profiles = original_segmentation_df.groupby('Cluster').mean()
st.dataframe(cluster_profiles)
st.markdown("""
*Interpret the profiles:*
* **Cluster 0:** Might represent younger customers with lower income and credit scores.
* **Cluster 1:** Could be mid-aged customers with moderate income and good credit scores.
* **And so on...** (These are examples, actual interpretation depends on the data)
""")


# --- Recommendation System ---
st.header("6. Recommendation System (Bank Products)")
st.markdown("Recommending bank products to customers using Collaborative Filtering (SVD).")

# Prepare data for recommendation (user-item interaction)
# We need 'CustomerID', 'Product', and some form of 'rating'.
# Since we don't have explicit ratings, we can use 1 for products a customer holds.
@st.cache_data
def prepare_recommendation_data(_product_df):
    reco_df = _product_df[['CustomerID', 'Product']].copy()
    reco_df['Rating'] = 1 # Implicit rating: 1 if customer has the product
    
    # Need to map CustomerID and Product to integer IDs for Surprise
    reader = Reader(rating_scale=(1, 1)) # Only one rating value
    data = Dataset.load_from_df(reco_df[['CustomerID', 'Product', 'Rating']], reader)
    return data, reco_df

reco_data, reco_df_orig = prepare_recommendation_data(product_df)
trainset, testset = surprise_train_test_split(reco_data, test_size=0.25, random_state=42)

st.subheader("Data for Recommendation")
if st.checkbox("Show Sample Interaction Data (for Recommendation)"):
    st.dataframe(reco_df_orig.head())

# Train SVD model
st.subheader("SVD Model Training")
svd_model = SVD(n_epochs=20, n_factors=50, random_state=42) # n_epochs, n_factors are hyperparameters
with st.spinner("Training SVD model... This might take a moment."):
    svd_model.fit(trainset)
st.success("SVD Model trained successfully!")

# Generate recommendations for a sample user
st.subheader("Generate Recommendations")
sample_customer_id = st.selectbox("Select a CustomerID to get recommendations:", options=customer_df['CustomerID'].unique()[:20]) # Show first 20 for selection

if sample_customer_id:
    # Get all unique products
    all_products = product_df['Product'].unique()
    # Products the user already has
    products_user_has = set(reco_df_orig[reco_df_orig['CustomerID'] == sample_customer_id]['Product'])
    
    st.write(f"Products {sample_customer_id} already has: {', '.join(products_user_has) if products_user_has else 'None'}")

    # Predict ratings for products the user doesn't have
    predictions = []
    for product_name in all_products:
        if product_name not in products_user_has:
            # Surprise uses raw user and item ids. Here, CustomerID and Product name are used as raw ids.
            pred = svd_model.predict(uid=sample_customer_id, iid=product_name)
            predictions.append({'Product': product_name, 'PredictedRating': pred.est})
    
    recommendations_df = pd.DataFrame(predictions)
    top_n_recommendations = recommendations_df.sort_values(by='PredictedRating', ascending=False).head(5)
    
    st.write(f"Top 5 product recommendations for {sample_customer_id}:")
    st.dataframe(top_n_recommendations)

# Evaluate Recommendation Model (NDCG - using the helper function)
st.subheader("Recommendation Model Evaluation (NDCG@10)")
# Make predictions on the test set
test_predictions_raw = svd_model.test(testset)
predictions_list = []
for pred_obj in test_predictions_raw:
    predictions_list.append({
        'uid': pred_obj.uid,
        'iid': pred_obj.iid,
        'rating': pred_obj.r_ui, # actual rating
        'prediction': pred_obj.est # predicted rating
    })
predictions_eval_df = pd.DataFrame(predictions_list)

if not predictions_eval_df.empty:
    ndcg_score = ndcg_at_k(predictions_eval_df, k=10)
    st.metric(label="NDCG@10 Score", value=f"{ndcg_score:.4f}")
    st.markdown("""
    *The Normalized Discounted Cumulative Gain (NDCG) score indicates the performance in ranking relevant items.
    A higher score (closer to 1) reflects the system's ability to effectively prioritize the most relevant recommendations.*
    """)
else:
    st.warning("Could not generate predictions for NDCG calculation (e.g., test set was empty or issues with UIDs).")


# --- Conclusion ---
st.header("7. Conclusion")
st.markdown("""
This Streamlit application demonstrated a pipeline for:
- **Generating synthetic banking data.**
- **Performing Exploratory Data Analysis** to understand data characteristics.
- **Building supervised learning models** for loan default prediction, evaluating their performance, and identifying key features.
- **Applying unsupervised learning (K-Means)** for customer segmentation and profiling.
- **Developing a collaborative filtering recommendation system** for bank products and evaluating its ranking quality.

This provides a foundational framework for leveraging data science in the banking sector to make informed decisions, manage risk, and enhance customer experience.
""")

st.sidebar.header("About")
st.sidebar.info("This app demonstrates various data science techniques applied to synthetic banking data, based on a Jupyter Notebook.")
```
