import streamlit as st
from faker import Faker
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import accuracy as surprise_accuracy
from collections import defaultdict
import scipy.stats as stats

warnings.filterwarnings('ignore')

# Helper function to display plots in Streamlit
def show_plot(fig=None):
    """Shows the current matplotlib figure in Streamlit, or a specific figure if provided."""
    if fig:
        st.pyplot(fig)
    else:
        st.pyplot(plt.gcf())
    plt.clf() # Clear the figure after displaying to avoid overlap

# --- Notebook Content Starts ---

st.set_page_config(layout="wide")

st.markdown("### PREDICTIVE ANALYTICS AND RECOMMENDATION SYSTEMS IN BANKING")
st.markdown("""
#### Project objective
     This project is about predicting the Loan Defaults using Supervised Learning, Customer Segmentation using Unsupervised Learning and Recommending Bank Products through a Recommendation Engine.
""")

st.markdown("---")
st.markdown("### Data Collection")
st.markdown("For this demonstration, we will generate synthetic data similar to what might be used in a banking context.")

# Wrapped data generation in a function with caching for performance
@st.cache_data
def generate_data(num_customers=2000, num_products=10, num_transactions=5000):
    fake = Faker()
    Faker.seed(0) # for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Customer data
    customers = []
    for i in range(num_customers):
        customers.append({
            'CustomerID': 1000 + i,
            'Age': random.randint(18, 70),
            'Gender': random.choice(['Male', 'Female']),
            'Income': round(random.uniform(20000, 150000), 2),
            'CreditScore': random.randint(300, 850),
            'CustomerSegment': random.choice(['Retail', 'Priority', 'Wealth']),
            'EmploymentStatus': random.choice(['Employed', 'Unemployed', 'Self-employed', 'Student', 'Retired']),
            'MaritalStatus': random.choice(['Single', 'Married', 'Divorced', 'Widowed']),
            'Dependents': random.randint(0, 5),
            'EducationLevel': random.choice(['High School', 'Bachelors', 'Masters', 'PhD', 'Some College']),
            'HousingType': random.choice(['Owned', 'Rented', 'Mortgaged']),
            'YearsWithBank': random.randint(0, 30)
        })
    customer_df = pd.DataFrame(customers)

    # Account data
    accounts = []
    for i in range(num_customers):
        num_accounts = random.randint(1, 3)
        for _ in range(num_accounts):
            accounts.append({
                'AccountID': 50000 + len(accounts),
                'CustomerID': 1000 + i,
                'AccountType': random.choice(['Savings', 'Checking', 'Loan', 'Credit Card']),
                'Balance': round(random.uniform(-5000, 50000) if random.choice(['Savings', 'Checking']) else random.uniform(500, 100000), 2),
                'AccountOpenDate': fake.date_between(start_date='-10y', end_date='today')
            })
    account_df = pd.DataFrame(accounts)

    # Loan specific data (for default prediction)
    loan_accounts = account_df[account_df['AccountType'] == 'Loan'].copy()
    if not loan_accounts.empty:
        loan_accounts['LoanAmount'] = loan_accounts['Balance'] # Assuming balance for loan is loan amount
        loan_accounts['LoanTermMonths'] = random.choices([12, 24, 36, 48, 60, 120, 240, 360], k=len(loan_accounts))
        loan_accounts['InterestRate'] = [random.uniform(2.5, 15.0) for _ in range(len(loan_accounts))]
        # Simulate Loan Default based on CreditScore, Income, and LoanAmount
        default_propensity = (850 - customer_df.loc[customer_df['CustomerID'].isin(loan_accounts['CustomerID']), 'CreditScore'].values) / 550 + \
                             (150000 - customer_df.loc[customer_df['CustomerID'].isin(loan_accounts['CustomerID']), 'Income'].values) / 130000 + \
                             loan_accounts['LoanAmount'].values / 100000
        
        # Normalize propensity and introduce randomness
        normalized_propensity = (default_propensity - np.min(default_propensity)) / (np.max(default_propensity) - np.min(default_propensity)) if len(default_propensity) > 1 and np.max(default_propensity) != np.min(default_propensity) else np.zeros_like(default_propensity)

        loan_accounts['LoanDefault'] = [1 if p > 0.65 and random.random() < p*0.8 else 0 for p in normalized_propensity] # Adjusted threshold for more defaults

    # Product data (for recommendation)
    product_list = [f'Product_{j}' for j in range(1, num_products + 1)]
    product_details = {
        'Product_1': 'High-Yield Savings Account', 'Product_2': 'Premium Checking Account',
        'Product_3': 'Personal Loan', 'Product_4': 'Home Mortgage',
        'Product_5': 'Platinum Credit Card', 'Product_6': 'Gold Credit Card',
        'Product_7': 'Investment Portfolio Management', 'Product_8': 'Retirement Planning Services',
        'Product_9': 'Student Loan', 'Product_10': 'Business Loan'
    }

    # Customer-Product Interaction (for recommendation)
    interactions = []
    for cust_id in customer_df['CustomerID']:
        num_prods_held = random.randint(1, 5)
        prods_held = random.sample(product_list, num_prods_held)
        for prod_name in prods_held:
            interactions.append({
                'uid': cust_id, # User ID
                'pid': prod_name, # Product ID
                'rating': random.randint(3, 5) # Explicit rating (e.g., satisfaction) or 1 if held
            })
    interaction_df = pd.DataFrame(interactions)
    interaction_df['pid_description'] = interaction_df['pid'].map(product_details)


    # Merge customer and relevant loan data for default prediction
    # First, ensure customer_df has CustomerID as index for easy mapping if needed, or use merge
    if not loan_accounts.empty:
        loan_df_merged = pd.merge(customer_df, loan_accounts[['CustomerID', 'LoanAmount', 'LoanTermMonths', 'InterestRate', 'LoanDefault']], on='CustomerID', how='inner')
    else: # Create an empty df with expected columns if no loans
        loan_df_merged = pd.DataFrame(columns=list(customer_df.columns) + ['LoanAmount', 'LoanTermMonths', 'InterestRate', 'LoanDefault'])

    return customer_df, account_df, loan_df_merged, interaction_df, product_details

customer_df, account_df, loan_df, interaction_df, product_details_dict = generate_data()

st.write("Generated Customer Data Snippet:")
st.dataframe(customer_df.head())
st.write("Generated Account Data Snippet:")
st.dataframe(account_df.head())
st.write("Generated Loan Data (for default prediction) Snippet:")
st.dataframe(loan_df.head())
st.write("Generated Customer-Product Interaction Data Snippet:")
st.dataframe(interaction_df.head())

# For the purpose of this Streamlit app, we will focus on the loan_df for default prediction
# and interaction_df for recommendations, similar to a typical notebook flow.
# The original notebook might have saved and loaded 'bank_data.csv'.
# We will use the generated 'loan_df' as 'df' for the supervised learning part.
df = loan_df.copy()


st.markdown("---")
st.markdown("### Data Exploration (Focusing on Loan Default Data)")
st.write("Using the generated 'Loan Data' for exploration and modeling loan defaults.")

if df.empty:
    st.warning("No loan data was generated. Some sections below might not render correctly.")
else:
    st.markdown("#### Basic Information")
    st.write("`df.head()`")
    st.dataframe(df.head())

    st.write("`df.info()`")
    from io import StringIO
    buffer = StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.write("`df.describe()`")
    st.dataframe(df.describe())

    st.write("`df.isnull().sum()`")
    st.text(df.isnull().sum())

    st.markdown("#### Handling Duplicates")
    st.write("Number of duplicated rows:", df.duplicated().sum())
    if df.duplicated().sum() > 0:
        df.drop_duplicates(inplace=True)
        st.write("Dropped duplicates. New shape:", df.shape)
    else:
        st.write("No duplicates found.")
    st.write("Current shape of the loan dataset:", df.shape)


    st.markdown("---")
    st.markdown("### Univariate Analysis")

    numerical_features = df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = df.select_dtypes(include='object').columns.tolist()
    # Remove CustomerID from numerical features for plotting if it's just an identifier
    if 'CustomerID' in numerical_features:
        numerical_features.remove('CustomerID')
    if 'LoanDefault' in numerical_features: # LoanDefault is target, handle separately if needed
        pass


    st.markdown("#### Numerical Features")
    if numerical_features:
        for col in numerical_features:
            if col != 'LoanDefault': # Don't plot target as general numerical here
                st.markdown(f"**Distribution of {col}**")
                fig, ax = plt.subplots(figsize=(8,4))
                sns.histplot(df[col], kde=True, ax=ax)
                ax.set_title(f'Histogram of {col}')
                show_plot(fig)
    else:
        st.write("No numerical features found for univariate analysis (excluding LoanDefault).")


    st.markdown("#### Categorical Features")
    if categorical_features:
        for col in categorical_features:
            st.markdown(f"**Distribution of {col}**")
            fig, ax = plt.subplots(figsize=(10,5))
            sns.countplot(y=df[col], order = df[col].value_counts().index, ax=ax)
            ax.set_title(f'Count Plot of {col}')
            show_plot(fig)
    else:
        st.write("No categorical features found for univariate analysis.")

    st.markdown("---")
    st.markdown("### Bivariate Analysis")

    st.markdown("#### Numerical vs Numerical")
    st.markdown("**Pairplot of Numerical Features (subset for readability)**")
    # Select a subset of numerical features for pairplot to avoid clutter/performance issues
    pairplot_features = [feat for feat in ['Age', 'Income', 'CreditScore', 'LoanAmount', 'InterestRate', 'LoanDefault'] if feat in df.columns]
    if len(pairplot_features) > 1:
        pair_fig = sns.pairplot(df[pairplot_features], hue='LoanDefault' if 'LoanDefault' in pairplot_features else None, diag_kind='kde')
        st.pyplot(pair_fig)
        plt.clf()
    else:
        st.write("Not enough numerical features for a pairplot.")


    st.markdown("**Correlation Heatmap**")
    if numerical_features and len(numerical_features) > 1: # Ensure there's something to correlate
        # Ensure only numeric types are included for correlation
        numeric_df_for_corr = df[numerical_features + (['LoanDefault'] if 'LoanDefault' in df.columns else [])].copy()
        # If any columns are accidentally object type after selection, drop them
        numeric_df_for_corr = numeric_df_for_corr.select_dtypes(include=np.number)

        if not numeric_df_for_corr.empty and numeric_df_for_corr.shape[1] > 1:
            corr_matrix = numeric_df_for_corr.corr()
            fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
            ax_corr.set_title('Correlation Heatmap of Numerical Features')
            show_plot(fig_corr)
        else:
            st.write("Not enough numerical data to compute a correlation heatmap.")
    else:
        st.write("Not enough numerical features for a correlation heatmap.")


    st.markdown("#### Numerical vs Categorical (LoanDefault vs Numerical Features)")
    target_col = 'LoanDefault'
    if target_col in df.columns:
        # Ensure LoanDefault is treated as categorical for these plots if it's not already object type
        df_copy_for_plot = df.copy()
        df_copy_for_plot[target_col] = df_copy_for_plot[target_col].astype('category')

        for num_col in numerical_features:
            if num_col != target_col: # Avoid plotting LoanDefault against itself if it was in numerical_features
                st.markdown(f"**{num_col} vs {target_col}**")
                fig, ax = plt.subplots(figsize=(8,5))
                sns.boxplot(x=target_col, y=num_col, data=df_copy_for_plot, ax=ax)
                ax.set_title(f'Boxplot of {num_col} by {target_col}')
                show_plot(fig)
    else:
        st.write(f"Target column '{target_col}' not found for Numerical vs Categorical analysis.")


    st.markdown("#### Categorical vs Categorical (Selected Categorical Features vs LoanDefault)")
    if target_col in df.columns and categorical_features:
        for cat_col in categorical_features:
            st.markdown(f"**{cat_col} vs {target_col}**")
            fig, ax = plt.subplots(figsize=(10, 6))
            ct = pd.crosstab(df[cat_col], df[target_col], normalize='index') # normalize for percentage
            ct.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
            ax.set_title(f'Stacked Bar Plot of {target_col} distribution across {cat_col}')
            ax.set_ylabel('Proportion')
            show_plot(fig)
    else:
        st.write(f"Target column '{target_col}' or categorical features not found for Categorical vs Categorical analysis.")


    st.markdown("---")
    st.markdown("### Feature Engineering")
    st.write("Creating new features based on existing ones.")

    # Age Group
    bins_age = [18, 30, 40, 50, 60, 100]
    labels_age = ['18-30', '31-40', '41-50', '51-60', '60+']
    if 'Age' in df.columns:
        df['Age_Group'] = pd.cut(df['Age'], bins=bins_age, labels=labels_age, right=False)

    # Income Category
    if 'Income' in df.columns:
        df['Income_Category'] = pd.qcut(df['Income'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'], duplicates='drop')

    # Credit Score Category
    bins_credit = [300, 580, 670, 740, 800, 851] # Adjusted max to 851 to include 850
    labels_credit = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    if 'CreditScore' in df.columns:
        df['Credit_Score_Category'] = pd.cut(df['CreditScore'], bins=bins_credit, labels=labels_credit, right=False)

    # Loan Amount to Income Ratio
    if 'LoanAmount' in df.columns and 'Income' in df.columns:
        df['Loan_to_Income_Ratio'] = df['LoanAmount'] / (df['Income'] + 1e-6) # Add epsilon to avoid division by zero

    # Years With Bank Category
    bins_ywb = [0, 1, 5, 10, 20, 50]
    labels_ywb = ['<1 Year', '1-5 Years', '6-10 Years', '11-20 Years', '20+ Years']
    if 'YearsWithBank' in df.columns:
        df['Years_With_Bank_Category'] = pd.cut(df['YearsWithBank'], bins=bins_ywb, labels=labels_ywb, right=False)


    st.write("DataFrame with new features (`df.head()`):")
    st.dataframe(df.head())
    
    # Update categorical features list
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()


    st.markdown("---")
    st.markdown("### Data Preprocessing")
    st.markdown("#### Label Encoding for Categorical Features")

    # Make a copy for encoding to keep original df for reference if needed elsewhere
    df_processed = df.copy()
    label_encoders = {}
    original_dtypes = df_processed.dtypes

    for col in categorical_features:
        if col in df_processed.columns: # Ensure column exists
            # Convert to string type to handle mixed types or NaNs gracefully before encoding
            df_processed[col] = df_processed[col].astype(str)
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            label_encoders[col] = le
            st.write(f"Encoded '{col}'. Mappings: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        else:
            st.warning(f"Column {col} not found in df_processed during label encoding.")


    st.write("DataFrame after Label Encoding (`df_processed.head()`):")
    st.dataframe(df_processed.head())


    st.markdown("---")
    st.markdown("### Supervised Learning: Loan Default Prediction")

    if 'LoanDefault' not in df_processed.columns:
        st.error("Target variable 'LoanDefault' not found in the processed data. Cannot proceed with model training.")
    else:
        st.markdown("#### Splitting Data into Features (X) and Target (y)")
        X = df_processed.drop(['LoanDefault', 'CustomerID'], axis=1, errors='ignore')
        y = df_processed['LoanDefault']

        # Ensure all X columns are numeric after encoding
        non_numeric_cols_in_X = X.select_dtypes(exclude=np.number).columns
        if not non_numeric_cols_in_X.empty:
            st.warning(f"Non-numeric columns found in X after encoding: {non_numeric_cols_in_X.tolist()}. Attempting to convert or drop.")
            for col in non_numeric_cols_in_X:
                 try:
                     X[col] = pd.to_numeric(X[col])
                 except ValueError:
                     st.error(f"Could not convert column {col} to numeric. It will be dropped.")
                     X = X.drop(col, axis=1)
        
        st.write("Shape of X:", X.shape)
        st.write("Shape of y:", y.shape)
        st.write("Target variable distribution (LoanDefault):")
        st.write(y.value_counts(normalize=True))


        st.markdown("#### Handling Imbalance Data using SMOTE")
        # Check for severe imbalance
        if y.value_counts().min() < 5 : # SMOTE needs at least k_neighbors samples in minority class (default k_neighbors=5)
            st.warning(f"Minority class has very few samples ({y.value_counts().min()}). SMOTE might not be effective or might fail. Skipping SMOTE.")
            X_resampled, y_resampled = X, y
        else:
            try:
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                st.write("Shape of X after SMOTE:", X_resampled.shape)
                st.write("Shape of y after SMOTE:", y_resampled.shape)
                st.write("Target variable distribution after SMOTE:")
                st.write(pd.Series(y_resampled).value_counts(normalize=True))
            except Exception as e:
                st.error(f"Error during SMOTE: {e}. Using original data.")
                X_resampled, y_resampled = X, y


        st.markdown("#### Train-Test Split")
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=42, stratify=y_resampled if len(np.unique(y_resampled)) > 1 else None)
        st.write("X_train shape:", X_train.shape)
        st.write("X_test shape:", X_test.shape)

        st.markdown("#### Feature Scaling")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store results for comparison
        model_performance = {}

        def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
            st.markdown(f"##### {model_name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            auc_score = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else "N/A"

            st.write(f"Accuracy: {accuracy:.4f}")
            st.write(f"AUC Score: {auc_score if isinstance(auc_score, str) else auc_score:.4f}")
            st.text("Classification Report:")
            st.json(report) # Display report as JSON for better readability

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_title(f'Confusion Matrix - {model_name}')
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')
            show_plot(fig_cm)
            
            return accuracy, report, auc_score


        # --- Logistic Regression ---
        lr_model = LogisticRegression(random_state=42, solver='liblinear', C=0.1) # Added some params from notebook
        acc_lr, _, auc_lr = train_and_evaluate_model(lr_model, X_train_scaled, y_train, X_test_scaled, y_test, "Logistic Regression")
        model_performance["Logistic Regression"] = {'Accuracy': acc_lr, 'AUC': auc_lr}

        # --- Decision Tree ---
        dt_model = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=10) # Added some params
        acc_dt, _, auc_dt = train_and_evaluate_model(dt_model, X_train_scaled, y_train, X_test_scaled, y_test, "Decision Tree Classifier")
        model_performance["Decision Tree"] = {'Accuracy': acc_dt, 'AUC': auc_dt}

        # --- Random Forest ---
        rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, min_samples_leaf=5) # Added some params
        acc_rf, _, auc_rf = train_and_evaluate_model(rf_model, X_train_scaled, y_train, X_test_scaled, y_test, "Random Forest Classifier")
        model_performance["Random Forest"] = {'Accuracy': acc_rf, 'AUC': auc_rf}

        # --- Gradient Boosting ---
        gb_model = GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=3) # Added some params
        acc_gb, _, auc_gb = train_and_evaluate_model(gb_model, X_train_scaled, y_train, X_test_scaled, y_test, "Gradient Boosting Classifier")
        model_performance["Gradient Boosting"] = {'Accuracy': acc_gb, 'AUC': auc_gb}

        st.markdown("#### Model Comparison")
        if model_performance:
            perf_df = pd.DataFrame(model_performance).T.sort_values(by='Accuracy', ascending=False)
            st.dataframe(perf_df)

            fig_comp, ax_comp = plt.subplots(figsize=(10,6))
            perf_df['Accuracy'].plot(kind='bar', ax=ax_comp, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
            ax_comp.set_title('Model Accuracy Comparison')
            ax_comp.set_ylabel('Accuracy')
            ax_comp.tick_params(axis='x', rotation=45)
            show_plot(fig_comp)
        else:
            st.write("No model performance data to compare.")


st.markdown("---")
st.markdown("### Unsupervised Learning: Customer Segmentation")
st.write("Using `customer_df` for segmentation.")

if customer_df.empty:
    st.warning("Customer data is empty. Skipping segmentation.")
else:
    # Select features for segmentation
    segmentation_features = ['Age', 'Income', 'CreditScore', 'YearsWithBank']
    # If any feature from the list is not in customer_df.columns, filter it out
    segmentation_features = [feat for feat in segmentation_features if feat in customer_df.columns]

    if not segmentation_features:
        st.warning("No suitable features found for segmentation in customer_df.")
    else:
        st.write(f"Features selected for segmentation: {segmentation_features}")
        customer_segment_df = customer_df[segmentation_features].copy()
        customer_segment_df.dropna(inplace=True) # Drop NA before scaling

        if customer_segment_df.empty:
            st.warning("DataFrame for segmentation is empty after dropping NA values.")
        else:
            st.markdown("#### Feature Scaling for Segmentation")
            scaler_segment = StandardScaler()
            scaled_segment_data = scaler_segment.fit_transform(customer_segment_df)
            st.write("Data scaled. Shape of scaled data:", scaled_segment_data.shape)

            st.markdown("#### K-Means Clustering: Elbow Method to Find Optimal K")
            inertia = []
            k_range = range(1, 11)
            for k_val in k_range:
                kmeans = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
                kmeans.fit(scaled_segment_data)
                inertia.append(kmeans.inertia_)

            fig_elbow, ax_elbow = plt.subplots(figsize=(8,5))
            ax_elbow.plot(k_range, inertia, marker='o', linestyle='--')
            ax_elbow.set_xlabel('Number of Clusters (K)')
            ax_elbow.set_ylabel('Inertia')
            ax_elbow.set_title('Elbow Method for Optimal K')
            show_plot(fig_elbow)
            st.write("Based on the elbow plot, choose an optimal K (e.g., where the 'elbow' bend is). For this demo, let's pick K=4 (assuming).")
            
            optimal_k = st.slider("Select Optimal K based on Elbow Plot:", min_value=2, max_value=10, value=4, key="kmeans_k")


            st.markdown(f"#### Applying K-Means with K={optimal_k}")
            kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
            customer_segment_df['Cluster'] = kmeans_final.fit_predict(scaled_segment_data)
            st.write(f"`customer_segment_df.head()` with 'Cluster' column:")
            st.dataframe(customer_segment_df.head())

            st.write("Cluster sizes:")
            st.write(customer_segment_df['Cluster'].value_counts())

            st.markdown("#### Visualizing Clusters")
            if len(segmentation_features) >= 2:
                st.markdown(f"**Scatter plot of Clusters (using {segmentation_features[0]} and {segmentation_features[1]})**")
                fig_scatter, ax_scatter = plt.subplots(figsize=(10,7))
                sns.scatterplot(x=segmentation_features[0], y=segmentation_features[1], hue='Cluster', data=customer_segment_df, palette='viridis', ax=ax_scatter, s=50) # s for size
                ax_scatter.set_title(f'Customer Segments based on {segmentation_features[0]} and {segmentation_features[1]}')
                show_plot(fig_scatter)

                # Display cluster means for interpretation
                st.write("Mean values for each feature per cluster:")
                cluster_summary = customer_segment_df.groupby('Cluster')[segmentation_features].mean()
                st.dataframe(cluster_summary)
            else:
                st.write("Need at least two features to create a 2D scatter plot for clusters.")

st.markdown("---")
st.markdown("### Recommendation System")
st.write("Using `interaction_df` (customer-product interactions) for building a recommendation system.")

if interaction_df.empty:
    st.warning("Interaction data is empty. Skipping recommendation system.")
else:
    st.write("Interaction data head:")
    st.dataframe(interaction_df.head())

    # The Surprise library needs data in a specific format (user, item, rating)
    # Our columns are 'uid', 'pid', 'rating'
    reader = Reader(rating_scale=(1, 5)) # Assuming ratings are 1-5 as per data generation
    data_surprise = Dataset.load_from_df(interaction_df[['uid', 'pid', 'rating']], reader)

    st.markdown("#### Train-Test Split for Surprise Dataset")
    trainset, testset = surprise_train_test_split(data_surprise, test_size=0.25, random_state=42)

    st.markdown("#### Using SVD Algorithm")
    algo_svd = SVD(n_factors=50, n_epochs=20, random_state=42) # Parameters from notebook snippet

    st.write("Training SVD model...")
    algo_svd.fit(trainset)
    st.write("Model training complete.")

    st.markdown("#### Evaluating the Model")
    predictions_svd = algo_svd.test(testset)
    rmse_svd = surprise_accuracy.rmse(predictions_svd)
    st.write(f"RMSE on the test set: {rmse_svd:.4f}")

    st.markdown("#### Generating Top-N Recommendations for a User")
    # Example: Get recommendations for a sample user
    # Get a user ID from the interaction_df for demonstration
    if not interaction_df['uid'].empty:
        sample_user_id = interaction_df['uid'].unique()[0]
        st.write(f"Generating recommendations for sample user ID: {sample_user_id}")

        # Get a list of all unique product IDs
        all_product_ids = interaction_df['pid'].unique()

        # Get items the user has already interacted with
        user_interacted_items = interaction_df[interaction_df['uid'] == sample_user_id]['pid'].tolist()
        st.write(f"Products already interacted with by user {sample_user_id}: {user_interacted_items}")

        # Predict ratings for items the user has NOT interacted with
        user_recommendations = []
        for product_id in all_product_ids:
            if product_id not in user_interacted_items:
                predicted_rating = algo_svd.predict(uid=sample_user_id, iid=product_id).est
                user_recommendations.append((product_id, predicted_rating))

        # Sort recommendations by predicted rating
        user_recommendations.sort(key=lambda x: x[1], reverse=True)

        st.write(f"Top 5 product recommendations for user {sample_user_id}:")
        recs_df_list = []
        for rec_pid, rec_rating in user_recommendations[:5]:
            recs_df_list.append({
                'ProductID': rec_pid,
                'ProductName': product_details_dict.get(rec_pid, "N/A"),
                'PredictedRating': round(rec_rating, 3)
            })
        if recs_df_list:
            st.dataframe(pd.DataFrame(recs_df_list))
        else:
            st.write("No new recommendations found for this user (perhaps they've interacted with all products or there are no products left).")
    else:
        st.write("No users in interaction data to generate recommendations for.")

    st.markdown("#### Evaluating Ranking with NDCG@k")
    st.write("The notebook snippet provided an NDCG function. Let's adapt and run it.")
    
    @st.cache_data # Cache the result of NDCG calculation
    def calculate_ndcg_for_streamlit(predictions_surprise_format, k=10):
        # Convert Surprise predictions to a DataFrame format similar to what the NDCG function expects
        # Expected format: DataFrame with 'uid', 'pid', 'prediction' (predicted rating), and needs 'actual_rating'
        
        pred_list = []
        for uid, pid, actual_rating, estimated_rating, _ in predictions_surprise_format:
            pred_list.append({'uid': uid, 'pid': pid, 'prediction': estimated_rating, 'actual_rating': actual_rating})
        
        pred_df = pd.DataFrame(pred_list)

        if pred_df.empty:
            return 0.0

        def ndcg_at_k_user(user_df, k_val):
            # Sort by prediction score
            user_df = user_df.sort_values(by='prediction', ascending=False)
            
            # Relevance is based on actual rating (higher is more relevant)
            # For simplicity, let's consider actual_rating directly as relevance score
            actual_relevance = user_df['actual_rating'].values[:k_val]
            
            # Ideal relevance: sort actual ratings in descending order
            ideal_relevance = np.sort(user_df['actual_rating'].values)[::-1][:k_val]
            
            # DCG
            actual_dcg = np.sum(actual_relevance / np.log2(np.arange(2, len(actual_relevance) + 2)))
            # IDCG
            ideal_dcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))
            
            return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0

        ndcg_scores = pred_df.groupby('uid').apply(lambda df: ndcg_at_k_user(df, k))
        return ndcg_scores.mean() if not ndcg_scores.empty else 0.0

    # Using the predictions from SVD model on the testset
    # This testset should contain actual ratings to compare against.
    if testset: # testset is a list of tuples (uid, iid, r_ui)
        # We need to predict on the testset for NDCG calculation.
        # The 'predictions_svd' object already contains (uid, iid, actual_rating, estimated_rating, details)
        ndcg_score_val = calculate_ndcg_for_streamlit(predictions_svd, k=10)
        st.write(f"Average NDCG@10 for the recommendation system: {ndcg_score_val:.4f}")
        st.markdown("""
        An NDCG score closer to 1.0 indicates strong performance in ranking relevant items higher. This reflects the system's ability to effectively prioritize the most relevant recommendations.
        The actual value depends heavily on the data and model quality.
        """)
    else:
        st.write("Testset is empty, cannot calculate NDCG.")


st.markdown("---")
st.markdown("### End of Notebook Demonstration")
st.info("This Streamlit app demonstrates the key steps from the provided Jupyter Notebook. Some complex visualizations or steps might be simplified for web display.")
