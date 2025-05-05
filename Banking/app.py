#--------------------------------Libraries-------------------------------------------#
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
#------------------------------------------------------------------------------------#

#--------------------------------Page Configuration----------------------------------#
st.set_page_config(page_title='Predictive Analytics and Recommendation Systems in Banking',
                   layout="wide",
                   initial_sidebar_state='expanded')

#------------------------------------------------------------------------------------#

#--------------------------------Sidebar Configuration-------------------------------#
st.sidebar.header('Predictive Analytics and Recommendation Systems in Banking')
st.sidebar.image('logo1.png')
st.sidebar.markdown("---")
menu = st.sidebar.radio(
    "Select an Option",
    ("Introduction", "Exploratory Data Analysis", "Customer Churn Prediction", "Customer Segmentation", "Loan Approval Prediction", "Credit Card Recommendation")
)
st.sidebar.markdown("---")
st.sidebar.write("Developed by Geeth Priya")
#------------------------------------------------------------------------------------#

#--------------------------------Introduction Page-----------------------------------#
if menu == "Introduction":
    st.snow()
    image = Image.open('logo2.png')
    st.image(image)
    st.markdown("<h1 style='text-align: center; color: White;'>Predictive Analytics and Recommendation Systems in Banking</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.header("Introduction")
    st.write("""
    Predictive analytics and recommendation systems are becoming increasingly important in the banking industry. With the vast amount of data available, banks can leverage these technologies to gain valuable insights into customer behavior, predict future trends, and offer personalized recommendations. This project aims to explore various applications of predictive analytics and recommendation systems in banking, including customer churn prediction, customer segmentation, loan approval prediction, and credit card recommendation.
    """)
    st.header("Objective")
    st.write("""
    The main objectives of this project are:
    - To analyze customer data to identify patterns and trends.
    - To build predictive models for customer churn and loan approval.
    - To segment customers based on their behavior and characteristics.
    - To develop a recommendation system for credit cards.
    - To provide actionable insights for banks to improve customer retention, acquisition, and profitability.
    """)
    st.header("Dataset")
    st.write("""
    The datasets used in this project are sourced from Kaggle and contain information about bank customers, including demographics, transaction history, loan details, and credit card usage. These datasets provide a rich source of information for building predictive models and recommendation systems.
    """)

#------------------------------------------------------------------------------------#

#--------------------------------Exploratory Data Analysis Page----------------------#
elif menu == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.markdown("---")

    # Load the dataset
    df = pd.read_csv('bank.csv')

    st.header("Dataset Overview")
    st.write("Shape of the dataset:", df.shape)
    st.write("First 5 rows of the dataset:")
    st.dataframe(df.head())

    st.header("Data Exploration")

    # Dropdown for selecting plots
    plot_choice = st.selectbox(
        "Select a plot to display:",
        (
            "Distribution of Age",
            "Job Distribution",
            "Marital Status Distribution",
            "Education Level Distribution",
            "Default Status Distribution",
            "Housing Loan Status Distribution",
            "Personal Loan Status Distribution",
            "Contact Communication Type Distribution",
            "Month of Last Contact Distribution",
            "Outcome of Previous Marketing Campaign Distribution",
            "Subscription to Term Deposit Distribution (Target Variable)",
            "Age vs Balance",
            "Correlation Heatmap"
        )
    )

    # Display selected plot
    if plot_choice == "Distribution of Age":
        fig = px.histogram(df, x='age', title='Distribution of Age')
        st.plotly_chart(fig, use_container_width=True)
        st.write("Insight: The majority of customers are in the age group of 30-40 years.")

    elif plot_choice == "Job Distribution":
        fig = px.bar(df['job'].value_counts(), title='Job Distribution')
        fig.update_layout(xaxis_title="Job", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        st.write("Insight: 'Management', 'Blue-collar', and 'Technician' are the most common job types.")

    elif plot_choice == "Marital Status Distribution":
        fig = px.pie(df, names='marital', title='Marital Status Distribution')
        st.plotly_chart(fig, use_container_width=True)
        st.write("Insight: Most customers are married.")

    elif plot_choice == "Education Level Distribution":
        fig = px.bar(df['education'].value_counts(), title='Education Level Distribution')
        fig.update_layout(xaxis_title="Education Level", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        st.write("Insight: 'Secondary' education is the most common level among customers.")

    elif plot_choice == "Default Status Distribution":
        fig = px.pie(df, names='default', title='Credit Default Status Distribution')
        st.plotly_chart(fig, use_container_width=True)
        st.write("Insight: Very few customers have credit in default.")

    elif plot_choice == "Housing Loan Status Distribution":
        fig = px.pie(df, names='housing', title='Housing Loan Status Distribution')
        st.plotly_chart(fig, use_container_width=True)
        st.write("Insight: More than half of the customers have a housing loan.")

    elif plot_choice == "Personal Loan Status Distribution":
        fig = px.pie(df, names='loan', title='Personal Loan Status Distribution')
        st.plotly_chart(fig, use_container_width=True)
        st.write("Insight: A small percentage of customers have a personal loan.")

    elif plot_choice == "Contact Communication Type Distribution":
        fig = px.bar(df['contact'].value_counts(), title='Contact Communication Type Distribution')
        fig.update_layout(xaxis_title="Contact Type", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        st.write("Insight: 'Cellular' is the most common method of contact.")

    elif plot_choice == "Month of Last Contact Distribution":
        fig = px.bar(df['month'].value_counts(), title='Month of Last Contact Distribution')
        fig.update_layout(xaxis_title="Month", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        st.write("Insight: May is the month with the highest number of contacts.")

    elif plot_choice == "Outcome of Previous Marketing Campaign Distribution":
        fig = px.bar(df['poutcome'].value_counts(), title='Outcome of Previous Marketing Campaign Distribution')
        fig.update_layout(xaxis_title="Previous Outcome", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        st.write("Insight: The outcome for the majority of previous campaigns is unknown.")

    elif plot_choice == "Subscription to Term Deposit Distribution (Target Variable)":
        fig = px.pie(df, names='deposit', title='Subscription to Term Deposit Distribution')
        st.plotly_chart(fig, use_container_width=True)
        st.write("Insight: The dataset is relatively balanced regarding the target variable ('deposit').")

    elif plot_choice == "Age vs Balance":
        fig = px.scatter(df, x='age', y='balance', title='Age vs Balance')
        st.plotly_chart(fig, use_container_width=True)
        st.write("Insight: Balance tends to increase slightly with age, but there is wide variation.")

    elif plot_choice == "Correlation Heatmap":
        # Convert categorical features to numerical for correlation calculation
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include='object').columns
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le

        # Calculate correlation matrix
        corr_matrix = df_encoded.corr()

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax) # Annot=False for cleaner look
        plt.title('Correlation Heatmap of Features')
        st.pyplot(fig)
        st.write("Insight: Shows the linear relationships between different numerical features. For example, 'pdays' and 'previous' show some correlation.")

    st.markdown("---")
    st.header("Summary of EDA")
    st.write("""
    The exploratory data analysis revealed several key insights:
    - The customer base is diverse in terms of age, job, and education.
    - Most customers are married and have secondary education.
    - Housing loans are common, while personal loans and credit defaults are less frequent.
    - Cellular communication is the primary contact method.
    - The dataset provides a good foundation for building predictive models, although the outcome of previous campaigns is often unknown.
    """)

#------------------------------------------------------------------------------------#

#--------------------------------Customer Churn Prediction Page----------------------#
elif menu == "Customer Churn Prediction":
    st.title("Customer Churn Prediction")
    st.markdown("---")

    # Load the dataset
    df = pd.read_csv('bank.csv') # Assuming 'deposit' column indicates churn ('no' = churn, 'yes' = not churn)
                                 # Let's redefine churn: If they didn't deposit ('no'), maybe they churned.
    df['churn'] = df['deposit'].apply(lambda x: 1 if x == 'no' else 0) # 1 = Churned, 0 = Not Churned

    st.header("Churn Data Overview")
    st.write("Shape of the dataset:", df.shape)
    st.write("Churn distribution:")
    st.write(df['churn'].value_counts())
    fig_churn = px.pie(df, names='churn', title='Customer Churn Distribution (1 = Churned, 0 = Not Churned)')
    st.plotly_chart(fig_churn, use_container_width=True)

    # Feature Engineering and Preprocessing
    st.header("Model Building")
    st.write("Preprocessing data...")

    # Select features and target
    features = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
    target = 'churn'

    X = df[features]
    y = df[target]

    # Encode categorical features
    categorical_cols = X.select_dtypes(include='object').columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Handle potential missing columns after get_dummies if test set differs
    # (Not strictly necessary here as we split after encoding, but good practice)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Scale numerical features
    scaler = StandardScaler()
    # Identify numerical columns after one-hot encoding
    numerical_cols = X.select_dtypes(include=np.number).columns
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    st.write(f"Training data shape: {X_train.shape}")
    st.write(f"Test data shape: {X_test.shape}")

    # Model Selection
    model_choice = st.selectbox(
        "Select a Classification Model:",
        ("Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine", "Naive Bayes", "Decision Tree", "Random Forest")
    )

    # Train and Evaluate Model
    @st.cache_data # Cache the model training
    def train_model(model_name, X_train, y_train):
        if model_name == "Logistic Regression":
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_name == "K-Nearest Neighbors":
            model = KNeighborsClassifier() # Default neighbors=5
        elif model_name == "Support Vector Machine":
            model = SVC(random_state=42, probability=True) # Probability needed for potential threshold tuning
        elif model_name == "Naive Bayes":
            model = GaussianNB()
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        else:
            return None

        model.fit(X_train, y_train)
        return model

    model = train_model(model_choice, X_train, y_train)

    if model:
        st.write(f"Training {model_choice} model...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model trained successfully!")

        st.subheader("Model Performance")
        st.write(f"Accuracy: {accuracy:.4f}")

        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churned', 'Churned'], yticklabels=['Not Churned', 'Churned'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(fig_cm)

        # Feature Importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            importances = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
            importances = importances.sort_values('importance', ascending=False).head(15) # Show top 15
            fig_imp = px.bar(importances, x='importance', y='feature', orientation='h', title='Top 15 Feature Importances')
            st.plotly_chart(fig_imp, use_container_width=True)

    else:
        st.error("Please select a valid model.")

    st.markdown("---")
    st.header("Conclusion")
    st.write("""
    The customer churn prediction model helps identify customers who are likely to stop using the bank's services (in this case, interpreted as not subscribing to a term deposit). By understanding the factors driving churn (feature importance), banks can implement targeted retention strategies to improve customer loyalty. Different models show varying performance, highlighting the importance of model selection and tuning.
    """)

#------------------------------------------------------------------------------------#

#--------------------------------Customer Segmentation Page--------------------------#
elif menu == "Customer Segmentation":
    st.title("Customer Segmentation using K-Means Clustering")
    st.markdown("---")

    # Load the clustered data
    try:
        df1 = pd.read_csv('Clustered_Customer_Data.csv')
        st.header("Clustered Data Overview")
        st.write("Shape of the dataset:", df1.shape)
        st.write("First 5 rows of the clustered data:")
        st.dataframe(df1.head())

        # Check if 'Cluster' column exists
        if 'Cluster' not in df1.columns:
            st.error("The loaded CSV file 'Clustered_Customer_Data.csv' does not contain the 'Cluster' column. Please ensure the clustering was performed and saved correctly.")
        else:
            st.header("Cluster Analysis")
            st.write("Number of customers in each cluster:")
            st.write(df1['Cluster'].value_counts())

            fig_cluster_pie = px.pie(df1, names='Cluster', title='Customer Distribution Across Clusters')
            st.plotly_chart(fig_cluster_pie, use_container_width=True)

            st.subheader("Visualizing Clusters")
            st.write("Select features to visualize the clusters:")

            # Ensure columns exist before offering them as choices
            available_cols = [col for col in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'] if col in df1.columns]
            if len(available_cols) < 2:
                 st.warning("Not enough numerical columns found in the data for 2D scatter plot visualization (Need at least 'Age', 'Annual Income (k$)', 'Spending Score (1-100)').")
            else:
                x_axis = st.selectbox("Select X-axis feature:", available_cols, index=0)
                remaining_cols = [col for col in available_cols if col != x_axis]
                y_axis = st.selectbox("Select Y-axis feature:", remaining_cols, index=0 if len(remaining_cols) > 0 else None)

                if x_axis and y_axis:
                    fig_scatter = px.scatter(df1, x=x_axis, y=y_axis, color='Cluster',
                                             title=f'Customer Segments based on {x_axis} and {y_axis}',
                                             labels={'Cluster': 'Customer Segment'},
                                             hover_data=['Gender'] if 'Gender' in df1.columns else None)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.warning("Please select both X and Y axes for visualization.")


            st.header("Cluster Profiles (Example based on typical K-Means results)")
            st.write("""
            Based on the clustering (typically performed on features like age, income, spending score), we can infer profiles:
            - **Cluster 0 (e.g., Target):** Often represents customers with high income and high spending scores. Prime candidates for premium products/services.
            - **Cluster 1 (e.g., Careful):** Might include customers with low income and low spending scores. Focus on basic services, potentially financial education.
            - **Cluster 2 (e.g., Standard):** Could be the average customer group with moderate income and spending. Standard marketing campaigns might work well.
            - **Cluster 3 (e.g., Potential Spenders):** Customers with high income but low spending scores. Potential for targeted offers to increase spending.
            - **Cluster 4 (e.g., Young Spenders):** Might represent younger customers with lower income but high spending scores. Focus on digital services, credit products.

            *(Note: The exact interpretation depends on the features used for clustering and the resulting cluster characteristics in 'Clustered_Customer_Data.csv'. Analyze the means/medians of features for each cluster in your specific data.)*
            """)

            # Allow users to see statistics per cluster
            st.subheader("View Cluster Statistics")
            cluster_select = st.selectbox("Select Cluster to view statistics:", sorted(df1['Cluster'].unique()))
            cluster_data = df1[df1['Cluster'] == cluster_select]
            st.write(f"Descriptive Statistics for Cluster {cluster_select}:")
            st.dataframe(cluster_data.describe())


    except FileNotFoundError:
        st.error("Error: 'Clustered_Customer_Data.csv' not found. Please make sure the file exists in the same directory as the script.")
    except Exception as e:
        st.error(f"An error occurred while loading or processing the cluster data: {e}")


    st.markdown("---")
    st.header("Conclusion")
    st.write("""
    Customer segmentation using K-Means clustering helps banks divide their customer base into distinct groups with similar characteristics. This allows for more targeted marketing campaigns, personalized product offerings, and tailored customer service strategies, ultimately leading to improved customer satisfaction and profitability.
    """)

#------------------------------------------------------------------------------------#

#--------------------------------Loan Approval Prediction Page-----------------------#
elif menu == "Loan Approval Prediction":
    st.title("Loan Approval Prediction")
    st.markdown("---")

    # Load the dataset
    try:
        df2 = pd.read_csv('bank_data.csv') # Assuming this is the loan dataset
        st.header("Loan Data Overview")
        st.write("Shape of the dataset:", df2.shape)
        st.write("First 5 rows:")
        st.dataframe(df2.head())

        # Basic Preprocessing (handle missing values, encode categoricals)
        st.header("Data Preprocessing")
        # Drop Loan_ID as it's not a predictive feature
        if 'Loan_ID' in df2.columns:
            df2 = df2.drop('Loan_ID', axis=1)
            st.write("Dropped 'Loan_ID' column.")

        # Handle Missing Values (Simple Imputation)
        st.write("Handling missing values using forward fill (simple strategy):")
        # Identify columns with missing values
        missing_cols = df2.columns[df2.isnull().any()].tolist()
        if missing_cols:
            st.write(f"Columns with missing values: {missing_cols}")
            df2.fillna(method='ffill', inplace=True)
            # Check if any missing values remain (ffill might not fill initial NaNs)
            df2.fillna(method='bfill', inplace=True) # Backfill for any remaining
            st.write("Missing values handled.")
            # Verify
            if df2.isnull().sum().sum() == 0:
                st.success("No missing values remaining.")
            else:
                st.warning(f"Still {df2.isnull().sum().sum()} missing values remaining after ffill/bfill. Consider more robust imputation.")
        else:
            st.write("No missing values found.")


        # Encode Categorical Features
        st.write("Encoding categorical features using Label Encoding:")
        categorical_cols = df2.select_dtypes(include='object').columns
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df2[col] = le.fit_transform(df2[col])
            label_encoders[col] = le
            st.write(f"Encoded '{col}'. Mappings: {dict(zip(le.classes_, le.transform(le.classes_)))}")

        st.write("Preprocessed Data Head:")
        st.dataframe(df2.head())

        # Model Building
        st.header("Model Building for Loan Approval")
        if 'Loan_Status' not in df2.columns:
            st.error("Target variable 'Loan_Status' not found in the dataset.")
        else:
            # Define features (X) and target (y)
            X = df2.drop('Loan_Status', axis=1)
            y = df2['Loan_Status'] # Assuming 'Y'/'N' encoded to 1/0

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            # Scale numerical features (Important for many models)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            st.write(f"Training data shape: {X_train.shape}")
            st.write(f"Test data shape: {X_test.shape}")

            # Model Selection
            model_choice_loan = st.selectbox(
                "Select a Classification Model for Loan Approval:",
                ("Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine", "Naive Bayes", "Decision Tree", "Random Forest"),
                key="loan_model_select" # Unique key for this selectbox
            )

            # Train and Evaluate Model
            @st.cache_data # Cache the model training
            def train_loan_model(model_name, X_train, y_train):
                if model_name == "Logistic Regression":
                    model = LogisticRegression(random_state=42)
                elif model_name == "K-Nearest Neighbors":
                    model = KNeighborsClassifier()
                elif model_name == "Support Vector Machine":
                    model = SVC(random_state=42, probability=True)
                elif model_name == "Naive Bayes":
                    model = GaussianNB()
                elif model_name == "Decision Tree":
                    model = DecisionTreeClassifier(random_state=42)
                elif model_name == "Random Forest":
                    model = RandomForestClassifier(random_state=42)
                else:
                    return None

                model.fit(X_train, y_train)
                return model

            loan_model = train_loan_model(model_choice_loan, X_train, y_train)

            if loan_model:
                st.write(f"Training {model_choice_loan} model...")
                y_pred_loan = loan_model.predict(X_test)
                accuracy_loan = accuracy_score(y_test, y_pred_loan)
                st.write(f"Model trained successfully!")

                st.subheader("Model Performance")
                st.write(f"Accuracy: {accuracy_loan:.4f}")

                st.subheader("Classification Report")
                # Get class names from the label encoder used for Loan_Status
                status_encoder = label_encoders.get('Loan_Status')
                target_names = status_encoder.classes_ if status_encoder else ['Rejected (0)', 'Approved (1)']
                report_loan = classification_report(y_test, y_pred_loan, target_names=target_names, output_dict=True)
                st.dataframe(pd.DataFrame(report_loan).transpose())

                st.subheader("Confusion Matrix")
                cm_loan = confusion_matrix(y_test, y_pred_loan)
                fig_cm_loan = plt.figure(figsize=(6, 4))
                sns.heatmap(cm_loan, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix - Loan Approval')
                st.pyplot(fig_cm_loan)

                # Feature Importance (for tree-based models)
                if hasattr(loan_model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    importances_loan = pd.DataFrame({'feature': X.columns, 'importance': loan_model.feature_importances_})
                    importances_loan = importances_loan.sort_values('importance', ascending=False).head(10) # Top 10
                    fig_imp_loan = px.bar(importances_loan, x='importance', y='feature', orientation='h', title='Top 10 Feature Importances for Loan Approval')
                    st.plotly_chart(fig_imp_loan, use_container_width=True)
                    st.write("Insight: Features like 'Credit_History', 'LoanAmount', and 'ApplicantIncome' often play significant roles in loan approval decisions.")


            else:
                st.error("Please select a valid model.")

    except FileNotFoundError:
        st.error("Error: 'bank_data.csv' not found. Please ensure the file exists in the same directory.")
    except KeyError as e:
        st.error(f"Error: Column {e} not found in the dataset. Please check the CSV file structure.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    st.markdown("---")
    st.header("Conclusion")
    st.write("""
    The loan approval prediction model assists banks in automating and standardizing the loan application process. By analyzing applicant details and financial history, the model predicts the likelihood of loan approval, helping banks make faster, more consistent decisions and manage risk effectively. Features like credit history are often strong predictors.
    """)

#------------------------------------------------------------------------------------#

#--------------------------------Credit Card Recommendation Page---------------------#
elif menu == "Credit Card Recommendation":
    st.title("Credit Card Recommendation")
    st.markdown("---")
    st.write("This section demonstrates a simple rule-based recommendation system based on user inputs.")

    # Load a base dataset for context (optional, could use segmentation data)
    try:
        data = pd.read_csv('bank_cleaned.csv') # Using cleaned bank data as an example
        st.header("Customer Data Sample (for context)")
        st.dataframe(data.head())
    except FileNotFoundError:
        st.warning("'bank_cleaned.csv' not found. Proceeding without sample data.")
    except Exception as e:
        st.warning(f"Could not load 'bank_cleaned.csv': {e}")


    # Load the pre-trained model, scaler and label encoder
    try:
        model = pickle.load(open('model.pkl', 'rb'))
        le = pickle.load(open('le.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        st.success("Recommendation model and preprocessors loaded successfully.")

        st.header("Enter Customer Details for Recommendation")

        # Input fields based on the features used for training the model
        # IMPORTANT: These must match the columns used when training 'model.pkl'
        # Based on common bank data:
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            # Assuming 'job' was label encoded during training
            job_options = list(le.classes_) # Get job types from the loaded LabelEncoder
            job = st.selectbox("Job", options=job_options, index=job_options.index('management') if 'management' in job_options else 0) # Default to management
            balance = st.number_input("Account Balance", value=1500)
            duration = st.number_input("Last Contact Duration (seconds)", value=200)

        with col2:
            campaign = st.number_input("Number of Contacts (This Campaign)", value=2, min_value=1)
            pdays = st.number_input("Days Since Last Contact (Previous Campaign, -1 if not contacted)", value=-1)
            previous = st.number_input("Number of Contacts (Previous Campaign)", value=0)
            # Add other relevant features the model was trained on
            # Example: housing loan (yes/no), personal loan (yes/no) - Need encoding consistency
            housing = st.radio("Has Housing Loan?", ('yes', 'no'), index=1)
            loan = st.radio("Has Personal Loan?", ('yes', 'no'), index=1)


        # Predict Button
        if st.button("Get Credit Card Recommendation"):
            # Prepare input data for prediction - MUST match training structure
            try:
                # Create a DataFrame with the input
                input_data = pd.DataFrame({
                    'age': [age],
                    'job': [job],
                    'balance': [balance],
                    'housing': [housing],
                    'loan': [loan],
                    'duration': [duration],
                    'campaign': [campaign],
                    'pdays': [pdays],
                    'previous': [previous]
                    # Add ALL other features the model expects, in the correct order!
                    # This requires knowing exactly how 'model.pkl' was trained.
                    # Assuming these were the key features based on common datasets.
                    # We need to add placeholders or handle missing ones if necessary.
                    # Example: If 'marital', 'education', 'default', 'contact', 'month', 'poutcome' were used:
                    # Need inputs for these too, or assume defaults/impute.
                    # For simplicity, let's assume the model only used the features above.
                    # THIS IS A CRITICAL POINT - INPUT MUST MATCH TRAINING DATA STRUCTURE
                })

                st.subheader("Processing Input:")
                st.dataframe(input_data)

                # Apply the same preprocessing as during training
                # 1. Label Encode 'job'
                input_data['job'] = le.transform(input_data['job'])

                # 2. Encode binary 'yes'/'no' features (assuming 1 for yes, 0 for no during training)
                input_data['housing'] = input_data['housing'].apply(lambda x: 1 if x == 'yes' else 0)
                input_data['loan'] = input_data['loan'].apply(lambda x: 1 if x == 'yes' else 0)

                # 3. Scale numerical features using the loaded scaler
                # Ensure columns match exactly what the scaler expects
                numerical_cols_for_scaling = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous'] # Example
                # Verify these columns exist in input_data
                if all(col in input_data.columns for col in numerical_cols_for_scaling):
                     # Select only the columns the scaler was trained on, in the correct order
                     # This requires knowing the scaler's feature names or order
                     # Assuming the scaler was fit on these columns in this order:
                    input_data_scaled = scaler.transform(input_data[numerical_cols_for_scaling])
                    # Create a DataFrame with scaled data and correct column names
                    input_scaled_df = pd.DataFrame(input_data_scaled, columns=numerical_cols_for_scaling, index=input_data.index)

                    # Combine scaled numerical with encoded categorical
                    # Ensure correct final feature set and order for the model
                    # Example: If model expects 'age', 'job', 'balance', 'housing', 'loan', 'duration', 'campaign', 'pdays', 'previous'
                    final_input_features = pd.concat([
                        input_scaled_df[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']], # Scaled numerical
                        input_data[['job', 'housing', 'loan']] # Encoded categorical
                    ], axis=1)

                    # Reorder columns to match the exact order the model was trained on
                    # THIS IS CRUCIAL and requires knowledge of the training process.
                    # Assuming the order was: age, job, balance, housing, loan, duration, campaign, pdays, previous
                    final_input_features = final_input_features[['age', 'job', 'balance', 'housing', 'loan', 'duration', 'campaign', 'pdays', 'previous']]

                    st.subheader("Preprocessed Data for Model:")
                    st.dataframe(final_input_features)


                    # Make prediction (predicting probability of subscribing 'yes')
                    prediction_proba = model.predict_proba(final_input_features)[:, 1] # Probability of class '1' (e.g., 'yes' deposit)
                    prediction = model.predict(final_input_features)

                    st.subheader("Recommendation Result")
                    st.write(f"Model Prediction (e.g., Likely to Subscribe to Deposit?): {'Yes' if prediction[0] == 1 else 'No'}")
                    st.write(f"Probability of Subscribing: {prediction_proba[0]:.2f}")

                    # Simple Rule-Based Recommendation based on prediction/probability
                    if prediction_proba[0] > 0.7: # High probability of 'yes' (e.g., likely engaged customer)
                        st.success("**Recommendation:** Offer **Premium Rewards Card** (High likelihood of engagement/value)")
                        st.image('card_premium.png', width=200) # Placeholder image name
                    elif prediction_proba[0] > 0.4: # Moderate probability
                         st.info("**Recommendation:** Offer **Standard Cashback Card** (Good potential customer)")
                         st.image('card_standard.png', width=200) # Placeholder image name
                    else: # Low probability
                        st.warning("**Recommendation:** Offer **Basic Low-Interest Card** or focus on other services (Lower engagement predicted)")
                        st.image('card_basic.png', width=200) # Placeholder image name

                else:
                     st.error("Mismatch between input features and columns expected by the scaler. Cannot proceed.")


            except FileNotFoundError:
                 st.error("Error: Model or preprocessor file ('model.pkl', 'le.pkl', 'scaler.pkl') not found.")
            except KeyError as e:
                 st.error(f"Error during preprocessing: Feature {e} not found in input or expected by preprocessor. Ensure all required inputs are provided and match training.")
            except ValueError as e:
                 st.error(f"Error during prediction: {e}. This often indicates a mismatch between the input data structure/values and what the model expects (e.g., unseen category in label encoder, wrong number/order of features).")
            except Exception as e:
                 st.error(f"An unexpected error occurred during recommendation: {e}")

    except FileNotFoundError:
        st.error("Error: Could not load recommendation model files ('model.pkl', 'le.pkl', 'scaler.pkl'). Please ensure these files are present.")
    except Exception as e:
        st.error(f"An error occurred loading the recommendation model components: {e}")


    st.markdown("---")
    st.header("Conclusion")
    st.write("""
    This simple recommendation system uses a predictive model (originally trained for deposit subscription) as a proxy for customer engagement or value. Based on the predicted probability, it suggests different types of credit cards. A more sophisticated system would involve collaborative filtering, content-based filtering, or models specifically trained on credit card adoption data, considering factors like spending habits, credit score, and income more directly.
    """)
#------------------------------------------------------------------------------------#
