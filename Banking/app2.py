import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Set page configuration
st.set_page_config(page_title="Banking Data Science Demo", layout="wide")

# Title
st.title("🏦 Banking Data Science Demo")
st.markdown("An interactive walkthrough of a data science project for banking clients.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_revised1.csv", sep=';')
    return df

df = load_data()

# Section: Data Overview
st.header("1. Data Overview")
st.markdown("Understanding the dataset is crucial. Here's a glimpse of the data:")

st.dataframe(df.head())

st.markdown("### Dataset Summary")
st.write(df.describe())

# Section: Data Preprocessing
st.header("2. Data Preprocessing")
st.markdown("We need to convert categorical variables into numerical ones for modeling.")

# Encode categorical variables
df_encoded = df.copy()
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

st.markdown("Categorical variables have been encoded.")

# Section: Exploratory Data Analysis
st.header("3. Exploratory Data Analysis")
st.markdown("Let's explore the distribution of the target variable.")

fig, ax = plt.subplots()
sns.countplot(x='y', data=df, ax=ax)
st.pyplot(fig)

# Section: Model Building
st.header("4. Model Building")
st.markdown("We'll build a logistic regression model to predict if a client will subscribe to a term deposit.")

# Features and target
X = df_encoded.drop('y', axis=1)
y = df_encoded['y']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Section: Model Evaluation
st.header("5. Model Evaluation")
st.markdown("Evaluating the model's performance on the test set.")

y_pred = model.predict(X_test)

st.markdown("### Classification Report")
st.text(classification_report(y_test, y_pred))

st.markdown("### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# Section: Conclusion
st.header("6. Conclusion")
st.markdown("""
This demo showcased the end-to-end process of a data science project:
- Data loading and overview
- Data preprocessing
- Exploratory data analysis
- Model building
- Model evaluation

Such analyses can provide valuable insights for banking institutions to make informed decisions.
""")
