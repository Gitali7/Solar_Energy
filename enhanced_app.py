import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle

# Title and Description
st.title("Enhanced Predictive Modeling Application")
st.write("Upload your dataset, select the target variable, and explore predictive results with multiple models and visualizations.")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Handle Missing Values
    st.subheader("Handle Missing Values")
    missing_option = st.radio("Choose how to handle missing values", ["Drop rows", "Fill with mean"])
    if missing_option == "Drop rows":
        data = data.dropna()
    else:
        data = data.fillna(data.mean())
    st.write("Updated Dataset:")
    st.dataframe(data.head())

    # Encode Categorical Variables
    st.subheader("Encode Categorical Variables")
    categorical_columns = data.select_dtypes(include=["object"]).columns
    if len(categorical_columns) > 0:
        st.write("Encoding the following categorical columns:", categorical_columns)
        encoder = LabelEncoder()
        for col in categorical_columns:
            data[col] = encoder.fit_transform(data[col])
    else:
        st.write("No categorical columns found.")

    # Select Target Variable
    target_variable = st.selectbox("Select the target variable", data.columns)
    features = [col for col in data.columns if col != target_variable]

    # Train-Test Split
    test_size = st.slider("Select train-test split ratio (test size)", 0.1, 0.5, 0.2)
    X = data[features]
    y = data[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Model Selection
    st.subheader("Model Selection")
    model_type = st.radio("Select model type", ["Random Forest", "Logistic Regression", "Decision Tree"])
    if model_type == "Random Forest":
        n_estimators = st.slider("Number of estimators", 10, 200, 100)
        max_depth = st.slider("Max depth of the tree", 1, 20, 5)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model_type == "Logistic Regression":
        model = LogisticRegression()
    else:
        max_depth = st.slider("Max depth of the tree", 1, 20, 5)
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

    # Train the Model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Display Results
    st.subheader("Results")
    if model_type in ["Random Forest", "Logistic Regression", "Decision Tree"]:
        accuracy = accuracy_score(y_test, predictions)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, predictions))

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, predictions)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
    else:
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R-squared: {r2:.2f}")

        # Predicted vs Actual Plot
        st.subheader("Predicted vs Actual")
        fig, ax = plt.subplots()
        ax.scatter(y_test, predictions, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

    # Feature Importance (for Random Forest)
    if model_type == "Random Forest":
        st.subheader("Feature Importance")
        feature_importances = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.bar_chart(feature_importances.set_index("Feature"))

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Interactive Scatter Plot
    st.subheader("Interactive Scatter Plot")
    x_axis = st.selectbox("Select X-axis for scatter plot", features)
    y_axis = st.selectbox("Select Y-axis for scatter plot", features)
    scatter_fig = px.scatter(data, x=x_axis, y=y_axis, color=target_variable)
    st.plotly_chart(scatter_fig)

    # Save Trained Model
    st.subheader("Save Trained Model")
    if st.button("Save Model"):
        with open("trained_model.pkl", "wb") as f:
            pickle.dump(model, f)
        st.write("Model saved as `trained_model.pkl`.")

    # Download Predictions
    st.subheader("Download Predictions")
    output = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
    csv = output.to_csv(index=False)
    st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")
