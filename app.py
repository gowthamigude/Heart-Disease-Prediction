import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
def load_data():
    data = pd.read_csv("UCI Heart Disease Dataset.csv")
    return data

data = load_data()

# Streamlit App
st.title("Heart Disease Dataset Explorer")

# Sidebar for user input
st.sidebar.header("Options")

# Show dataset checkbox
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Raw Dataset")
    st.dataframe(data)

# Dataset statistics
if st.sidebar.checkbox("Show dataset statistics"):
    st.subheader("Dataset Statistics")
    st.write(data.describe())

# Visualizations
st.sidebar.subheader("Visualizations")

# Correlation heatmap
if st.sidebar.checkbox("Correlation heatmap"):
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(plt)

# Distribution of age
if st.sidebar.checkbox("Age distribution"):
    st.subheader("Age Distribution")
    plt.figure(figsize=(8, 5))
    sns.histplot(data["age"], kde=True, color="blue", bins=20)
    plt.title("Age Distribution")
    st.pyplot(plt)

# Target distribution
if st.sidebar.checkbox("Target distribution"):
    st.subheader("Target Distribution")
    plt.figure(figsize=(6, 4))
    sns.countplot(x="target", data=data, palette="viridis")
    plt.title("Target Distribution (0 = No Disease, 1 = Disease)")
    st.pyplot(plt)

# Scatter plot: Age vs. Max Heart Rate
if st.sidebar.checkbox("Age vs Max Heart Rate"):
    st.subheader("Age vs. Maximum Heart Rate")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="age", y="thalach", hue="target", data=data, palette="coolwarm")
    plt.title("Age vs Maximum Heart Rate by Target")
    st.pyplot(plt)

st.sidebar.text("Developed with ❤️ using Streamlit")
