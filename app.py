# ======================================================
# PROFESSIONAL HOUSING ML DASHBOARD
# Supervised + Unsupervised Learning
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import r2_score, silhouette_score

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="Housing ML Dashboard", layout="wide")
st.title("🏠 Housing Price Prediction & Clustering Dashboard")

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Housing.csv")
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ======================================================
# PREPROCESSING
# ======================================================
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("price", axis=1)
y = df_encoded["price"]

X_cluster = df_encoded.copy()

# ======================================================
# TRAIN MODELS
# ======================================================

# Supervised
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Unsupervised
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataframe
df["KMeans_Cluster"] = kmeans_labels

# Create cluster summary
cluster_summary = df.groupby("KMeans_Cluster").mean(numeric_only=True)

# Sort clusters by average price
cluster_price_rank = cluster_summary["price"].sort_values().index.tolist()

# Assign meaningful names
cluster_names = {}

cluster_names[cluster_price_rank[0]] = "Budget Homes"
cluster_names[cluster_price_rank[1]] = "Mid-Range Homes"
cluster_names[cluster_price_rank[2]] = "Premium Homes"

agg_model = AgglomerativeClustering(n_clusters=3)
agg_labels = agg_model.fit_predict(X_scaled)

# ======================================================
# MODEL PERFORMANCE SECTION
# ======================================================
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Linear Regression")
    lr_r2 = r2_score(y_test, lr_model.predict(X_test))
    st.write("R2 Score:", lr_r2)

with col2:
    st.markdown("### Random Forest")
    rf_r2 = r2_score(y_test, rf_model.predict(X_test))
    st.write("R2 Score:", rf_r2)

st.markdown("### Clustering Performance")
st.write("KMeans Silhouette Score:", silhouette_score(X_scaled, kmeans_labels))
st.write("Hierarchical Silhouette Score:", silhouette_score(X_scaled, agg_labels))

# ======================================================
# NEW INPUT SECTION
# ======================================================
st.subheader("🔍 Analyze New House")

input_data = {}

for column in X.columns:
    input_data[column] = st.number_input(f"{column}", value=0.0)

if st.button("Analyze House"):

    new_df = pd.DataFrame([input_data])

    # Align with training columns
    new_df = new_df.reindex(columns=X.columns, fill_value=0)

    # ------------------
    # Price Prediction
    # ------------------
    lr_price = lr_model.predict(new_df)[0]
    rf_price = rf_model.predict(new_df)[0]

    # ------------------
    # Cluster Prediction
    # ------------------
    new_full = new_df.copy()
    new_full["price"] = rf_price
    new_full = new_full.reindex(columns=X_cluster.columns, fill_value=0)

    new_scaled = scaler.transform(new_full)
    cluster_label = kmeans.predict(new_scaled)[0]
    cluster_type = cluster_names[cluster_label]

    # ==================================================
    # RESULTS DISPLAY
    # ==================================================
    st.success("✅ Analysis Complete")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Linear Regression Price", f"{lr_price:,.2f}")

    with col2:
        st.metric("Random Forest Price", f"{rf_price:,.2f}")

    with col3:
        st.metric("Cluster Number", cluster_label)

    st.markdown("### 🏷 Cluster Classification")
    st.info(f"This house belongs to: **{cluster_type}**")

    # Show cluster characteristics
    st.markdown("### 📊 Cluster Characteristics")

    cluster_info = cluster_summary.loc[cluster_label]

    st.write("Average Price:", round(cluster_info["price"], 2))

    if "area" in cluster_info:
        st.write("Average Area:", round(cluster_info["area"], 2))

    if "bedrooms" in cluster_info:
        st.write("Average Bedrooms:", round(cluster_info["bedrooms"], 2))

# ======================================================
# VISUALIZATION SECTION
# ======================================================
st.subheader("📈 Visualization")

fig = plt.figure()
plt.scatter(df["area"], df["price"], c=kmeans_labels)
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("KMeans Clustering (Area vs Price)")
st.pyplot(fig)

# ======================================================
# FOOTER
# ======================================================
