import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# Set Page Configuration
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# Dashboard Title
st.title("📊 Sales Forecasting & Demand Intelligence System")
st.write("Developed using XGBoost, Isolation Forest and KMeans")

# Load Dataset
sales_df = pd.read_csv("train.csv")
# st.write(sales_df.columns)
# st.write(sales_df.head())
vg_df = pd.read_csv("vgsales.csv")


xgb_model = XGBRegressor()
xgb_model.load_model("xgboost_model.json")

isolation_model = pickle.load(open("isolation_forest.pkl", "rb"))

kmeans_model = pickle.load(open("kmeans.pkl", "rb"))

# Sidebar Navigation
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Sales Overview Dashboard",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments"
    ]
)

# ===========================
# PAGE 1 : SALES OVERVIEW
# ===========================

if page == "Sales Overview Dashboard":

    st.header("📊 Sales Overview Dashboard")

    # Convert Date column
    sales_df["Order Date"] = pd.to_datetime(
    sales_df["Order Date"],
    dayfirst=True,
    errors="coerce"
    )

    # Create Year and Month columns
    sales_df["Year"] = sales_df["Order Date"].dt.year
    sales_df["Month"] = sales_df["Order Date"].dt.to_period("M")

    # ----------------------------
    # Sidebar Filters
    # ----------------------------

    st.subheader("Filters")

    regions = ["All"] + sorted(sales_df["Region"].unique().tolist())
    selected_region = st.selectbox("Select Regions", regions)
    categories = ["All"] + sorted(sales_df["Category"].astype(str).unique().tolist())
    selected_category = st.selectbox("Select Category", categories)

    filtered_df = sales_df.copy()

    if selected_region != "All":
        filtered_df = filtered_df[
            filtered_df["Region"] == selected_region
        ]

    if selected_category != "All":
        filtered_df = filtered_df[
            filtered_df["Category"].astype(str) == selected_category
        ]

    st.markdown("---")
    st.subheader("📊 Total Sales by Year")

    year_sales = (
        filtered_df.groupby("Year")["Sales"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(
        year_sales["Year"].astype(str),
        year_sales["Sales"]
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Total Sales")
    ax.set_title("Total Sales by Year")

    st.pyplot(fig)  # bar chart
    # =====================================================================
    st.markdown("---")
    st.subheader("📈 Monthly Sales Trend")

    monthly_sales = (
        filtered_df.groupby("Month")["Sales"]
        .sum()
        .reset_index()
    )

    monthly_sales["Month"] = monthly_sales["Month"].astype(str)

    fig2, ax2 = plt.subplots(figsize=(10,4))

    ax2.plot(
        monthly_sales["Month"],
        monthly_sales["Sales"],
        marker="o"
    )

    plt.xticks(rotation=45)

    ax2.set_xlabel("Month")
    ax2.set_ylabel("Sales")
    ax2.set_title("Monthly Sales Trend")

    st.pyplot(fig2) #line chart
    # ===========================================================

    st.markdown("---")
    st.subheader("🌍 Sales by Region")

    region_sales = (
        filtered_df.groupby("Region")["Sales"]
        .sum()
        .reset_index()
    )

    fig3, ax3 = plt.subplots(figsize=(7,4))

    ax3.bar(
        region_sales["Region"],
        region_sales["Sales"]
    )

    ax3.set_xlabel("Region")
    ax3.set_ylabel("Sales")
    ax3.set_title("Sales by Region")

    st.pyplot(fig3) #sales by region
    # =====================================================================

    st.markdown("---")
    st.subheader("🛒 Sales by Category")

    category_sales = (
        filtered_df.groupby("Category")["Sales"]
        .sum()
        .reset_index()
    )

    fig4, ax4 = plt.subplots(figsize=(7,4))

    ax4.bar(
        category_sales["Category"],
        category_sales["Sales"]
    )

    ax4.set_xlabel("Category")
    ax4.set_ylabel("Sales")
    ax4.set_title("Sales by Category")

    st.pyplot(fig4) #sale by category

# ==================================================================================

# ===========================
# PAGE 2 : FORECAST EXPLORER
# ===========================

if page == "Forecast Explorer":

    st.header("📈 Forecast Explorer")
    st.markdown("---")

    st.subheader("Forecast Settings")

    regions = ["All"] + sorted(sales_df["Region"].unique().tolist())
    selected_region = st.selectbox(
        "Select Region",
        regions,
        key="forecast_region"
    )

    categories = ["All"] + sorted(
        sales_df["Category"].astype(str).unique().tolist()
    )

    selected_category = st.selectbox(
        "Select Category",
        categories,
        key="forecast_category"
    )

    forecast_month = st.slider(
        "Forecast Horizon (Months)",
        min_value=1,
        max_value=3,
        value=1
    )
    forecast_df = sales_df.copy()

    forecast_df["Order Date"] = pd.to_datetime(
    forecast_df["Order Date"],
    dayfirst=True,
    errors="coerce"
)


    if selected_region != "All":
        forecast_df = forecast_df[
            forecast_df["Region"] == selected_region
        ]

    if selected_category != "All":
        forecast_df = forecast_df[
            forecast_df["Category"] == selected_category
        ]
    # print(forecast_df.columns)
    st.markdown("---")
    st.subheader("Filtered Dataset")

    st.dataframe(forecast_df.head())

    st.markdown("---")
    st.header("🤖 Sales Forecast")

    # Monthly sales
    monthly_sales = (
        forecast_df
        .groupby(pd.Grouper(key="Order Date", freq="M"))["Sales"]
        .sum()
        .reset_index()
    )

    monthly_sales.set_index("Order Date", inplace=True)

    # Create features
    monthly_sales["Lag1"] = monthly_sales["Sales"].shift(1)
    monthly_sales["Lag2"] = monthly_sales["Sales"].shift(2)
    monthly_sales["Lag3"] = monthly_sales["Sales"].shift(3)

    monthly_sales["RollingMean"] = (
        monthly_sales["Sales"]
        .rolling(3)
        .mean()
    )

    monthly_sales["Month"] = monthly_sales.index.month
    monthly_sales["Quarter"] = monthly_sales.index.quarter
    monthly_sales["Season"] = ((monthly_sales.index.month % 12) // 3) + 1

    monthly_sales.dropna(inplace=True)
    if monthly_sales.empty:
     st.warning("Not enough data available for forecasting.")
     st.stop()

    feature_df = monthly_sales[
        [
            "Lag1",
            "Lag2",
            "Lag3",
            "RollingMean",
            "Month",
            "Quarter",
            "Season",
        ]
    ]

    prediction = xgb_model.predict(feature_df)

    monthly_sales["Predicted Sales"] = prediction

    st.subheader("Actual vs Predicted Sales")

    st.dataframe(
        monthly_sales[
            ["Sales", "Predicted Sales"]
    ].round(2)
    )

    fig, ax = plt.subplots(figsize=(13,6))

    ax.plot(
        monthly_sales.index,
        monthly_sales["Sales"],
        label="Actual Sales",
        linewidth=2
    )

    ax.plot(
        monthly_sales.index,
        monthly_sales["Predicted Sales"].round(2),
        label="Predicted Sales",
        linewidth=2
    )

    ax.set_title("Actual vs Predicted Sales")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()

    st.pyplot(fig)
    st.markdown("---")
    st.subheader("Forecast Horizon")

    future_prediction = prediction[-forecast_month:]

    future_dates = pd.date_range(
        monthly_sales.index[-1] + pd.offsets.MonthEnd(),
        periods=forecast_month,
        freq="M"
    )

    forecast_result = pd.DataFrame({
        "Month": future_dates,
        "Forecast Sales": future_prediction.round(2)
    })

    st.dataframe(forecast_result)


# ==========================
# PAGE 3 : ANOMALY REPORT
# ==========================

if page == "Anomaly Report":

    st.header("📉 Anomaly Report")
    st.markdown("---")

    # Convert date (DD/MM/YYYY)
    sales_df["Order Date"] = pd.to_datetime(
        sales_df["Order Date"],
        dayfirst=True
    )

    # Monthly sales
    monthly_sales = (
        sales_df
        .groupby(
            pd.Grouper(key="Order Date", freq="MS")
        )["Sales"]
        .sum()
        .reset_index()
    )

    # Detect anomalies
    monthly_sales["Anomaly"] = isolation_model.predict(
        monthly_sales[["Sales"]]
    )

    anomaly_df = monthly_sales[
        monthly_sales["Anomaly"] == -1
    ]
    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        monthly_sales["Order Date"],
        monthly_sales["Sales"],
        label="Monthly Sales"
    )

    ax.scatter(
        anomaly_df["Order Date"],
        anomaly_df["Sales"],
        color="red",
        s=100,
        label="Anomaly"
    )

    ax.set_title("Sales Anomaly Detection")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()

    st.pyplot(fig)

# table
    st.markdown("---")
    st.subheader("Detected Anomalies")

    st.dataframe(
        anomaly_df[
            ["Order Date", "Sales"]
        ]
    )


# ==========================
# PAGE 4 : PRODUCT DEMAND SEGMENTS
# ==========================

if page == "Product Demand Segments":

    st.header("📦 Product Demand Segments")
    st.markdown("---")
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    cluster_df = (
    sales_df.groupby("Sub-Category")
    .agg(
        TotalSales=("Sales", "sum"),
        Volatility=("Sales", "std"),
        AverageOrder=("Sales", "mean")
       )
    )

    cluster_df["GrowthRate"] = (
        cluster_df["TotalSales"].pct_change()
    )

    cluster_df = cluster_df.fillna(0)

    cluster_df.index.name = "Sub-Category"
    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        cluster_df[
            [
                "TotalSales",
                "GrowthRate",
                "Volatility",
                "AverageOrder"
            ]
        ]
    )

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )
    cluster_df["Cluster"] = kmeans.fit_predict(scaled_data)
    cluster_names = {
        0: "High Volume, Stable Demand",
        1: "Growing Demand",
        2: "Low Volume, High Volatility",
        3: "Declining Demand"
    }

    cluster_df["Demand Type"] = (
        cluster_df["Cluster"].map(cluster_names)
    )
    fig, ax = plt.subplots(figsize=(10,6))

    scatter = ax.scatter(
        cluster_df["TotalSales"],
        cluster_df["AverageOrder"],
        c=cluster_df["Cluster"],
        s=120
    )

    ax.set_xlabel("Total Sales")
    ax.set_ylabel("Average Order")
    ax.set_title("Product Demand Clusters")

    st.pyplot(fig)
    st.markdown("---")
    st.subheader("Demand Cluster Table")

    display_df = cluster_df.reset_index()

    st.dataframe(display_df)