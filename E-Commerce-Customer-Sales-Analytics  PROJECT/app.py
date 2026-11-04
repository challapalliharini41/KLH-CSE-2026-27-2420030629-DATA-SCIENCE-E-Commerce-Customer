import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="E-Commerce Analytics Studio", page_icon="🛍️", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
.main {background: #f7f9fc;}
.block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
.hero {padding: 1.35rem 1.5rem; border-radius: 18px; color: white;
       background: linear-gradient(115deg,#142c55 0%,#2864a8 58%,#4e8fbd 100%); margin-bottom: 1rem;}
.hero h1 {font-size: 2rem; margin: 0 0 .25rem 0;}
.hero p {margin: 0; opacity: .9; font-size: 1rem;}
.kpi {background: white; border: 1px solid #e5eaf2; border-radius: 14px; padding: 1rem 1.1rem;}
.kpi-label {color:#64748b; font-size:.82rem; font-weight:600;}
.kpi-value {color:#13294b; font-size:1.65rem; font-weight:750; margin-top:.15rem;}
.kpi-note {color:#64748b; font-size:.75rem; margin-top:.1rem;}
.section-note {color:#64748b; font-size:.9rem;}
div[data-testid="stMetric"] {background:white; border:1px solid #e5eaf2; padding:12px 15px; border-radius:12px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🛍️ E-Commerce Customer & Sales Analytics</h1>
  <p>Interactive data science workspace · Customer segmentation · Anomaly detection · Sales trends</p>
</div>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def find_col(df, candidates):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in df.columns:
        cl = str(c).strip().lower()
        if any(cand.lower() in cl for cand in candidates):
            return c
    return None

def standardize_columns(raw):
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    mapping = {
        "order_date": find_col(df, ["Order Date", "order_date", "order date", "date", "purchase date", "transaction date"]),
        "customer": find_col(df, ["Customer ID", "customer_id", "customer id", "customer", "client id"]),
        "sales": find_col(df, ["Sales", "sales amount", "revenue", "amount", "total"]),
        "profit": find_col(df, ["Profit", "profit"]),
        "quantity": find_col(df, ["Quantity", "qty", "units"]),
        "discount": find_col(df, ["Discount", "discount"]),
        "category": find_col(df, ["Category", "product category", "category"]),
        "sub_category": find_col(df, ["Sub-Category", "Sub Category", "subcategory"]),
        "product": find_col(df, ["Product Name", "product", "item name"]),
        "order_id": find_col(df, ["Order ID", "order_id", "order id", "invoice", "transaction id"]),
        "region": find_col(df, ["Region", "region"]),
        "segment": find_col(df, ["Segment", "customer segment"]),
    }
    for target, source in mapping.items():
        if source and source != target:
            df[target] = df[source]
    if "sales" not in df:
        st.error("Could not identify a sales column. Include a column such as Sales, Revenue, or Amount.")
        st.stop()
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    for c in ["profit", "quantity", "discount"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "order_date" in df:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    if "customer" not in df:
        df["customer"] = "Customer-" + (df.index.astype(str))
    if "order_id" not in df:
        df["order_id"] = "Order-" + df.index.astype(str)
    if "profit" not in df:
        df["profit"] = 0.0
    if "quantity" not in df:
        df["quantity"] = 1
    if "discount" not in df:
        df["discount"] = 0.0
    if "category" not in df:
        df["category"] = "Unspecified"
    if "region" not in df:
        df["region"] = "Unspecified"
    df = df.dropna(subset=["sales"]).copy()
    df["customer"] = df["customer"].fillna("Unknown").astype(str)
    df["sales"] = df["sales"].clip(lower=0)
    df["profit"] = df["profit"].fillna(0)
    df["quantity"] = df["quantity"].fillna(1).clip(lower=0)
    df["discount"] = df["discount"].fillna(0)
    df["category"] = df["category"].fillna("Unspecified").astype(str)
    df["region"] = df["region"].fillna("Unspecified").astype(str)
    return df, mapping

def money(v):
    return f"₹{v:,.0f}" if abs(v) < 1e7 else f"₹{v/1e7:,.2f} Cr"

def build_customer_features(df):
    if "order_date" in df and df["order_date"].notna().any():
        anchor = df["order_date"].max() + pd.Timedelta(days=1)
        agg = df.groupby("customer").agg(
            Annual_Spending=("sales", "sum"),
            Purchase_Frequency=("order_id", "nunique"),
            Average_Order_Value=("sales", "mean"),
            Recency_Days=("order_date", lambda s: (anchor - s.max()).days if s.notna().any() else np.nan),
            Total_Profit=("profit", "sum"),
        ).reset_index()
    else:
        agg = df.groupby("customer").agg(
            Annual_Spending=("sales", "sum"),
            Purchase_Frequency=("order_id", "nunique"),
            Average_Order_Value=("sales", "mean"),
            Total_Profit=("profit", "sum"),
        ).reset_index()
        agg["Recency_Days"] = np.nan
    return agg

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Analysis controls")
uploaded = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])
default_path = os.path.join(os.path.dirname(__file__), "data", "sample_superstore.csv")
if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
        dataset_label = uploaded.name
    except Exception as e:
        st.error(f"Unable to read CSV: {e}")
        st.stop()
else:
    raw = pd.read_csv(default_path)
    dataset_label = "Built-in sample Superstore-style dataset"
    st.sidebar.caption("Using the included sample data. Upload your own CSV to analyze your dataset.")

df, original_mapping = standardize_columns(raw)
st.sidebar.markdown("---")
st.sidebar.caption(f"Loaded: **{dataset_label}**")
st.sidebar.caption(f"{len(df):,} transaction rows · {df['customer'].nunique():,} customers")

# Date filters if dates available
if "order_date" in df and df["order_date"].notna().any():
    valid_dates = df["order_date"].dropna()
    dmin, dmax = valid_dates.min().date(), valid_dates.max().date()
    date_range = st.sidebar.date_input("Order date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        df = df[df["order_date"].dt.date.between(date_range[0], date_range[1])]
else:
    st.sidebar.info("Add an Order Date column to enable date filtering.")

regions = sorted(df["region"].dropna().astype(str).unique().tolist())
selected_regions = st.sidebar.multiselect("Region", regions, default=regions)
if selected_regions:
    df = df[df["region"].astype(str).isin(selected_regions)]
categories = sorted(df["category"].dropna().astype(str).unique().tolist())
selected_categories = st.sidebar.multiselect("Category", categories, default=categories)
if selected_categories:
    df = df[df["category"].astype(str).isin(selected_categories)]

if df.empty:
    st.warning("No rows match the selected filters. Widen the date range or select more regions/categories.")
    st.stop()

# ---------- Main KPIs ----------
total_sales = df["sales"].sum()
total_profit = df["profit"].sum()
orders = df["order_id"].nunique()
customers = df["customer"].nunique()
avg_order = total_sales / max(orders, 1)
margin = (total_profit / total_sales * 100) if total_sales else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("💰 Total sales", money(total_sales))
k2.metric("📈 Total profit", money(total_profit))
k3.metric("🧾 Orders", f"{orders:,}")
k4.metric("👥 Customers", f"{customers:,}")
k5.metric("🛒 Avg. order value", money(avg_order))
st.caption(f"Profit margin: **{margin:.1f}%** · Current view reflects all sidebar filters.")

# ---------- Tabs ----------
tab_overview, tab_customers, tab_anomalies, tab_forecast, tab_data = st.tabs([
    "📊 Overview", "👥 Customer Segments", "🚨 Outlier Detection", "🔮 Sales Trends", "🧹 Data Quality & Export"
])

with tab_overview:
    left, right = st.columns([1.55, 1])
    with left:
        st.subheader("Sales over time")
        if "order_date" in df and df["order_date"].notna().any():
            monthly = df.dropna(subset=["order_date"]).set_index("order_date").resample("MS").agg(
                Sales=("sales", "sum"), Profit=("profit", "sum")
            ).reset_index()
            fig = px.line(monthly, x="order_date", y=["Sales", "Profit"], markers=True,
                          labels={"order_date":"Month", "value":"Amount", "variable":"Metric"},
                          title="Monthly sales and profit")
            fig.update_layout(legend_title_text="", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("A valid Order Date column is needed for the monthly trend chart.")
    with right:
        st.subheader("Sales by category")
        cat = df.groupby("category", as_index=False).agg(Sales=("sales","sum"), Profit=("profit","sum")).sort_values("Sales", ascending=False)
        fig = px.bar(cat, x="Sales", y="category", orientation="h", color="Profit",
                     color_continuous_scale="Tealgrn", title="Category performance")
        fig.update_layout(yaxis_title="", xaxis_title="Sales")
        st.plotly_chart(fig, use_container_width=True)
    a, b = st.columns(2)
    with a:
        st.subheader("Sales by region")
        reg = df.groupby("region", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
        st.plotly_chart(px.pie(reg, names="region", values="sales", hole=.48, title="Revenue contribution"), use_container_width=True)
    with b:
        st.subheader("Profit distribution")
        fig = px.histogram(df, x="profit", nbins=35, marginal="box", title="Transaction-level profit")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Key business observations")
    best_cat = cat.iloc[0]["category"] if len(cat) else "N/A"
    best_region = reg.iloc[0]["region"] if len(reg) else "N/A"
    c1, c2, c3 = st.columns(3)
    c1.info(f"**Top sales category:** {best_cat}")
    c2.info(f"**Top region by sales:** {best_region}")
    c3.info(f"**Average profit margin:** {margin:.1f}%")

with tab_customers:
    st.subheader("Customer-level behavioural segmentation")
    features = build_customer_features(df)
    numeric_features = ["Annual_Spending", "Purchase_Frequency", "Average_Order_Value"]
    if len(features) < 3:
        st.warning("At least three distinct customers are required for clustering.")
    else:
        max_k = min(8, len(features)-1)
        default_k = min(6, max_k)
        k = st.slider("Number of customer clusters (K)", min_value=2, max_value=max(2, max_k), value=max(2, default_k))
        X = features[numeric_features].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median()).fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        features["Cluster"] = model.fit_predict(X_scaled).astype(str)
        score = silhouette_score(X_scaled, features["Cluster"]) if len(features) > k and features["Cluster"].nunique() > 1 else np.nan
        m1, m2, m3 = st.columns(3)
        m1.metric("Clusters", k)
        m2.metric("Silhouette score", f"{score:.3f}" if np.isfinite(score) else "N/A")
        m3.metric("Clustered customers", f"{len(features):,}")
        l, r = st.columns([1.3, 1])
        with l:
            fig = px.scatter(features, x="Annual_Spending", y="Average_Order_Value", color="Cluster",
                             size="Purchase_Frequency", hover_name="customer",
                             hover_data={"Purchase_Frequency":True, "Total_Profit":":,.2f"},
                             title="Customer map: spending vs average order value")
            st.plotly_chart(fig, use_container_width=True)
        with r:
            cluster_summary = features.groupby("Cluster").agg(
                Customers=("customer","count"),
                Avg_Spending=("Annual_Spending","mean"),
                Avg_Frequency=("Purchase_Frequency","mean"),
                Avg_Order_Value=("Average_Order_Value","mean"),
                Avg_Profit=("Total_Profit","mean")
            ).reset_index()
            fig = px.bar(cluster_summary, x="Cluster", y="Customers", color="Avg_Spending",
                         color_continuous_scale="Blues", title="Customers per cluster")
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Cluster profile")
        st.dataframe(cluster_summary.style.format({
            "Avg_Spending":"₹{:,.2f}", "Avg_Frequency":"{:,.2f}",
            "Avg_Order_Value":"₹{:,.2f}", "Avg_Profit":"₹{:,.2f}"
        }), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download customer segments", features.to_csv(index=False).encode("utf-8"),
                           "customer_segments.csv", "text/csv")

with tab_anomalies:
    st.subheader("Unusual transaction detection")
    st.markdown("Use statistical IQR bounds or Isolation Forest to flag transactions that deserve review. **A flagged record is not automatically an error or fraud.**")
    method = st.radio("Detection method", ["IQR (interpretable statistical rule)", "Isolation Forest (machine learning)"], horizontal=True)
    metric_options = [c for c in ["sales", "profit", "quantity", "discount"] if c in df.columns]
    chosen_metric = st.selectbox("Primary variable to inspect", metric_options, index=0)
    work = df.copy()
    if method.startswith("IQR"):
        q1, q3 = work[chosen_metric].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        work["Anomaly"] = (work[chosen_metric] < lower) | (work[chosen_metric] > upper)
        st.caption(f"IQR bounds for **{chosen_metric}**: lower = {lower:,.2f}; upper = {upper:,.2f}")
    else:
        cols = ["sales", "profit", "quantity", "discount"]
        Xanom = work[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        contamination = st.slider("Expected anomaly fraction", 0.01, 0.15, 0.05, 0.01)
        iso = IsolationForest(contamination=contamination, random_state=42)
        work["Anomaly"] = iso.fit_predict(Xanom) == -1
        work["Anomaly score"] = -iso.decision_function(Xanom)
    flagged = work[work["Anomaly"]].copy()
    pct = 100 * len(flagged) / max(len(work), 1)
    a, b, c = st.columns(3)
    a.metric("Flagged transactions", f"{len(flagged):,}")
    b.metric("Flagged share", f"{pct:.1f}%")
    c.metric("Sales in flagged rows", money(flagged["sales"].sum()))
    fig = px.scatter(work, x="sales", y=chosen_metric, color=work["Anomaly"].map({True:"Flagged", False:"Typical"}),
                     hover_data=[c for c in ["customer","order_id","category","region"] if c in work],
                     title=f"Transaction outliers: {chosen_metric}", labels={"color":"Status"})
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Flagged records for review**")
    display_cols = [c for c in ["order_date","order_id","customer","category","sales","profit","quantity","discount","region","Anomaly score"] if c in flagged.columns]
    st.dataframe(flagged[display_cols].sort_values(chosen_metric, ascending=False), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download flagged transactions", flagged.to_csv(index=False).encode("utf-8"),
                       "flagged_transactions.csv", "text/csv")

with tab_forecast:
    st.subheader("Monthly sales trend and baseline forecast")
    if "order_date" not in df or not df["order_date"].notna().any():
        st.warning("Sales forecasting requires an Order Date column with valid dates.")
    else:
        monthly = df.dropna(subset=["order_date"]).set_index("order_date")["sales"].resample("MS").sum().reset_index()
        monthly.columns = ["Month", "Sales"]
        monthly = monthly.sort_values("Month").reset_index(drop=True)
        st.caption("This is an educational baseline linear-trend forecast, not a production demand-planning system. Reliable forecasting needs enough history and time-based validation.")
        if len(monthly) < 6:
            st.warning("At least six months of sales history are recommended. Add a larger dataset for more meaningful forecasts.")
        if len(monthly) >= 3:
            monthly["t"] = np.arange(len(monthly))
            horizon = st.slider("Months to forecast", 1, 6, 3)
            train_n = max(2, int(len(monthly) * .8))
            model = LinearRegression().fit(monthly[["t"]].iloc[:train_n], monthly["Sales"].iloc[:train_n])
            test_pred = model.predict(monthly[["t"]].iloc[train_n:]) if train_n < len(monthly) else np.array([])
            if len(test_pred):
                mae = mean_absolute_error(monthly["Sales"].iloc[train_n:], test_pred)
                rmse = np.sqrt(mean_squared_error(monthly["Sales"].iloc[train_n:], test_pred))
                e1, e2 = st.columns(2)
                e1.metric("Holdout MAE", money(mae))
                e2.metric("Holdout RMSE", money(rmse))
            final_model = LinearRegression().fit(monthly[["t"]], monthly["Sales"])
            future_t = np.arange(len(monthly), len(monthly)+horizon)
            future_dates = pd.date_range(monthly["Month"].max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
            future_sales = np.maximum(0, final_model.predict(future_t.reshape(-1,1)))
            hist = monthly[["Month","Sales"]].copy()
            hist["Type"] = "Historical"
            fut = pd.DataFrame({"Month":future_dates, "Sales":future_sales, "Type":"Forecast"})
            combined = pd.concat([hist, fut], ignore_index=True)
            fig = px.line(combined, x="Month", y="Sales", color="Type", markers=True,
                          title="Historical sales and baseline forecast")
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(fut.style.format({"Sales":"₹{:,.2f}"}), use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download forecast", fut.to_csv(index=False).encode("utf-8"), "sales_forecast.csv", "text/csv")
        else:
            st.info("Not enough monthly observations to build a trend model.")

with tab_data:
    st.subheader("Data quality summary")
    quality = pd.DataFrame({
        "Check": ["Rows after filters", "Columns", "Duplicate rows", "Missing cells", "Unique customers", "Unique orders"],
        "Result": [len(df), len(df.columns), int(df.duplicated().sum()), int(df.isna().sum().sum()),
                   int(df["customer"].nunique()), int(df["order_id"].nunique())]
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)
    st.subheader("Preview")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download filtered data", df.to_csv(index=False).encode("utf-8"), "filtered_ecommerce_data.csv", "text/csv")
    st.subheader("Column mapping")
    st.markdown("The app automatically recognizes common column names such as **Order Date, Customer ID, Sales, Profit, Quantity, Discount, Category, Region, Order ID**.")
    st.code("Recommended columns: Order Date, Customer ID, Order ID, Sales, Profit, Quantity, Discount, Category, Region", language="text")

st.markdown("---")
st.caption("E-Commerce Customer & Sales Analytics · Student Data Science Project · Python · Pandas · Scikit-learn · Plotly · Streamlit")
