"""Customer Sales Analysis & Prediction : Streamlit app.

Run:  streamlit run app.py
Data: uses data/customers.csv + data/orders.csv by default, or upload your own in the sidebar.
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Sales Analysis & Prediction", page_icon="📊", layout="wide")

DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------- data
@st.cache_data
def load_default():
    c_path, o_path = DATA_DIR / "customers.csv", DATA_DIR / "orders.csv"
    if c_path.exists() and o_path.exists():
        cust = pd.read_csv(c_path, parse_dates=["signup_date"])
        orders = pd.read_csv(o_path, parse_dates=["order_date"])
        return cust, orders
    # No data/ folder (e.g. only the .py files were uploaded to GitHub): generate sample data
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_data import make_data
    return make_data()


def load_uploaded(cust_file, ord_file):
    cust = pd.read_csv(cust_file, parse_dates=["signup_date"])
    orders = pd.read_csv(ord_file, parse_dates=["order_date"])
    return cust, orders


def validate(cust, orders):
    need_c = {"customer_id", "age", "gender", "city", "signup_date"}
    need_o = {"order_id", "customer_id", "order_date", "category", "quantity",
              "unit_price", "discount", "total_amount"}
    miss_c, miss_o = need_c - set(cust.columns), need_o - set(orders.columns)
    if miss_c or miss_o:
        st.error(f"Missing columns. customers: {sorted(miss_c)}  orders: {sorted(miss_o)}")
        st.stop()


@st.cache_data
def merge(cust, orders):
    df = orders.merge(cust, on="customer_id", how="left")
    return df.drop_duplicates("order_id").dropna(subset=["total_amount"])


@st.cache_data
def build_features(df, cust, horizon_days):
    """Features from orders before the cutoff; target = spend in the horizon after it (no leakage)."""
    cutoff = df.order_date.max() - pd.Timedelta(days=horizon_days)
    past, future = df[df.order_date <= cutoff], df[df.order_date > cutoff]
    f = past.groupby("customer_id").agg(
        orders=("order_id", "count"), total_spend=("total_amount", "sum"),
        avg_order=("total_amount", "mean"), avg_discount=("discount", "mean"),
        last_order=("order_date", "max"), n_categories=("category", "nunique"))
    f["recency_days"] = (cutoff - f.pop("last_order")).dt.days
    c = cust.set_index("customer_id")
    f = f.join(c[["age", "gender", "city"]])
    f["tenure_days"] = (cutoff - c.signup_date.reindex(f.index)).dt.days
    f["target"] = future.groupby("customer_id")["total_amount"].sum().reindex(f.index).fillna(0)
    X = pd.get_dummies(f.drop(columns="target"), columns=["gender", "city"], drop_first=True)
    return X, f["target"], cutoff


@st.cache_data
def train_models(X, y, n_trees):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {"Linear Regression": LinearRegression(),
              "Random Forest": RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=5,
                                                     n_jobs=-1, random_state=42)}
    base = np.full(len(yte), ytr.mean())
    rows = {"Baseline (mean)": [mean_absolute_error(yte, base),
                                np.sqrt(mean_squared_error(yte, base)), 0.0]}
    preds = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        preds[name] = p
        rows[name] = [mean_absolute_error(yte, p), np.sqrt(mean_squared_error(yte, p)), r2_score(yte, p)]
    metrics = pd.DataFrame(rows, index=["MAE", "RMSE", "R2"]).T.round(3)
    imp = pd.Series(models["Random Forest"].feature_importances_, X.columns).sort_values()
    return metrics, imp, yte, preds


@st.cache_data
def segment_customers(df, k):
    snap = df.order_date.max() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_id").agg(
        recency=("order_date", lambda s: (snap - s.max()).days),
        frequency=("order_id", "count"), monetary=("total_amount", "sum"))
    rfm["segment"] = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(
        StandardScaler().fit_transform(rfm))
    summary = rfm.groupby("segment").agg(
        customers=("monetary", "size"), avg_recency_days=("recency", "mean"),
        avg_orders=("frequency", "mean"), avg_spend=("monetary", "mean")).round(1)
    return rfm, summary


# ---------------------------------------------------------------- sidebar
st.sidebar.title("📊 Sales Analysis")
src = st.sidebar.radio("Data source", ["Sample data", "Upload CSVs"])
if src == "Upload CSVs":
    cf = st.sidebar.file_uploader("customers.csv", type="csv")
    of = st.sidebar.file_uploader("orders.csv", type="csv")
    if not (cf and of):
        st.info("Upload both files in the sidebar (same columns as the sample data), or switch to Sample data.")
        st.stop()
    cust, orders = load_uploaded(cf, of)
else:
    cust, orders = load_default()
validate(cust, orders)
df = merge(cust, orders)

st.sidebar.markdown("---")
d0, d1 = df.order_date.min().date(), df.order_date.max().date()
rng = st.sidebar.date_input("Date range", (d0, d1), min_value=d0, max_value=d1)
cities = st.sidebar.multiselect("City", sorted(df.city.dropna().unique()))
cats = st.sidebar.multiselect("Category", sorted(df.category.dropna().unique()))

view = df.copy()
if isinstance(rng, tuple) and len(rng) == 2:
    view = view[(view.order_date.dt.date >= rng[0]) & (view.order_date.dt.date <= rng[1])]
if cities:
    view = view[view.city.isin(cities)]
if cats:
    view = view[view.category.isin(cats)]

# ---------------------------------------------------------------- page
st.title("Customer Sales Analysis & Prediction")
st.caption("EDA, visualization, model training and evaluation  ·  Python, Pandas, NumPy, Matplotlib, Scikit-learn")

tab_over, tab_eda, tab_pred, tab_seg, tab_data = st.tabs(
    ["Overview", "EDA", "Prediction", "Customer segments", "Data"])

# ---- Overview
with tab_over:
    if view.empty:
        st.warning("No orders match the current filters.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Revenue", f"{view.total_amount.sum():,.0f}")
        c2.metric("Orders", f"{len(view):,}")
        c3.metric("Customers", f"{view.customer_id.nunique():,}")
        c4.metric("Avg order value", f"{view.total_amount.mean():,.1f}")
        monthly = view.groupby(view.order_date.dt.to_period("M").astype(str))["total_amount"].sum()
        st.subheader("Monthly revenue")
        st.line_chart(monthly)

# ---- EDA
with tab_eda:
    if view.empty:
        st.warning("No orders match the current filters.")
    else:
        a, b = st.columns(2)
        with a:
            st.subheader("Revenue by category")
            st.bar_chart(view.groupby("category").total_amount.sum().sort_values(ascending=False))
        with b:
            st.subheader("Revenue by city")
            st.bar_chart(view.groupby("city").total_amount.sum().sort_values(ascending=False))
        a, b = st.columns(2)
        with a:
            st.subheader("Order value distribution")
            fig, ax = plt.subplots()
            ax.hist(view.total_amount, bins=40)
            ax.set_xlabel("Order value"); ax.set_ylabel("Orders")
            st.pyplot(fig)
        with b:
            st.subheader("Discount vs. average order value")
            st.bar_chart(view.groupby("discount").total_amount.mean())
        st.subheader("Summary statistics")
        st.dataframe(view[["quantity", "unit_price", "discount", "total_amount"]].describe().round(2))

# ---- Prediction
with tab_pred:
    st.subheader("Predict each customer's future spend")
    st.write("Features come from a customer's orders **before** a cutoff date; the model predicts how much "
             "they spend in the days **after** it. (Predicting an order's total from quantity × price would "
             "be target leakage, so that is avoided.) Uses all data, ignoring the sidebar filters.")
    p1, p2 = st.columns(2)
    horizon = p1.slider("Prediction horizon (days)", 30, 180, 90, step=30)
    trees = p2.slider("Random Forest trees", 50, 400, 200, step=50)
    X, y, cutoff = build_features(df, cust, horizon)
    st.caption(f"Cutoff date: {cutoff.date()}  ·  {len(X):,} customers with history before the cutoff")
    metrics, imp, yte, preds = train_models(X, y, trees)
    st.dataframe(metrics)
    st.caption("Lower MAE/RMSE is better; R² of 0 means no better than guessing the average. "
               "Evaluated on a held-out 20% of customers.")
    best = st.selectbox("Model for the charts", list(preds), index=0)
    a, b = st.columns(2)
    with a:
        st.subheader("Actual vs predicted")
        fig, ax = plt.subplots()
        ax.scatter(yte, preds[best], s=8, alpha=0.4)
        lim = [0, max(yte.max(), preds[best].max())]
        ax.plot(lim, lim, "r--")
        ax.set_xlabel("Actual spend"); ax.set_ylabel("Predicted spend")
        st.pyplot(fig)
    with b:
        st.subheader("Top feature importances (Random Forest)")
        st.bar_chart(imp.tail(10))

# ---- Segments
with tab_seg:
    st.subheader("RFM customer segmentation (KMeans)")
    k = st.slider("Number of segments", 2, 8, 4)
    rfm, summary = segment_customers(df, k)
    st.dataframe(summary)
    fig, ax = plt.subplots()
    sc = ax.scatter(rfm.frequency, rfm.monetary, c=rfm.segment, cmap="tab10", s=10, alpha=0.7)
    ax.set_xlabel("Frequency (orders)"); ax.set_ylabel("Monetary (total spend)")
    ax.legend(*sc.legend_elements(), title="Segment")
    st.pyplot(fig)
    st.download_button("Download segments CSV", rfm.to_csv().encode(), "customer_segments.csv", "text/csv")

# ---- Data
with tab_data:
    st.subheader("Filtered orders")
    st.dataframe(view.head(1000))
    st.download_button("Download filtered orders", view.to_csv(index=False).encode(),
                       "filtered_orders.csv", "text/csv")
