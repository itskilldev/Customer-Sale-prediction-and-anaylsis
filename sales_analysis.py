"""Customer sales analysis & prediction: EDA, visualization, model training, evaluation.

Data source: CSV files in data/ by default. To read from MySQL instead, set USE_MYSQL = True
and fill in MYSQL_URL (needs: pip install sqlalchemy pymysql).

Prediction task: using each customer's PAST behaviour (before a cutoff date), predict how much
they will spend in the FOLLOWING period. This avoids target leakage (order total is just
quantity * price * (1 - discount), so predicting it from those columns would be meaningless).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

USE_MYSQL = False
MYSQL_URL = "mysql+pymysql://user:password@localhost:3306/sales_db"
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

# ---------- 1. Load ----------
if USE_MYSQL:
    from sqlalchemy import create_engine
    eng = create_engine(MYSQL_URL)
    customers = pd.read_sql("SELECT * FROM customers", eng, parse_dates=["signup_date"])
    orders = pd.read_sql("SELECT * FROM orders", eng, parse_dates=["order_date"])
else:
    customers = pd.read_csv("data/customers.csv", parse_dates=["signup_date"])
    orders = pd.read_csv("data/orders.csv", parse_dates=["order_date"])

df = orders.merge(customers, on="customer_id", how="left")

# ---------- 2. Clean & EDA ----------
print("Shape:", df.shape)
print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0])
df = df.drop_duplicates("order_id").dropna(subset=["total_amount"])
print(df[["quantity", "unit_price", "discount", "total_amount"]].describe().round(2))

df["month"] = df["order_date"].dt.to_period("M")
monthly = df.groupby("month")["total_amount"].sum()
by_cat = df.groupby("category")["total_amount"].sum().sort_values()
by_city = df.groupby("city")["total_amount"].sum().sort_values()

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
monthly.plot(ax=ax[0, 0], marker="o", title="Monthly revenue")
by_cat.plot.barh(ax=ax[0, 1], title="Revenue by category")
by_city.plot.barh(ax=ax[1, 0], title="Revenue by city")
df["total_amount"].plot.hist(bins=40, ax=ax[1, 1], title="Order value distribution")
plt.tight_layout(); plt.savefig(OUT / "eda.png", dpi=120); plt.close()

# ---------- 3. Future-spend prediction (no leakage) ----------
# Features come from orders BEFORE the cutoff; target is spend in the 90 days AFTER it.
cutoff = df.order_date.max() - pd.Timedelta(days=90)
past, future = df[df.order_date <= cutoff], df[df.order_date > cutoff]

feats = past.groupby("customer_id").agg(
    orders=("order_id", "count"),
    total_spend=("total_amount", "sum"),
    avg_order=("total_amount", "mean"),
    avg_discount=("discount", "mean"),
    last_order=("order_date", "max"),
    n_categories=("category", "nunique"),
)
feats["recency_days"] = (cutoff - feats.pop("last_order")).dt.days
feats = feats.join(customers.set_index("customer_id")[["age", "gender", "city"]])
feats["tenure_days"] = (cutoff - customers.set_index("customer_id").signup_date.reindex(feats.index)).dt.days
feats["target"] = future.groupby("customer_id")["total_amount"].sum().reindex(feats.index).fillna(0)

X = pd.get_dummies(feats.drop(columns="target"), columns=["gender", "city"], drop_first=True)
y = feats["target"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

models = {"LinearRegression": LinearRegression(),
          "RandomForest": RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                                n_jobs=-1, random_state=42)}
results = {"Baseline (mean)": dict(MAE=mean_absolute_error(yte, np.full(len(yte), ytr.mean())),
                                   RMSE=np.sqrt(mean_squared_error(yte, np.full(len(yte), ytr.mean()))),
                                   R2=0.0)}
for name, m in models.items():
    m.fit(Xtr, ytr); p = m.predict(Xte)
    results[name] = dict(MAE=mean_absolute_error(yte, p),
                         RMSE=np.sqrt(mean_squared_error(yte, p)), R2=r2_score(yte, p))
res = pd.DataFrame(results).T.round(3)
print("\nFuture 90-day spend prediction, test-set evaluation:\n", res)
res.to_csv(OUT / "model_metrics.csv")

rf = models["RandomForest"]
imp = pd.Series(rf.feature_importances_, X.columns).sort_values().tail(10)
imp.plot.barh(title="Top feature importances (RandomForest)", figsize=(8, 5))
plt.tight_layout(); plt.savefig(OUT / "feature_importance.png", dpi=120); plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(yte, rf.predict(Xte), s=8, alpha=0.4)
lim = [0, max(yte.max(), rf.predict(Xte).max())]
plt.plot(lim, lim, "r--")
plt.xlabel("Actual 90-day spend"); plt.ylabel("Predicted"); plt.title("Actual vs predicted")
plt.tight_layout(); plt.savefig(OUT / "actual_vs_predicted.png", dpi=120); plt.close()

# ---------- 4. Customer segmentation (RFM + KMeans) ----------
snap = df.order_date.max() + pd.Timedelta(days=1)
rfm = df.groupby("customer_id").agg(
    recency=("order_date", lambda s: (snap - s.max()).days),
    frequency=("order_id", "count"), monetary=("total_amount", "sum"))
rfm["segment"] = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(
    StandardScaler().fit_transform(rfm))
seg = rfm.groupby("segment").agg(customers=("monetary", "size"), recency=("recency", "mean"),
                                 frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(1)
print("\nCustomer segments:\n", seg)
seg.to_csv(OUT / "customer_segments.csv")
print("\nOutputs saved in", OUT)
