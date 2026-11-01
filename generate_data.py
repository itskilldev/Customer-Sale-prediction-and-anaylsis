"""Generate a synthetic customers/orders dataset (use your own data if you have it).

- `make_data()` returns (customers, orders) DataFrames; app.py uses it when no data/ folder exists.
- Running this file saves them as CSVs in data/.
"""
import numpy as np
import pandas as pd
from pathlib import Path


def make_data(n_cust=1500, n_ord=12000, seed=42):
    rng = np.random.default_rng(seed)
    cities = ["Lahore", "Karachi", "Islamabad", "Faisalabad", "Multan", "Peshawar"]
    cats = {"Electronics": 250, "Clothing": 40, "Grocery": 15, "Home": 80, "Sports": 60}

    customers = pd.DataFrame({
        "customer_id": np.arange(1, n_cust + 1),
        "age": rng.integers(18, 70, n_cust),
        "gender": rng.choice(["Male", "Female"], n_cust),
        "city": rng.choice(cities, n_cust),
        "signup_date": pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 700, n_cust), unit="D"),
    })

    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 660, n_ord), unit="D")
    category = rng.choice(list(cats), n_ord)
    base = np.array([cats[c] for c in category])
    unit_price = np.round(base * rng.uniform(0.7, 1.4, n_ord), 2)
    quantity = rng.integers(1, 6, n_ord)
    discount = rng.choice([0, 0.05, 0.1, 0.2], n_ord, p=[0.5, 0.2, 0.2, 0.1])
    season = 1 + 0.25 * np.isin(dates.month, [11, 12])  # festive-season bump

    # customers differ in how often they buy (some loyal, many occasional)
    w = rng.gamma(0.8, 1.0, n_cust)
    orders = pd.DataFrame({
        "order_id": np.arange(1, n_ord + 1),
        "customer_id": rng.choice(customers.customer_id, n_ord, p=w / w.sum()),
        "order_date": dates,
        "category": category,
        "quantity": quantity,
        "unit_price": unit_price,
        "discount": discount,
    })
    orders["total_amount"] = np.round(
        orders.quantity * orders.unit_price * (1 - orders.discount) * season, 2)
    return customers, orders.sort_values("order_date").reset_index(drop=True)


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    customers, orders = make_data()
    customers.to_csv("data/customers.csv", index=False)
    orders.to_csv("data/orders.csv", index=False)
    print("Saved", len(customers), "customers and", len(orders), "orders to data/")
