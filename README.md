# Customer Sales Analysis & Prediction

Data analysis and ML model for sales prediction and customer insights.
Includes EDA, visualization, model training and evaluation, now as an interactive **Streamlit** app.

**Tech:** Python, Pandas, NumPy, Matplotlib, Scikit-learn, Streamlit | MySQL, SQL

## Run the Streamlit app
```bash
pip install -r requirements.txt
streamlit run app.py
```
The app opens in your browser (http://localhost:8501).

Tabs: **Overview** (KPIs, monthly revenue) · **EDA** (category/city revenue, distributions) ·
**Prediction** (model comparison, feature importance, actual vs predicted) ·
**Customer segments** (RFM + KMeans) · **Data** (filtered table + download).

Sidebar: choose sample data or upload your own `customers.csv` / `orders.csv`, and filter by date, city and category.

### Using your own data (e.g. from Kaggle)
Upload two CSVs with these columns:
- `customers.csv`: customer_id, age, gender, city, signup_date
- `orders.csv`: order_id, customer_id, order_date, category, quantity, unit_price, discount, total_amount

## Project structure
| File | Purpose |
|---|---|
| `app.py` | Streamlit app |
| `sales_analysis.py` | Same analysis as a plain script (saves charts to `outputs/`) |
| `generate_data.py` | Creates sample data (the app calls it automatically if there is no `data/` folder) |
| `schema.sql` | MySQL tables + example queries |
| `requirements.txt` | Python dependencies |

## Plain script (no Streamlit)
```bash
python sales_analysis.py
```

## Using MySQL
1. Run `schema.sql` and load the CSVs (see the commented `LOAD DATA` lines).
2. `pip install sqlalchemy pymysql`
3. In `sales_analysis.py` set `USE_MYSQL = True` and edit `MYSQL_URL`.

## Prediction approach
Features come from a customer's past orders (before a cutoff date); the model predicts their spend in the
following period. Predicting an order's total from quantity x price would be target leakage, so it is avoided.

## Deploy on Streamlit Community Cloud
Push this folder to GitHub, then create a new app at share.streamlit.io pointing to `app.py`.
