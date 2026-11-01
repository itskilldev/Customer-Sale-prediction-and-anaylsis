-- Customer Sales Analysis & Prediction : MySQL schema
CREATE DATABASE IF NOT EXISTS sales_db;
USE sales_db;

CREATE TABLE IF NOT EXISTS customers (
    customer_id INT PRIMARY KEY,
    age INT,
    gender VARCHAR(10),
    city VARCHAR(50),
    signup_date DATE
);

CREATE TABLE IF NOT EXISTS orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE,
    category VARCHAR(50),
    quantity INT,
    unit_price DECIMAL(10,2),
    discount DECIMAL(4,2),
    total_amount DECIMAL(12,2),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Load the CSVs produced by generate_data.py
-- LOAD DATA LOCAL INFILE 'data/customers.csv' INTO TABLE customers
--   FIELDS TERMINATED BY ',' IGNORE 1 LINES;
-- LOAD DATA LOCAL INFILE 'data/orders.csv' INTO TABLE orders
--   FIELDS TERMINATED BY ',' IGNORE 1 LINES;

-- Example analysis queries
-- Monthly revenue
SELECT DATE_FORMAT(order_date, '%Y-%m') AS month, SUM(total_amount) AS revenue
FROM orders GROUP BY month ORDER BY month;

-- Top 10 customers by spend
SELECT c.customer_id, c.city, SUM(o.total_amount) AS spend
FROM orders o JOIN customers c USING (customer_id)
GROUP BY c.customer_id, c.city ORDER BY spend DESC LIMIT 10;

-- Revenue by category
SELECT category, SUM(total_amount) AS revenue, COUNT(*) AS orders
FROM orders GROUP BY category ORDER BY revenue DESC;
