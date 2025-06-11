CREATE TABLE sales (
    id SERIAL PRIMARY KEY,
    client TEXT,
    trader TEXT,
    commission FLOAT
);

CREATE TABLE allocations (
    id SERIAL PRIMARY KEY,
    sales_id INTEGER,
    allocation_date DATE,
    amount FLOAT,
    FOREIGN KEY (sales_id) REFERENCES sales(id)
);

