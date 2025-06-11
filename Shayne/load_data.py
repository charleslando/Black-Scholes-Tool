import pandas as pd
from sqlalchemy import create_engine, text

# Connect to your local PostgreSQL database
engine = create_engine("postgresql+psycopg://postgres:Rubixcube5@localhost:5432/eclipse")

# Read your CSV file into a DataFrame
df = pd.read_csv("Eclipse Trades 2022-2024 - 2022-2024.csv")

# preview the data
print("Loaded data:")
print(df.head())
#
# #get the data to insert into PostgreSQL
#
# clients = df['Client'].unique().tolist()
# print("Unique clients:", clients[:10])  # Display first 10 unique clients for brevity
#
# traders = df['Trader'].unique().tolist()
# print("Unique traders:", traders[:10])
#
# commissions = df['Commission'].unique().tolist()
#
#
#
#
#
#
# # Create the table (if not exists)
# with engine.begin() as conn:
#     conn.execute(text("""
#     select Client from trades
#
#     """))

# # Insert the data into the table
df.to_sql("trades", con=engine, if_exists="replace", index=False)

print("ran query!")
