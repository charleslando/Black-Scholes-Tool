\connect trades
SELECT
    client,
    trader,
    sum(commission) as com_sum,
    broker
FROM trades
WHERE commission > 0
  AND broker = 'Madison Brittner'

GROUP BY client, trader, broker


-- select trade_date
-- from trades
-- where trade_date between '04/01/2023' AND '05/01/2023'