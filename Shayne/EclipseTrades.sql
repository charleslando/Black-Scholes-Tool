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
