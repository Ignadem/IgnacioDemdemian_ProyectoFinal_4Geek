-- Phase 1 SQL Analysis
-- Database: data/fraud_financial.db
-- Table: financial_records

-- 1. Total records
SELECT
  COUNT(*) AS total_records
FROM financial_records;

-- 2. Fraud vs non-fraud count
SELECT
  CASE WHEN AAER_ID IS NOT NULL THEN 1 ELSE 0 END AS target_fraud,
  COUNT(*) AS records
FROM financial_records
GROUP BY target_fraud
ORDER BY target_fraud;

-- 3. Fraud rate by financial year
SELECT
  Financial_Year,
  COUNT(*) AS total_records,
  SUM(CASE WHEN AAER_ID IS NOT NULL THEN 1 ELSE 0 END) AS fraud_cases,
  ROUND(AVG(CASE WHEN AAER_ID IS NOT NULL THEN 1.0 ELSE 0.0 END) * 100, 4) AS fraud_rate_percent
FROM financial_records
GROUP BY Financial_Year
ORDER BY Financial_Year;

-- 4. Average key financial values by fraud label
SELECT
  CASE WHEN AAER_ID IS NOT NULL THEN 1 ELSE 0 END AS target_fraud,
  COUNT(*) AS records,
  ROUND(AVG(sale), 2) AS avg_sales,
  ROUND(AVG(ni), 2) AS avg_net_income,
  ROUND(AVG(at), 2) AS avg_assets,
  ROUND(AVG(lt), 2) AS avg_liabilities,
  ROUND(AVG(che), 2) AS avg_cash
FROM financial_records
GROUP BY target_fraud
ORDER BY target_fraud;

-- 5. Receivables-to-sales ratio by fraud label
SELECT
  CASE WHEN AAER_ID IS NOT NULL THEN 1 ELSE 0 END AS target_fraud,
  COUNT(*) AS records,
  ROUND(AVG(rect), 2) AS avg_receivables,
  ROUND(AVG(sale), 2) AS avg_sales,
  ROUND(AVG(rect / NULLIF(sale, 0)), 4) AS avg_receivables_to_sales_ratio
FROM financial_records
GROUP BY target_fraud
ORDER BY target_fraud;

-- 6. Liabilities-to-assets ratio by fraud label
SELECT
  CASE WHEN AAER_ID IS NOT NULL THEN 1 ELSE 0 END AS target_fraud,
  COUNT(*) AS records,
  ROUND(AVG(lt), 2) AS avg_liabilities,
  ROUND(AVG(at), 2) AS avg_assets,
  ROUND(AVG(lt / NULLIF(at, 0)), 4) AS avg_liabilities_to_assets_ratio
FROM financial_records
GROUP BY target_fraud
ORDER BY target_fraud;
