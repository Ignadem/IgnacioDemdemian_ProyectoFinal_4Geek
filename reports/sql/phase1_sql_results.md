# Phase 1 SQL Results

Generated from `data/fraud_financial.db` using `scripts/run_sql_analysis.py`.

## Total Records

| total_records |
| --- |
| 87974 |

## Fraud vs Non-Fraud Count

| target_fraud | records |
| --- | --- |
| 0 | 87399 |
| 1 | 575 |

## Fraud Rate by Financial Year

| Financial_Year | total_records | fraud_cases | fraud_rate_percent |
| --- | --- | --- | --- |
| FY1995 | 2568 | 11 | 0.4283 |
| FY1996 | 2927 | 12 | 0.41 |
| FY1997 | 3329 | 20 | 0.6008 |
| FY1998 | 3618 | 29 | 0.8015 |
| FY1999 | 4134 | 46 | 1.1127 |
| FY2000 | 4048 | 65 | 1.6057 |
| FY2001 | 3892 | 69 | 1.7729 |
| FY2002 | 3878 | 60 | 1.5472 |
| FY2003 | 3977 | 51 | 1.2824 |
| FY2004 | 4078 | 36 | 0.8828 |
| FY2005 | 4072 | 28 | 0.6876 |
| FY2006 | 4081 | 16 | 0.3921 |
| FY2007 | 4039 | 15 | 0.3714 |
| FY2008 | 3727 | 14 | 0.3756 |
| FY2009 | 3671 | 17 | 0.4631 |
| FY2010 | 3649 | 20 | 0.5481 |
| FY2011 | 3603 | 14 | 0.3886 |
| FY2012 | 3578 | 10 | 0.2795 |
| FY2013 | 3600 | 9 | 0.25 |
| FY2014 | 3687 | 11 | 0.2983 |
| FY2015 | 3557 | 8 | 0.2249 |
| FY2016 | 3502 | 8 | 0.2284 |
| FY2017 | 3402 | 5 | 0.147 |
| FY2018 | 3357 | 1 | 0.0298 |

## Average Key Financial Values by Fraud Label

| target_fraud | records | avg_sales | avg_net_income | avg_assets | avg_liabilities | avg_cash |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 87399.0 | 7739.0 | 428.34 | 20335.17 | 15512.79 | 4433.98 |
| 1.0 | 575.0 | 4679.09 | 192.53 | 18328.12 | 16010.56 | 2499.1 |

## Receivables-to-Sales Ratio by Fraud Label

| target_fraud | records | avg_receivables | avg_sales | avg_receivables_to_sales_ratio |
| --- | --- | --- | --- | --- |
| 0.0 | 87399.0 | 1413.32 | 7739.0 | 0.2931 |
| 1.0 | 575.0 | 3862.2 | 4679.09 | 0.9426 |

## Liabilities-to-Assets Ratio by Fraud Label

| target_fraud | records | avg_liabilities | avg_assets | avg_liabilities_to_assets_ratio |
| --- | --- | --- | --- | --- |
| 0.0 | 87399.0 | 15512.79 | 20335.17 | 0.5968 |
| 1.0 | 575.0 | 16010.56 | 18328.12 | 0.5298 |
