# Feature Glossary (12 Features)

This list is the approved v1 feature set used in the model.

- `Financial_Year`: fiscal year value, cleaned to numeric form.
- `sale`: total company sales / revenue for the year.
- `ni`: net income (profit after expenses and taxes).
- `at`: total assets.
- `lt`: total liabilities (what the company owes).
- `che`: cash and equivalents / short-term investments (liquid resources).
- `rect`: accounts receivable (money owed by customers).
- `invt`: inventory value.
- `cogs`: cost of goods sold.
- `txt`: total taxes paid or owed.
- `xint`: interest expense.
- `prcc_f`: a dataset-provided market/financial feature column named `prcc_f`; it is used as supplied by this dataset.

Interpretation notes:

- Larger `sale`, `at`, and `che` often indicate bigger companies.
- Large `rect` relative to `sale` can indicate aggressive revenue recognition patterns.
- Elevated `xint`, `lt`, and negative `ni` can indicate stress and potential reporting pressure.
