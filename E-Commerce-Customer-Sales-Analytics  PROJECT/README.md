# E-Commerce Customer & Sales Analytics Studio

A ready-to-run, interactive Data Science project for a 3rd-year CSE demonstration.

## Features
- Interactive sales, profit, order, customer and average-order-value KPIs
- Date, region and category filters
- Sales trends, category and regional visualizations
- Customer-level feature engineering and K-Means segmentation
- Adjustable K and Silhouette Score
- Outlier detection using IQR or Isolation Forest
- Baseline monthly sales forecasting with holdout MAE/RMSE
- Data quality summary and CSV exports
- Upload your own Superstore/e-commerce CSV file

## Requirements
- Python 3.10–3.12 recommended
- Internet connection for the first-time package installation
- VS Code or terminal

## Run on macOS (including MacBook Air)

1. Extract `E-Commerce-Customer-Sales-Analytics.zip`.
2. Open the extracted folder in VS Code.
3. Open Terminal in VS Code (`Terminal` → `New Terminal`).
4. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. Install packages:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Start the app:

   ```bash
   streamlit run app.py
   ```

7. Streamlit prints a local URL, normally `http://localhost:8501`. Open it in your browser.

### Windows
Use `py -m venv .venv`, then `.venv\Scripts\activate`, install requirements, and run `streamlit run app.py`.

## Use your own dataset
- Use the sidebar's **Upload your CSV dataset** control.
- The app automatically detects common column names.
- Best results come from columns such as:
  `Order Date, Customer ID, Order ID, Sales, Profit, Quantity, Discount, Category, Region`
- A `Sales`/`Revenue`/`Amount` column is required.
- Date-based charts and forecasting require a valid date column.
- If Customer ID is missing, the app treats each row as a separate customer; upload a true customer identifier for meaningful segmentation.

## Included sample dataset
`data/sample_superstore.csv` is synthetic demonstration data generated for this project. It is not the official Kaggle dataset and should not be presented as real business data. Upload your course/project dataset for final analysis.

## Models and interpretation
- **K-Means**: segments customers using Annual Spending, Purchase Frequency, and Average Order Value. Inputs are standardized before clustering.
- **Silhouette Score**: gives a clustering separation measure; interpret alongside the visualizations and business context.
- **IQR**: flags values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR.
- **Isolation Forest**: unsupervised anomaly detector over sales, profit, quantity and discount.
- **Sales forecast**: simple linear trend baseline with a chronological holdout split. It is for educational comparison, not production-grade prediction.

A flagged transaction is not necessarily incorrect or fraudulent. Investigate it before removing it. High sales/profit values can be legitimate.

## Suggested project structure
```
E-Commerce-Customer-Sales-Analytics/
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── sample_superstore.csv
```

## Suggested viva explanation
"This project converts raw e-commerce transactions into customer-level features, segments customers using K-Means, detects potentially unusual transactions using statistical and machine-learning methods, and presents sales trends and a baseline forecast in an interactive Streamlit dashboard. We evaluate clustering using the Silhouette Score and the forecasting baseline using MAE and RMSE."

## Frontend and backend explanation
- `app.py` is the Streamlit frontend and application controller. It displays navigation tabs, upload controls, filters, charts, tables and download buttons.
- The data preparation and analysis logic in `app.py` forms the backend layer: it standardizes uploaded columns, cleans values, calculates KPIs, creates customer features, trains the clustering/anomaly/forecast models, and returns results to the interface.
- Streamlit runs both layers in one Python application, so a separate web server or database is not required for this beginner-to-intermediate academic project.

## Quick start
On macOS, open Terminal in this folder and run:

```bash
chmod +x run_mac.command
./run_mac.command
```

Or follow the manual setup instructions above. Python 3.10–3.12 is recommended; avoid using a copied virtual environment from another computer.
