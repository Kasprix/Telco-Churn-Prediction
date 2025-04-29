# Telco Customer Churn Prediction

## Project Overview
This project aims to predict customer churn for a telecom company using machine learning models. The dataset contains information about customer demographics, services, and account details. The goal is to identify customers likely to churn and provide actionable insights to improve customer retention strategies.

---

## Dataset
The dataset used in this project is the **Telco Customer Churn Dataset**, which includes 7,043 observations and 21 variables. Key features include:

- **Demographics**: Gender, Senior Citizen status, Partner, Dependents.
- **Services**: Phone Service, Internet Service, Online Security, Streaming TV, etc.
- **Account Information**: Tenure, Contract Type, Payment Method, Monthly Charges, Total Charges.
- **Target Variable**: `Churn` (Yes/No).

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
- **Initial Checks**: Data shape, column types, missing values, and duplicates.
- **Data Cleaning**: Dropped unnecessary columns (e.g., `customerID`), handled missing values in `TotalCharges`, and converted data types.
- **Visualization**: 
  - Churn distribution (imbalanced dataset with ~3:1 ratio).
  - Correlation heatmaps, histograms, and box plots for numeric features.
  - Count plots and feature-target relationships for categorical features.

### 2. Feature Engineering
- Created new features such as `AverageMonthlySpend` to capture customer spending behavior.
- Preselected features using variance threshold, high-cardinality filtering, and mutual information.

### 3. Model Training and Evaluation
- **Models Used**:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - XGBoost
  - LightGBM
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC.
  - Focused on recall and F1-Score due to the imbalanced dataset.

### 4. Results
| Model                 | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|-----------------------|----------|-----------|--------|----------|---------|--------|
| LightGBM             | 0.76     | 0.54      | 0.78   | 0.64     | 0.84    | 0.64   |
| XGBoost              | 0.76     | 0.53      | 0.77   | 0.63     | 0.84    | 0.64   |
| Logistic Regression  | 0.73     | 0.50      | 0.80   | 0.62     | 0.83    | 0.61   |
| Random Forest        | 0.76     | 0.53      | 0.75   | 0.62     | 0.83    | 0.63   |
| SVM                  | 0.69     | 0.45      | 0.82   | 0.58     | 0.83    | 0.61   |

- **Best Model**: LightGBM achieved the highest F1-Score (0.64) and Recall (0.78), making it the recommended model for deployment.

---

## Key Insights
1. **Churn Drivers**:
   - Customers with month-to-month contracts, higher monthly charges, and no internet security are more likely to churn.
   - Tenure and total charges are positively correlated, indicating loyal customers tend to spend more.

2. **Imbalanced Dataset**:
   - The dataset is imbalanced (~3:1 ratio for Not-Churn:Churn). Techniques like SMOTE and class weighting were used to address this.

3. **Model Performance**:
   - LightGBM and XGBoost performed the best, with a good balance between recall and precision.

---

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Telco-Churn-Prediction.git
   cd Telco-Churn-Prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/EDA.ipynb
   ```

4. Train models and evaluate results using the provided pipeline in the notebook.

---

## Future Work
- **Precision Optimization**: Fine-tune decision thresholds to reduce false positives.
- **Explainability**: Use SHAP values to explain model predictions.
- **Deployment**: Deploy the LightGBM model as a REST API for real-time predictions.

---

## Acknowledgments
- Dataset: [Kaggle - Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- Libraries: Pandas, Scikit-learn, XGBoost, LightGBM, Seaborn, Matplotlib, Imbalanced-learn.

---

## License
This project is licensed under the MIT License.