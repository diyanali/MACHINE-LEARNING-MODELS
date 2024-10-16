# MACHINE-LEARNING-MODELS

---

# Machine Learning Models Repository

Welcome to the **Machine Learning Models Repository**, a collection of basic implementations of commonly used machine learning algorithms. This repository is designed for anyone looking to understand how to build and train machine learning models from scratch using Python and popular libraries such as `scikit-learn`, `pandas`, `matplotlib`, and `seaborn`.

## Models Included:
- **Linear Regression**
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Decision Tree**



### 1. Importing Necessary Libraries
Before starting with any machine learning project, we import the necessary libraries such as:
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computations.
- `matplotlib` & `seaborn`: For data visualization.
- `scikit-learn`: For machine learning models and evaluation metrics.

### 2. Importing the Dataset
Data is the backbone of any machine learning model. The dataset may be imported from:
- CSV files (`pandas.read_csv()`).
- URLs.
- Public datasets like those available in `sklearn.datasets`.

### 3. Exploratory Data Analysis (EDA)
Before building models, understanding the data is crucial. Key EDA tasks include:
- Checking for missing values.
- Summary statistics (`describe()`).
- Data distribution visualization using histograms, box plots, etc.
- Feature correlation using heatmaps.

### 4. Data Preprocessing
To prepare the data for modeling, the following preprocessing steps are covered:
- Handling missing values.
- Encoding categorical variables (e.g., `pd.get_dummies()` or `LabelEncoder`).
- Splitting the data into training and testing sets (`train_test_split()`).

### 5. Model Building
Each script demonstrates the process of building a machine learning model using `scikit-learn`:
- **Linear Regression**: For predicting continuous target variables.
- **Logistic Regression**: For binary classification tasks.
- **Support Vector Machine (SVM)**: For classification by finding a decision boundary.
- **Decision Tree**: For both classification and regression using tree-based models.

### 6. Model Training
The models are trained using the training set with the `fit()` method.

### 7. Model Evaluation
Once trained, the models are evaluated using metrics like:
- **Mean Squared Error (MSE)** for regression models.
- **Accuracy, Precision, Recall, F1 Score, and ROC-AUC** for classification models.

### 8. Visualization of Results
We visualize the performance of the models using:
- Actual vs predicted value plots (for regression).
- Confusion matrices (for classification).
- ROC Curves (for binary classification).
