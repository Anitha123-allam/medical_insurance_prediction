

# 🏥 Medical Insurance Cost Prediction Using Machine Learning
<img width="1534" height="838" alt="image" src="https://github.com/user-attachments/assets/17de7227-0f23-453b-896c-1d97583d832c" />

## 📌 Project Overview

Medical insurance plays an important role in managing healthcare expenses. Insurance companies need to estimate the medical costs of individuals based on different personal and lifestyle factors.

This project focuses on predicting individual medical insurance charges using Machine Learning algorithms. The model analyzes factors such as age, BMI, gender, smoking habits, number of children, and region to estimate insurance costs.

The main objective is to build and compare multiple Machine Learning models and identify the best-performing model for predicting medical insurance expenses.

## 🎯 Objectives

- Analyze medical insurance data to understand important factors affecting insurance charges.
- Perform Exploratory Data Analysis (EDA) to identify patterns and relationships.
- Clean and preprocess the dataset.
- Apply different Machine Learning algorithms.
- Compare model performance using evaluation metrics.
- Select and save the best-performing model.
- Develop an interactive web application using Streamlit to predict insurance charges.

## 📂 Dataset Information

**Dataset Name:** Medical Insurance Dataset

**Target Variable:** `charges`

The dataset contains the following features:

| Feature | Description |
|---|---|
| age | Age of the individual |
| sex | Gender of the individual |
| bmi | Body Mass Index |
| children | Number of children |
| smoker | Smoking status (Yes/No) |
| region | Residential region |
| charges | Medical insurance cost (Target Variable) |

## 🛠️ Technologies & Tools Used

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Model Deployment:** Streamlit
- **Development Environment:** Jupyter Notebook / VS Code
- **Version Control:** Git & GitHub

## 🔍 Project Workflow

1. **Data Collection:** Load the medical insurance dataset.
2. **Data Understanding:** Explore dataset shape, columns, data types, and statistical information.
3. **Data Cleaning:** Check and handle missing values, duplicates, and inconsistent data.
4. **Exploratory Data Analysis:** Analyze relationships between insurance charges and individual features using visualizations.
5. **Data Preprocessing:** Perform categorical encoding and prepare input features.
6. **Model Building:** Train multiple Machine Learning algorithms.
7. **Model Evaluation:** Compare model performance using suitable evaluation metrics.
8. **Best Model Selection:** Select the model based on evaluation results.
9. **Model Saving:** Save the trained model using Pickle.
10. **Deployment:** Build an interactive Streamlit application for insurance cost prediction.

## 🤖 Machine Learning Algorithms

The following algorithms are used for model building and comparison:

- Linear Regression
- Decision Tree Regressor
- K-Nearest Neighbors (KNN) Regressor
- Random forest

## 📊 Model Evaluation Metrics

The models are evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

These metrics help identify how accurately each model predicts medical insurance costs.

## 🌐 Streamlit Web Application

An interactive Streamlit application allows users to enter personal information such as:

- Age
- Gender
- BMI
- Number of Children
- Smoking Status
- Residential Region

Based on the entered details, the application predicts the estimated medical insurance charges.

## 📁 Project Structure

```text
Medical-Insurance-Cost-Prediction/
│
├── app.py
├── insurance.csv
├── best_insurance_model.pkl
├── Medical_Insurance_Prediction.ipynb
├── requirements.txt
└── README.md
```

## ⚙️ Installation & Execution

**Step 1: Clone the repository**

```bash
https://github.com/Anitha123-allam/medical_insurance_prediction
```

**Step 2: Navigate to the project folder**

```bash
cd Medical-Insurance-Cost-Prediction
```

**Step 3: Install required libraries**

```bash
pip install -r requirements.txt
```

**Step 4: Run the Streamlit application**

```bash
streamlit run app.py
```

## 💡 Key Learning Outcomes

- Understanding real-world regression problems.
- Performing data cleaning and Exploratory Data Analysis.
- Applying feature encoding and preprocessing techniques.
- Building and comparing multiple regression algorithms.
- Evaluating Machine Learning models.
- Saving and loading trained models.
- Deploying a Machine Learning model using Streamlit.

## 🚀 Future Improvements

- Explore advanced regression algorithms such as Random Forest and Gradient Boosting.
- Improve prediction accuracy through hyperparameter tuning.
- Add more healthcare-related features.
- Deploy the application on Streamlit Community Cloud.

## 👩‍💻 Author

**Anitha Allam**

Aspiring Data Analyst | Machine Learning Enthusiast

GitHub: https://github.com/Anitha123-allam


Deploy link:https://medicalinsuranceprediction-yvipzcvydxoagia9magu79.streamlit.app/

---

⭐ If you find this project useful, feel free to star the repository!
