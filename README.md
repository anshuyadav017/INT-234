## 📖 About the Project
This project implements a **Predictive Analytics system** to estimate the **selling price of used cars** using machine learning algorithms.  
It is developed as part of **INT 234 – Predictive Analytics** coursework and demonstrates data preprocessing, exploratory data analysis, and model building.

---

## 🎯 Project Objectives
- Analyze a real-world used car dataset
- Clean and preprocess data for machine learning
- Perform exploratory data analysis (EDA)
- Build regression and classification models
- Compare models using evaluation metrics
- Identify the best-performing machine learning model

---

## 📂 Dataset Details
- **Dataset Name:** Car Details v3
- **Source:** Kaggle (Car Dekho Dataset)
- **Total Records:** 8,128
- **Total Features:** 13
- **Target Variable:** Selling Price

### Attributes
**Numerical Features**
- Year  
- Selling Price  
- Kilometers Driven  
- Mileage  
- Engine  
- Max Power  
- Torque  
- Seats  

**Categorical Features**
- Car Name  
- Fuel Type  
- Seller Type  
- Transmission  
- Owner  

---

## 🧹 Data Preprocessing
- Removed duplicate records
- Handled missing values using median and mode
- Extracted numerical values from string columns
- Applied label encoding to categorical variables
- Feature scaling using StandardScaler
- Split dataset into training and testing sets

---

## 📊 Exploratory Data Analysis (EDA)
- Distribution analysis using histograms and box plots
- Correlation analysis using heatmaps
- PCA used for dimensionality reduction
- Observed strong correlation between selling price and:
  - Year
  - Engine capacity
  - Max power

---

## 🤖 Machine Learning Models Used

### Regression Models
- Simple Linear Regression
- Multiple Linear Regression
- Polynomial Regression

### Classification Models
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest
- Multi-Layer Perceptron (MLP)

### Unsupervised Learning
- K-Means Clustering
- Hierarchical Clustering
- Principal Component Analysis (PCA)

---

## 🏆 Results
- **Best Regression Model:** Multiple Linear Regression
- **Best Classification Model:** Random Forest (91.9% accuracy)

---

## 🛠 Technologies Used
- **Programming Language:** Python
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
- **Environment:** Jupyter Notebook

---

## 📌 Conclusion
The project successfully demonstrates the use of predictive analytics and machine learning in solving real-world automotive pricing problems. The models provide accurate predictions and insights into key factors influencing car prices.

---

## 🔮 Future Scope
- Implement deep learning models
- Use real-time automobile market data
- Deploy the model as a web application
- Apply explainable AI techniques
