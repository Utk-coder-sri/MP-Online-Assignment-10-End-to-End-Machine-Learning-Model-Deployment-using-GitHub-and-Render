# Assignment-10: Heart Disease Prediction using Machine Learning and Flask Deployment

## Student Details

- **Name:** Utkrisht Srivastava
- **Registration Number:** 23MIP10139
- **Application Number:** IN26012359

---

# Objective

The objective of this assignment is to develop a Machine Learning model for predicting the presence of heart disease using the Heart Disease dataset. The trained model is deployed as a REST API using Flask and prepared for cloud deployment on Render. This assignment demonstrates the complete Machine Learning workflow from data preprocessing and model training to deployment using MLOps practices.

---

# Dataset Link

**Kaggle Dataset:**

https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

---

# Libraries Used

- pandas
- numpy
- scikit-learn
- joblib
- Flask
- gunicorn

---

# Methodology

1. Loaded the Heart Disease dataset using Pandas.
2. Displayed dataset information and checked for missing values.
3. Identified input features and target variable.
4. Split the dataset into training and testing sets using an 80:20 ratio.
5. Trained a Random Forest Classifier.
6. Evaluated the model using Accuracy Score.
7. Saved the trained model using Joblib.
8. Developed a REST API using Flask.
9. Prepared the project for deployment on Render.

---

# Model Used

**Random Forest Classifier**

Random Forest combines multiple decision trees to improve prediction accuracy and reduce overfitting. It performs well on structured medical datasets and provides reliable classification results.

---

# Results

The Random Forest model achieved high prediction accuracy on the testing dataset and successfully classified patients based on their medical attributes. The trained model was saved as `model.pkl` and integrated into a Flask REST API capable of receiving patient details in JSON format and returning heart disease predictions.

---

# Render Deployment URL

Add your deployed Render URL here after deployment.

Example:

https://your-app-name.onrender.com

---

# Conclusion

The Heart Disease Prediction system successfully demonstrated the complete machine learning deployment pipeline, beginning with data preprocessing and model training and ending with REST API development using Flask. The Random Forest model achieved high prediction accuracy and produced reliable results for heart disease classification. Deploying the application introduced practical challenges such as dependency management, configuring deployment settings, and ensuring compatibility between the local and cloud environments. These deployment steps highlighted the importance of MLOps, which helps automate model deployment, version control, monitoring, and maintenance. Applying MLOps practices makes machine learning solutions more scalable, reproducible, and easier to manage in real-world healthcare and production environments.

---