CROP RECOMMENDATION SYSTEM USING MACHINE LEARNING
=================================================

PROJECT OVERVIEW
----------------
This project implements an end-to-end Crop Recommendation System using
Machine Learning techniques. The system recommends the most suitable crop
to cultivate based on soil nutrients and environmental conditions.

It helps farmers and agricultural planners make data-driven decisions to
maximize crop yield and optimize resource usage.


PROBLEM STATEMENT
-----------------
Traditional crop selection methods rely on experience and manual judgment,
which may not always be accurate. This project uses machine learning models
to predict the best crop based on soil and climatic parameters.


INPUT FEATURES
--------------
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall


KEY FEATURES
------------
- Multi-class crop recommendation
- Trained and evaluated using multiple ML algorithms
- High accuracy on benchmark dataset
- End-to-end pipeline: data preprocessing to prediction
- Deployed using Streamlit for real-time usage


MODELS USED
-----------
- Logistic Regression
- Decision Tree
- Random Forest
- Naïve Bayes

Best performing model achieved accuracy of 98%+.


DATASET
-------
- Public agriculture dataset
- Contains soil nutrient and climatic data
- Multiple crop labels
- Dataset split into:
  - Training set
  - Test set


DATA PREPROCESSING
------------------
- Handling missing values
- Feature scaling and normalization
- Train-test split
- Label encoding for crop names


RESULTS
-------
Model Performance Summary:
- Logistic Regression : ~95%
- Decision Tree       : ~96%
- Naïve Bayes         : ~97%
- Random Forest       : 98%+ (Best Model)

The Random Forest model was selected for deployment due to its superior
accuracy and stability.


DEPLOYMENT
----------
- Built an interactive web application using Streamlit
- Users can input soil and climate values
- System instantly predicts the most suitable crop
- Deployed on Streamlit Cloud


TECHNOLOGIES USED
-----------------
Programming Language : Python
Libraries           : NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
Framework           : Streamlit
Model Type          : Supervised Machine Learning (Classification)


HOW TO RUN THE PROJECT
---------------------

Step 1: Clone the Repository
git clone 
cd crop-recommendation-system

Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Run the Application
streamlit run app.py


USE CASES
---------
- Smart agriculture
- Decision support for farmers
- Precision farming
- Crop yield optimization


FUTURE IMPROVEMENTS
-------------------
- Include fertilizer recommendation
- Integrate weather API for real-time data
- Add region-specific crop suggestions
- Mobile application deployment


REFERENCES
----------
- Scikit-learn Documentation
- Streamlit Documentation
- Agricultural Dataset Sources


AUTHOR
------
Name     : Arin Mudgal
Degree   : B.Tech – Computer Science and Engineering
Interest : Machine Learning, Data Science, Smart Agriculture


NOTE
----
If this project helps you, consider starring the repository.
