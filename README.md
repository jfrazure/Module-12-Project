# Module-12-Project

Skip to left side bar





/Module-12-Project/
Jonathan Frazure
Last Modified














Python 3 (ipykernel)
Credit Risk Classification¶

Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. In this Challenge, you’ll use various techniques to train and evaluate models with imbalanced classes. You’ll use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

Instructions:

This challenge consists of the following subsections:

Split the Data into Training and Testing Sets

Create a Logistic Regression Model with the Original Data

Predict a Logistic Regression Model with Resampled Training Data

Split the Data into Training and Testing Sets

Open the starter code notebook and then use it to complete the following steps.

Read the lending_data.csv data from the Resources folder into a Pandas DataFrame.

Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns.

Note A value of 0 in the “loan_status” column means that the loan is healthy. A value of 1 means that the loan has a high risk of defaulting.

Check the balance of the labels variable (y) by using the value_counts function.

Split the data into training and testing datasets by using train_test_split.

Create a Logistic Regression Model with the Original Data

Employ your knowledge of logistic regression to complete the following steps:

Fit a logistic regression model by using the training data (X_train and y_train).

Save the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model.

Evaluate the model’s performance by doing the following:

Calculate the accuracy score of the model.

Generate a confusion matrix.

Print the classification report.

Answer the following question: How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

Predict a Logistic Regression Model with Resampled Training Data

Did you notice the small number of high-risk loan labels? Perhaps, a model that uses resampled data will perform better. You’ll thus resample the training data and then reevaluate the model. Specifically, you’ll use RandomOverSampler.

To do so, complete the following steps:

Use the RandomOverSampler module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points.

Use the LogisticRegression classifier and the resampled data to fit the model and make predictions.

Evaluate the model’s performance by doing the following:

Calculate the accuracy score of the model.

Generate a confusion matrix.

Print the classification report.

Answer the following question: How well does the logistic regression model, fit with oversampled data, predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

Write a Credit Risk Analysis Report

For this section, you’ll write a brief report that includes a summary and an analysis of the performance of both machine learning models that you used in this challenge. You should write this report as the README.md file included in your GitHub repository.

Structure your report by using the report template that Starter_Code.zip includes, and make sure that it contains the following:

An overview of the analysis: Explain the purpose of this analysis.

The results: Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of both machine learning models.

A summary: Summarize the results from the machine learning models. Compare the two versions of the dataset predictions. Include your recommendation for the model to use, if any, on the original vs. the resampled data. If you don’t recommend either model, justify your reasoning.


# Import the modules
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced
​
import warnings
warnings.filterwarnings('ignore')
Split the Data into Training and Testing Sets
Step 1: Read the lending_data.csv data from the Resources folder into a Pandas DataFrame.

# Read the CSV file from the Resources folder into a Pandas DataFrame
lending_data_df = pd.read_csv(
    Path('Resources/lending_data.csv')
)
​
# Review the DataFrame
lending_data_df
loan_size	interest_rate	borrower_income	debt_to_income	num_of_accounts	derogatory_marks	total_debt	loan_status
0	10700.0	7.672	52800	0.431818	5	1	22800	0
1	8400.0	6.692	43600	0.311927	3	0	13600	0
2	9000.0	6.963	46100	0.349241	3	0	16100	0
3	10700.0	7.664	52700	0.430740	5	1	22700	0
4	10800.0	7.698	53000	0.433962	5	1	23000	0
...	...	...	...	...	...	...	...	...
77531	19100.0	11.261	86600	0.653580	12	2	56600	1
77532	17700.0	10.662	80900	0.629172	11	2	50900	1
77533	17600.0	10.595	80300	0.626401	11	2	50300	1
77534	16300.0	10.068	75300	0.601594	10	2	45300	1
77535	15600.0	9.742	72300	0.585062	9	2	42300	1
77536 rows × 8 columns

Step 2: Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns.

# Separate the data into labels and features
​
# Separate the y variable, the labels
y = lending_data_df['loan_status']
​
# Separate the X variable, the features
X = lending_data_df.drop(columns = 'loan_status')

# Review the y variable Series
display(y)
0        0
1        0
2        0
3        0
4        0
        ..
77531    1
77532    1
77533    1
77534    1
77535    1
Name: loan_status, Length: 77536, dtype: int64

# Review the X variable DataFrame
display(X)
loan_size	interest_rate	borrower_income	debt_to_income	num_of_accounts	derogatory_marks	total_debt
0	10700.0	7.672	52800	0.431818	5	1	22800
1	8400.0	6.692	43600	0.311927	3	0	13600
2	9000.0	6.963	46100	0.349241	3	0	16100
3	10700.0	7.664	52700	0.430740	5	1	22700
4	10800.0	7.698	53000	0.433962	5	1	23000
...	...	...	...	...	...	...	...
77531	19100.0	11.261	86600	0.653580	12	2	56600
77532	17700.0	10.662	80900	0.629172	11	2	50900
77533	17600.0	10.595	80300	0.626401	11	2	50300
77534	16300.0	10.068	75300	0.601594	10	2	45300
77535	15600.0	9.742	72300	0.585062	9	2	42300
77536 rows × 7 columns

Step 3: Check the balance of the labels variable (y) by using the value_counts function.

# Check the balance of our target values
y.value_counts()
loan_status
0    75036
1     2500
Name: count, dtype: int64
Step 4: Split the data into training and testing datasets by using train_test_split.

# Import the train_test_learn module
from sklearn.model_selection import train_test_split
​
# Split the data using train_test_split
# Assign a random_state of 1 to the function
from sklearn.model_selection import train_test_split
​
# Split the data using train_test_split
# Assign a random_state of 1 to the function
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    random_state = 1
)
Create a Logistic Regression Model with the Original Data
Step 1: Fit a logistic regression model by using the training data (X_train and y_train).

# Import the LogisticRegression module from SKLearn
from sklearn.linear_model import LogisticRegression
​
# Instantiate the Logistic Regression model
# Assign a random_state parameter of 1 to the model
LR_model = LogisticRegression(random_state = 1)
​
# Fit the model using training data
LR_model.fit(X_train, y_train)

LogisticRegression
LogisticRegression(random_state=1)
Step 2: Save the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model.

# Make a prediction using the testing data
LR_predictions = LR_model.predict(X_test)
Step 3: Evaluate the model’s performance by doing the following:

Calculate the accuracy score of the model.

Generate a confusion matrix.

Print the classification report.


# Print the balanced_accuracy score of the model
balanced_accuracy_score(y_test, LR_predictions)
0.9520479254722232

# Generate a confusion matrix for the model
cm_imbalanced = confusion_matrix(y_test, LR_predictions)
cm_imbalanced_df = pd.DataFrame(cm_imbalanced, 
                                index = ['Actual Healthy Loans (low-risk)', 
                                'Actual Non-Healthy Loans (high-risk)'], 
                                columns = ['Predicted Healthy Loans (low-risk)', 'Predicted Non-Healthy Loans (high-risk)']
                              )
cm_imbalanced_df
Predicted Healthy Loans (low-risk)	Predicted Non-Healthy Loans (high-risk)
Actual Healthy Loans (low-risk)	18663	102
Actual Non-Healthy Loans (high-risk)	56	563

# Print the classification report for the model
print(classification_report_imbalanced(y_test, LR_predictions))
                   pre       rec       spe        f1       geo       iba       sup

          0       1.00      0.99      0.91      1.00      0.95      0.91     18765
          1       0.85      0.91      0.99      0.88      0.95      0.90       619

avg / total       0.99      0.99      0.91      0.99      0.95      0.91     19384

Step 4: Answer the following question.
Question: How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

**Answer:

The model performs well according to the balanced accuracy score (95%) due to the data being imbalanced. The number of healthy loans (low-risk) highly outweighs the number of non-healthy (high-risk) loans indicating that the model would predict loan status's as healthy better than being able to predict loan status's as non-healthy.

According to the imbalanced classification report, the model predicted healthy loans [0] 100% of the time and predicted non-healthy loans [1] 85% of the time.)

Predict a Logistic Regression Model with Resampled Training Data
Step 1: Use the RandomOverSampler module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points.

# Import the RandomOverSampler module form imbalanced-learn
from imblearn.over_sampling import RandomOverSampler
​
# Instantiate the random oversampler model
# # Assign a random_state parameter of 1 to the model
ROS_model = RandomOverSampler(random_state = 1)
​
# Fit the original training data to the random_oversampler model
X_oversampled, y_oversampled = ROS_model.fit_resample(X_train, y_train)

# Count the distinct values of the resampled labels data
y_oversampled.value_counts()
loan_status
0    56271
1    56271
Name: count, dtype: int64
Step 2: Use the LogisticRegression classifier and the resampled data to fit the model and make predictions.

# Instantiate the Logistic Regression model
# Assign a random_state parameter of 1 to the model
LR_oversampled_model = LogisticRegression(random_state = 1)
​
# Fit the model using the resampled training data
LR_oversampled_model.fit(X_oversampled, y_oversampled)
​
# Make a prediction using the testing data
LR_oversampled_pred = LR_oversampled_model.predict(X_test)
Step 3: Evaluate the model’s performance by doing the following:

Calculate the accuracy score of the model.

Generate a confusion matrix.

Print the classification report.


# Print the balanced_accuracy score of the model 
balanced_accuracy_score(y_test, LR_oversampled_pred)
0.9936781215845847

# Generate a confusion matrix for the model
cm_oversampled = confusion_matrix(y_test, LR_oversampled_pred)
cm_oversampled_df = pd.DataFrame(cm_oversampled, 
                                index = ['Actual Healthy Loans (low-risk)', 
                                'Actual Non-Healthy Loans (high-risk)'], 
                                columns = ['Predicted Healthy Loans (low-risk)', 'Predicted Non-Healthy Loans (high-risk)']
                              )
cm_oversampled_df
Predicted Healthy Loans (low-risk)	Predicted Non-Healthy Loans (high-risk)
Actual Healthy Loans (low-risk)	18649	116
Actual Non-Healthy Loans (high-risk)	4	615

# Print the classification report for the model
print(classification_report_imbalanced(y_test, LR_oversampled_pred))
                   pre       rec       spe        f1       geo       iba       sup

          0       1.00      0.99      0.99      1.00      0.99      0.99     18765
          1       0.84      0.99      0.99      0.91      0.99      0.99       619

avg / total       0.99      0.99      0.99      0.99      0.99      0.99     19384

Step 4: Answer the following question
Question: How well does the logistic regression model, fit with oversampled data, predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

**Answer:

The oversampled model generated an accuracy score of 99% which is higher than the model fitted with imbalanced data. The oversampled model performs better because it does a exceptional job in catching mistakes such as labeling non-healthy (high-risk) loans as healthy (low-risk). This is analyzed based off of the recall score increasing from the imbalanced model to the oversampled model 0.91 --> 0.99

A lending company might want a model that requires a higher recall because healthy loans being identified as a non-healthy loan might be more costly for a lending company since it might cause the loss of customers but it would not be as big of a deal since they have not provided any funds to the customer thus no losses for bad loans. Non-healthy loans being identified as a healthy loan might surely be more costly for a lending company due to bad loan losses for the lender.