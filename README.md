# Multi_Classification

# Logistic Regression Model Comparison

This notebook explores two approaches for Logistic Regression for multi-class classification:

1. **One-vs-Rest (OvR)**
2. **Multinomial**

## Data Preparation

- We use `make_classification` from `sklearn.datasets` to generate a synthetic dataset with 1000 samples, 10 features, and 2 classes.
- The dataset is split into training and testing sets using `train_test_split` with a test size of 30%.

## Model Training and Evaluation

### One-vs-Rest (OvR)
Use code with caution
python from sklearn.linear_model import LogisticRegression from sklearn.model_selection import train_test_split from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

Create and train the model
model = LogisticRegression(multi_class='ovr', solver='lbfgs') X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1) model.fit(X_train, y_train)

Make predictions
y_pred = model.predict(X_test)

Evaluate the model
print(confusion_matrix(y_test, y_pred)) print(accuracy_score(y_test, y_pred)) print(classification_report(y_test, y_pred))

 
### Multinomial
Use code with caution
python from sklearn.linear_model import LogisticRegression from sklearn.model_selection import train_test_split from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

Create and train the model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs') X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1) model.fit(X_train, y_train)

Make predictions
y_pred = model.predict(X_test)

Evaluate the model
print(confusion_matrix(y_test, y_pred)) print(accuracy_score(y_test, y_pred)) print(classification_report(y_test, y_pred))

 
## Conclusion

- Both OvR and Multinomial approaches are applied to the multi-class classification problem.
- Performance metrics are used to compare the two strategies and determine which one is more suitable for this particular dataset.
- The code provides a clear demonstration of how to implement and evaluate these approaches using scikit-learn.
Use code with caution
Note: The markdown file summarizes the code and its purpose. To see the actual output and detailed results, run the code in the notebook.

Sources
scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_multinomial.html
www.aaronlichtner.com/2018/12/01/building-a-flask-app-model-serving/
www.cnblogs.com/cch-EX/p/13507673.html
silencebreaker/Python
