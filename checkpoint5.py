"""
In this checkpoint, we are going to work  on the heart disease dataset , 
this time we will use logistic regression to predict if a patient will have TenyearsCHD 
1. Apply logistic regression. 
2. Use a confusion matrix to validate your model. 
3. Another validation matrix for classification is ROC / AUC. 
Do your research on them, explain them, and apply them in our case.
"""
# Import relevant library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# Load dataset
data = pd.read_csv("Workspace/GoMyCode/heart_disease.csv")

cat_data = data.select_dtypes('object')
num_data = data.select_dtypes('number')

# print(cat_data, num_data)

# Check and treat for null value
# print(data.isnull().sum())

for i in data.columns:
    if i in cat_data.columns:
        data[i] = data[i].fillna(data[i].mode())
    else:
        data[i] = data[i].fillna(data[i].mean())

# print(data.isnull().sum())

# Transformation of data

encoder = LabelEncoder()
for i in data.columns:
    if i in cat_data.columns:
        data[i] = encoder.fit_transform(data[i])
        
# print(data.head())

# Selecting features
x = data.drop('TenYearCHD', axis=1)
y = data['TenYearCHD']
y = y.astype(int)

# Splitting into training and test set
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

# Confusion matrix
con_matrix = confusion_matrix(y_test,y_pred)
report = classification_report(y_test,y_pred)
print(f'Classification report {report}')
print(f'Confusion matrix {con_matrix}')

"""
The ROC (Receiver Operating Characteristic) curve is a visual representation that demonstrates how well a binary classifier can distinguish between classes by adjusting its discrimination threshold. 
It showcases the relationship between Sensitivity (True Positive Rate) and Specificity (True Negative Rate) across different threshold values.
AUC (Area Under the ROC Curve) is a numerical measure that assesses the effectiveness of a binary classification model. 
It spans from 0 to 1, where a value of 1 signifies flawless performance, while a score of 0.5 suggests random guessing with no discriminatory ability.
"""

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test,y_pred)
roc_auc = auc(fpr,tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve(area=%0.2f)' % roc_auc)
plt.plot([0,1], [0,1], color = 'navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

""" Interpretation:
For class 0, the precision is 0.89, and for class 1, it is 0.56. 
So, among the predicted positive cases, 89% of them are actually positive for class 0, and 56% for class 1.
The class 0 recall is 0.99, indicating that 99% of actual positive cases for class 0 are correctly predicted. 
For class 1, it is only 0.05, suggesting a low recall rate for class 1.
F1 score for class 0, is 0.94, and for class 1, it is 0.09.
Overall accuracy of the model is 0.89, indicating the proportion of correctly predicted instances out of the total instances.
The confusion matrix indicates that there are 747 true negatives (TN), 4 false positives (FP), 92 false negatives (FN), and 5 true positives (TP).
An AUC of 0.52 suggests that the model's ability to discriminate between positive and negative cases is close to random guessing, indicating poor performance.
In summary, while the model has high precision and accuracy for class 0, it performs poorly in terms of recall, F1-score, and AUC for class 1, indicating that it struggles to correctly identify instances of class 1, possibly due to class imbalance or other issues in the dataset. 
Further analysis and possibly model refinement are needed to improve its performance.
"""



