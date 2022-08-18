import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import confusion_matrix, roc_auc_score
import statsmodels.api as sm

# Importing data

dataset = pd.read_csv("Credit Risk.csv")

# Dependent and Independent veriable selection

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

# Encode the target veriable, i.e. y veriable

LE_y = LabelEncoder()

y = LE_y.fit_transform(y)

# Now it's y veriable is in terms of 1s and 0s.

####################### 1st part is clear and now comes to the main part, i.e. Data clearning ###############

# Pre-processing

# Check for missing values and check for outliers.

nans = x.isnull().sum()
nans

# Learning more about perticular veriable. Let's say gender

# Gender

x['Gender'].describe()

x.Gender.value_counts()

x.Gender.isnull().sum()

# Filling NaNs 

x["Gender"].fillna("Male", inplace = True)

x.Gender.isnull().sum()


# Married

x['Married'].describe()

x.Married.value_counts()

x.Married.isnull().sum()

# Filling Nans

x["Married"].fillna("Yes", inplace = True)

x.Married.isnull().sum()


# Dependents

x["Dependents"].describe()

x.Dependents.value_counts()

x.Dependents.isnull().sum()

x["Dependents"].fillna("0", inplace = True)

x.Dependents.isnull().sum()


# Education

x["Education"].describe()

x.Education.isnull().sum()  # No missing values.



# Self_Employed

x["Self_Employed"].describe()

x.Self_Employed.isnull().sum()

x["Self_Employed"].fillna("No", inplace = True)

x.Self_Employed.isnull().sum()


# ApplicantIncome

x["ApplicantIncome"].describe()

x.ApplicantIncome.isnull().sum() # No missing values

# Check for outliers

x.boxplot("ApplicantIncome")

# Outliers treatment

q75, q25 = np.percentile(x.ApplicantIncome, [75, 25])

# Inter Quartile Range (iqr)

iqr = q75 - q25

iqr

# Upper Threshold value 

a = q75 + (1.5 * iqr)

a

# Lower Threshold value

b = q25 - (1.5 * iqr)

b