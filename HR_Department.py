import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.width',None)

data=pd.read_csv("Human_Resources.csv")
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

# Let's replace the 'Attrition' and 'overtime' and 'Over18' column with integers before performing any visualizations

data["Attrition"]=data["Attrition"].apply(lambda x:1 if x=="Yes" else 0)
data["OverTime"]=data["OverTime"].apply(lambda x:1 if x=="Yes" else 0)
data["Over18"]=data["Over18"].apply(lambda x:1 if x=="Y" else 0)

print(data.head())
print(data.info())

hist_bar=data.hist(bins=30, figsize=(15,10),color="Red")
plt.tight_layout()
# plt.show()
plt.close()

"""Findings"""
# Several features such as 'MonthlyIncome' and 'TotalWorkingYears' are tailed heavy
# It makes sense to drop 'EmployeeCount'(all are 1's) and 'StandardHours'(fixed @ 80) and 'Over18 (all are 1's) 'EmployeeNumber'(seems like Index)
# since they do not change from one employee to the other dropping it could be ideal.

data.drop(["EmployeeCount","StandardHours","Over18","EmployeeNumber"],axis=1,inplace=True)
print(data.head())

# Let's see how many employees left the company!

left_df=data[data["Attrition"]==1]
stayed_df=data[data["Attrition"]==0]
print(left_df)
print(stayed_df)

# Count of employees who are left and Stayed

count_left=data.value_counts(data["Attrition"]==1)
print(count_left)

print("Total no. of employees =",len(data))
print("No. of employees left =",len(left_df))
print("% of employees who left =", 1.*len(left_df)/len(data)*100,"%")

print("No. of employees stayed =",len(stayed_df))
print("% of employee who stayed =",1.*len(stayed_df)/len(data)*100,"%")

# Let's compare the mean and std of the employees who stayed and left
print(left_df.describe())
print(stayed_df.describe())

"""Findings"""
# 'age': mean age of the employees who stayed is higher compared to who left
# 'DailyRate': Rate of employees who stayed is higher
# 'DistanceFromHome': Employees who stayed live closer to home
# 'EnvironmentSatisfaction' & 'JobSatisfaction': Employees who stayed are generally more satisifed with their jobs
# 'StockOptionLevel': Employees who stayed tend to have higher stock option


"""Plotting the employees who left and stayed based on their age"""

plt.figure(figsize=(25,12))
sns.countplot(x="Age",hue="Attrition",data=data)
# plt.show()
plt.close()

"""Plotting the employees who left and stayed based on JobRole,MaritalStatus,JobInvolvement,JobLevel"""

plt.figure(figsize=(15,10))
plt.subplot(411)
sns.countplot(x="JobRole",hue="Attrition",data=data)
plt.xticks(fontsize=8)
plt.tight_layout()

plt.subplot(412)
sns.countplot(x="MaritalStatus", hue="Attrition",data=data)

plt.subplot(413)
sns.countplot(x="JobInvolvement",hue="Attrition",data=data)

plt.subplot(414)
sns.countplot(x="JobLevel",hue="Attrition",data=data)

plt.tight_layout()
# plt.show()
plt.close()

"""Findings"""

# JobRole - Sales Representatives tend to leave compared to any other job
# MaritalStatus - Single employees tend to leave compared to married and divorced
# JobInvolvement - Less involved employees tend to leave the company
# JobLevel - Less experienced (low job level) tend to leave the company


"""KDE (Kernel Density Estimate) is used for visualizing the Probability Density of a continuous variable. 
KDE describes the probability density at different values in a continuous variable."""

plt.figure(figsize=(20,20))

sns.kdeplot(left_df["DistanceFromHome"],label="Employees who left",fill=True,color="r")
sns.kdeplot(stayed_df["DistanceFromHome"],label="Employee who stayed",fill=True,color="b")
plt.xlabel("Distance from home")
plt.legend()
# plt.show()
plt.close()

plt.figure(figsize=(15,10))

sns.kdeplot(left_df["TotalWorkingYears"],label="Employees who left",fill=True,color="g")
sns.kdeplot(stayed_df["TotalWorkingYears"],label="Employees who stayed",fill=True,color="r")
plt.legend()
# plt.show()
plt.close()

plt.figure(figsize=(10,5))

sns.kdeplot(left_df["YearsWithCurrManager"],label="Employees who left",fill=True,color="g")
sns.kdeplot(stayed_df["YearsWithCurrManager"],label="Employees who stayed",fill=True,color="b")
plt.legend()
# plt.show()
plt.close()


"""Findings"""
# DistanceFromHome - Employees who stay closer are tend to stay and employees who stay far from office tends to leave
# TotalWorkingYears - As total no. of working years increases people more likely to stay
# YearsWithCurrManager - As working with current manager increases people tend to stay

"""Using Boxplot for Gender vs MonthlyIncome and JobRole vs MonthlyIncome"""

sns.boxplot(x="MonthlyIncome",y="Gender",data=data)
# plt.show()
plt.close()

plt.figure(figsize=(10,5))
sns.boxplot(x="MonthlyIncome",y="JobRole",data=data)
# plt.show()
plt.close()

"""Findings"""
# Gender Equality maintained in Paycheck
# Manager gets lot of paid and Sales Representative gets lowest paid

"""Preprocessing before feeding data into Machine Learning Model"""
# Converting all categorical data into Numerical data using Onehot_encoder
# Scaling the data using MinMaxScaler for better prediction.

print(data.dtypes)
print(data.select_dtypes(include=["object"]).columns)

"""All categorical data are Non-Ordinal category, so using OneHotEncoding"""
from sklearn.preprocessing import OneHotEncoder

X_cat=(data[["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus"]])
print(X_cat)

OH_encoder=OneHotEncoder()
X_cat=OH_encoder.fit_transform(X_cat).toarray()
print(X_cat)

X_cat=pd.DataFrame(X_cat)
print(X_cat)
print(X_cat.shape)

print(data.select_dtypes(include=["int"]).columns)

"""Leaving Attrition in X, coz its going to be our Target Y"""

X_num=data[['Age','DailyRate', 'DistanceFromHome', 'Education',
       'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',
       'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
       'OverTime', 'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
       'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
       'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']]

print(X_num)

"""Concat Categorical & Numerical column"""

X_all=pd.concat([X_cat,X_num],axis=1)
print(X_all)

"""Scaling the data"""
from sklearn.preprocessing import MinMaxScaler

X_all.columns=X_all.columns.astype(str)

scaler=MinMaxScaler()
x=scaler.fit_transform(X_all)
print(x)

y=data["Attrition"]
print(y)

"""Training the model and predicting using Logistic regression"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
print(x_train.shape)
print(x_test.shape)

model=LogisticRegression()
model.fit(x_train,y_train)
prediction=model.predict(x_test)
print(prediction)

"""Checking accuracy using metrics"""

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

accuracy=accuracy_score(y_test,prediction)
print("Accuracy=",accuracy)

cm=confusion_matrix(y_test,prediction)
sns.heatmap(cm,annot=True)
# plt.show()
plt.close()

class_report=classification_report(y_test,prediction)
print(class_report)

"""Predicting using Random forest"""

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()
model.fit(x_train,y_train)
prediction=model.predict(x_test)
print(prediction)

c_m=confusion_matrix(y_test,prediction)
sns.heatmap(c_m,annot=True)
# plt.show()
plt.close()

classified_report=classification_report(y_test,prediction)
print(classified_report)


