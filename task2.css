import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
data = pd.read_csv("/Titanic-Dataset.csv")
# Data Cleaning
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Cabin'] = data['Cabin'].fillna("Unknown")
data.drop(['Ticket', 'Name'], axis=1, inplace=True)
# Encode categorical variables
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)
# Drop the 'Cabin' column before calculating correlation
data_for_corr = data.drop(columns=['Cabin']) # Drop 'Cabin' as it's non-numeric
# EDA: Univariate Analysis
plt.figure(figsize=(6, 4))
data['Survived'].value_counts().plot(kind='bar', title="Survival Count")
plt.show()
# EDA: Bivariate Analysis
sns.barplot(x="Pclass", y="Survived", data=data)
plt.title("Survival by Passenger Class")
plt.show()
# Correlation Analysis
plt.figure(figsize=(10, 8))
sns.heatmap(data_for_corr.corr(), annot=True, cmap="coolwarm") # Use the modified DataFrame
plt.title("Feature Correlation")
plt.show()
