from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#read in data
df = pd.read_csv('train.csv',sep=',')
df.columns = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']

#store names and delete from dataframe
names = df.loc[:,['PassengerId','Name']]
del df['Name']

#convert Sex to 0 for female and 1 for male
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'].values)

#fill in missing values
df['Age'].fillna(df['Age'].mean(),inplace=True)

cols = ['Sex','Age','Pclass','SibSp','Parch','Fare']

print(df.shape[0])

sns.set(style='whitegrid',context='notebook')
sns.pairplot(df[cols].dropna(),size=2.5)
plt.show()




