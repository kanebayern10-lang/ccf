import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
ccdf=pd.read_csv('creditcard.csv')
ccdf.head()
legit=ccdf[ccdf.Class==0]
fraud=ccdf[ccdf.Class==1]
# legit.value_counts()
fraud

legitsample=legit.sample(n=85)
legitsample
creaditcarddf=pd.concat([legitsample,fraud],axis=0)
creaditcarddf['Class'].value_counts()
creaditcarddf.groupby('Class').mean()

x=creaditcarddf.drop(columns='Class',axis=1)
y=creaditcarddf['Class']
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("Features scaled successfully!")
model_scaled = LogisticRegression(max_iter=1000) # Increased max_iter
model_scaled.fit(x_train_scaled, y_train)

ypred_scaled = model_scaled.predict(x_test_scaled)
accuracy_scaled = accuracy_score(ypred_scaled, y_test)

print(f"Accuracy after scaling and increasing max_iter: {accuracy_scaled}")