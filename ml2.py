import pandas as pd
df=pd.read_csv('blood_donation_eligibility_500.csv')
df.head()
x= df[['Age', 'Weight (kg)', 'Hemoglobin (g/dL)', 
        'Last Donation (months)', 'Disease History (Yes=1/No=0)']]

y=df['Eligible']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)  
model.score(x_test,y_test)
from sklearn.metrics import accuracy_score
y_pred=model.predict(x_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))