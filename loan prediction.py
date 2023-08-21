import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv("LoanApprovalPrediction.csv")


##Data Preprocessing and Visualization
##Get the number of columns of object datatype.

obj = (data.dtypes == 'object')
##print("Categorical variables:",len(list(obj[obj].index)))


# Dropping Loan_ID column
data.drop(['Loan_ID'],axis=1,inplace=True)


#label encoding
# Import label encoder
# label_encoder object knows how
# to understand word labels.
label_encoder = LabelEncoder()
obj = (data.dtypes == 'object')

for col in list(obj[obj].index):
    data[col] = label_encoder.fit_transform(data[col])


#missing values
for col in data.columns:
    data[col] = data[col].fillna(data[col].mean())
    data.isna().sum()


#splitting data set

from sklearn.model_selection import train_test_split

X = data.drop(['Loan_Status'],axis=1)
Y = data['Loan_Status']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.4,random_state=1)


#accuracy

models = {"random_forest": RandomForestClassifier(n_estimators=100)}

model = models['random_forest']
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
print("Accuracy score of ",model.__class__.__name__,"=",100*metrics.accuracy_score(Y_test,Y_pred))


# single data prediction

new_gender=input("your gender:?")
new_married_sts=input("married or unmarried")
dependent_count=int(input("your dependency"))
education_sts=input("graduate or not graduate")
self_emp_sts=input("are you self_employed")
applicant=int(input("your income"))
co_applicant=int(input("your co applicant income"))
loan_amount=int(input("loan amount"))
loan_amount_term=int(input("loan amount term"))
credit=int(input("credit history"))
property_area=input("property location")


new_gender_transform=label_encoder.fit_transform([new_gender])
new_married_sts_transform=label_encoder.fit_transform([new_married_sts])
education_sts_transform=label_encoder.fit_transform([education_sts])
self_emp_sts_transform=label_encoder.fit_transform([self_emp_sts])
property_area_transform=label_encoder.fit_transform([property_area])

prediction=model.predict([[new_gender_transform[0],new_married_sts_transform[0],dependent_count,education_sts_transform[0],self_emp_sts_transform[0],applicant,co_applicant,loan_amount,loan_amount_term,credit,property_area_transform[0]]])
print(prediction)


###improve accuracy method
##import pandas as pd
##import numpy as np
##from sklearn.feature_selection import SelectKBest
##from sklearn.feature_selection import chi2
##data = pd.read_csv("D://Blogs//train.csv")
##X = data.iloc[:,0:20]  #independent columns
##y = data.iloc[:,-1]    #target column i.e price range
###apply SelectKBest class to extract top 10 best features
##bestfeatures = SelectKBest(score_func=chi2, k=10)
##fit = bestfeatures.fit(X,y)
##dfscores = pd.DataFrame(fit.scores_)
##dfcolumns = pd.DataFrame(X.columns)
###concat two dataframes for better visualization 
##featureScores = pd.concat([dfcolumns,dfscores],axis=1)
##featureScores.columns = ['Specs','Score']  #naming the dataframe columns
##print(featureScores.nlargest(10,'Score'))  #print 10 best features
##

