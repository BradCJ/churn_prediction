from matplotlib.pyplot import fill
import pandas as pd
from scipy.fftpack import sc_diff
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


df = pd.read_csv(r'/mnt/c/Users/hpt-de/Documents/Central_Project/ML_Models/Churn_Prediction/CHURN_DATA.csv', dtype='unicode')
# print(df[['CHERN_FLAG']])

X = df[['TD_VISIT','CFH_VISIT','TM_VISIT','TOTAL']]
Y = df[['CHERN_FLAG']]


label_encoder = LabelEncoder()

# Encoding X
encoding_x = ['TD_VISIT','CFH_VISIT','TM_VISIT','TOTAL']
x_encoded = pd.DataFrame()
for i in encoding_x:
    x_encoded[i] = pd.to_numeric(label_encoder.fit_transform(X[i]))
print(x_encoded.head())

#  Encoding Y
y_encoded = pd.DataFrame()
y_encoded['CHERN_FLAG'] = pd.to_numeric(label_encoder.fit_transform(Y['CHERN_FLAG']))
print(y_encoded.head())

# scaler = StandardScaler()
# scaler.fit(x_encoded)
# x_sca = scaler.transform(x_encoded)
# x_sca = pd.DataFrame(x_sca, columns=['TD_VISIT','CFH_VISIT','TM_VISIT','TOTAL'])

# sns.pairplot(x_sca.iloc[:, 0:5])


X_train, X_test, y_train, y_test = train_test_split(x_encoded, y_encoded, train_size = .8, random_state = 101)

algo = [
    [LogisticRegression(solver='lbfgs'), 'LogisticRegression'],
    [tree.DecisionTreeClassifier(), 'DecisionTreeClassifier'],
    [GradientBoostingClassifier(), 'GradientBoostingClassifier'],
    [RandomForestClassifier(), 'RandomForestClassifier']
]
model_score=[]
for a in algo:
    model=a[0]
    model.fit(X_train, y_train) 
    y_pred=model.predict(X_test) 
    score=model.score(X_test, y_test)
    model_score.append([score, a[1]])
    print(f'{a[1]} score = {score}') 
    print(metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))
    print('-' * 100)
# print(model_score)

print(pd.DataFrame(model_score, columns=['score', 'model']).sort_values(by='score', ascending=False))


# clf = tree.DecisionTreeClassifier()
# model = clf.fit(x_encoded, y_encoded)

# print(model)


# fn = ['TD_VISIT','CFH_VISIT','TM_VISIT','TOTAL']

# tree.plot_tree(model,
#                 feature_names = fn,
#                 filled=True)

# tree.export_graphviz(model,
#                     out_file=r'/mnt/c/Users/hpt-de/Documents/Central_Project/ML_Models/Churn_Prediction/tree.dot',
#                     feature_names = fn,
#                     filled = True)
                