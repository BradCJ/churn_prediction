import pandas as pd
import graphviz
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


# Import Churn Data
df = pd.read_csv(r'/mnt/c/Users/hpt-de/Documents/Central_Project/ML_Models/Churn_Prediction/CHURN_DATA.csv', dtype='unicode')


# Separate Feature and Target
X = df[["OLYMPIC_SEGMENT","TOTAL","GP","BUSKET_SIZE","TOTAL_PRODUCT","LAST_VISIT_DATE","LAPSED_DAYS","CFH_VISIT","CFH_TOTAL_VISIT","CFH_LAST_VISIT_DATE","TD_VISIT","TD_TOTAL_VISIT","TD_LAST_VISIT_DATE","TM_VISIT","TM_TOTAL_VISIT","TM_LAST_VISIT_DATE"]]
Y = df[["CHERN_FLAG"]]


# Encode Churn Data to Binary Format
label_encoder = LabelEncoder()

# Encoding X
encoding_x = ["OLYMPIC_SEGMENT","TOTAL","GP","BUSKET_SIZE","TOTAL_PRODUCT","LAST_VISIT_DATE","LAPSED_DAYS","CFH_VISIT","CFH_TOTAL_VISIT","CFH_LAST_VISIT_DATE","TD_VISIT","TD_TOTAL_VISIT","TD_LAST_VISIT_DATE","TM_VISIT","TM_TOTAL_VISIT","TM_LAST_VISIT_DATE"]
x_encoded = pd.DataFrame()
for i in encoding_x:
    x_encoded[i] = pd.to_numeric(label_encoder.fit_transform(X[i]))
print(x_encoded.head())

#  Encoding Y
y_encoded = pd.DataFrame()
y_encoded["CHERN_FLAG"] = pd.to_numeric(label_encoder.fit_transform(Y['CHERN_FLAG']))
print(y_encoded.head())


# Create New DataFrame
df = pd.concat([x_encoded,y_encoded], axis=1)


# Pair Plot Graph
sns.pairplot(df, vars=["OLYMPIC_SEGMENT","TOTAL","GP","BUSKET_SIZE","TOTAL_PRODUCT","LAST_VISIT_DATE","LAPSED_DAYS","CFH_VISIT","CFH_TOTAL_VISIT","CFH_LAST_VISIT_DATE","TD_VISIT","TD_TOTAL_VISIT","TD_LAST_VISIT_DATE","TM_VISIT","TM_TOTAL_VISIT","TM_LAST_VISIT_DATE","CHERN_FLAG"], hue="CHERN_FLAG")


# Transform Data with Scaler 
scaler = StandardScaler()
scaler.fit(x_encoded)
x_sca = scaler.transform(x_encoded)
x_sca = pd.DataFrame(x_sca, columns=["OLYMPIC_SEGMENT","TOTAL","GP","BUSKET_SIZE","TOTAL_PRODUCT","LAST_VISIT_DATE","LAPSED_DAYS","CFH_VISIT","CFH_TOTAL_VISIT","CFH_LAST_VISIT_DATE","TD_VISIT","TD_TOTAL_VISIT","TD_LAST_VISIT_DATE","TM_VISIT","TM_TOTAL_VISIT","TM_LAST_VISIT_DATE"])


# Train and Prediction Model
X_train, X_test, y_train, y_test = train_test_split(x_sca, y_encoded, train_size = .8, random_state = 101)

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

print(pd.DataFrame(model_score, columns=['score', 'model']).sort_values(by='score', ascending=False))


# Plot Tree Graph
clf = tree.DecisionTreeClassifier()
model = clf.fit(x_sca, y_encoded)

fn = ["OLYMPIC_SEGMENT","TOTAL","GP","BUSKET_SIZE","TOTAL_PRODUCT","LAST_VISIT_DATE","LAPSED_DAYS","CFH_VISIT","CFH_TOTAL_VISIT","CFH_LAST_VISIT_DATE","TD_VISIT","TD_TOTAL_VISIT","TD_LAST_VISIT_DATE","TM_VISIT","TM_TOTAL_VISIT","TM_LAST_VISIT_DATE"]

tree.plot_tree(model, 
               feature_names = fn, 
               filled=True)

tree.export_graphviz(model,
                    out_file=r'/mnt/c/Users/hpt-de/Documents/Central_Project/ML_Models/Churn_Prediction/D_Tree.dot',
                    feature_names = fn,
                    filled = True)

dot_data = tree.export_graphviz(model,
                    out_file=None,
                    feature_names = fn,
                    filled = True)
graph = graphviz.Source(dot_data)
graph


# Status Detail
model_best = sm.Logit(y_encoded,x_sca).fit()
model_best.summary2()
