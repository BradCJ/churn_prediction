import numpy as np
import pandas as pd
import pytds
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


# Connect SQL Server
def load_data():
    conn = pytds.connect('####', '####', '####', '####')
    cur = conn.cursor()
    data = cur.execute(
        """"
        SELECT
            ID
            ,CHERN_FLAG
            ,OLYMPIC_SEGMENT
            ,TOTAL
            ,GP
            ,BUSKET_SIZE
            ,TOTAL_PRODUCT
        FROM ####.####.####
        """
    )
    rows = data.fetchall()
    df = pd.DataFrame(rows)
    df.columns = ['ID','CHURN_FLAG','OLYMPIC_SEGMENT','TOTAL','GP','BUSKET_SIZE','TOTAL_PRODUCT']
    cur.close()
    conn.close()
    return df
df = load_data()
print(df.head())

# Separate Feature and Target
ID = df[['ID']]
X = df[['ID','OLYMPIC_SEGMENT','TOTAL','GP','BUSKET_SIZE','TOTAL_PRODUCT']]
Y = df[['CHURN_FLAG']]


# Encode Churn Data to Binary Format
label_encoder = LabelEncoder()

# Encoding ID
encoding_id = pd.DataFrame()
encoding_id['ID'] = pd.to_numeric(label_encoder.fit_transform(ID['ID']))
encoding_id.columns = ['ENCODED_ID']
Master_ID_encoded = pd.concat([ID,encoding_id], axis=1)
print(Master_ID_encoded.head())

# Encoding X
encoding_x = ['ID','OLYMPIC_SEGMENT','TOTAL','GP','BUSKET_SIZE','TOTAL_PRODUCT']
x_encoded = pd.DataFrame()
for i in encoding_x:
    x_encoded[i] = pd.to_numeric(label_encoder.fit_transform(X[i]))
print(x_encoded.head())

#  Encoding Y
y_encoded = pd.DataFrame()
y_encoded['CHURN_FLAG'] = pd.to_numeric(label_encoder.fit_transform(Y['CHURN_FLAG']))
print(y_encoded.head())


# Create New DataFrame
df = pd.concat([x_encoded,y_encoded], axis = 1)


## Pair Plot Graph
# sns.pairplot(df, vars=['OLYMPIC_SEGMENT','TOTAL','GP','BUSKET_SIZE','TOTAL_PRODUCT','CHURN_FLAG'], hue='CHURN_FLAG')


# # Transform to Scaled Data
# scaler = StandardScaler()
# x_sca = scaler.fit_transform(x_encoded.T)
# x_sca = pd.DataFrame(x_sca, columns = ['OLYMPIC_SEGMENT','TOTAL','GP','BUSKET_SIZE','TOTAL_PRODUCT'])


# Train and Predict
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y_encoded, train_size = .8, random_state = 13, stratify = Y)

algo = [
    # [LogisticRegression(solver='lbfgs'), 'LogisticRegression'],
    # [tree.DecisionTreeClassifier(), 'DecisionTreeClassifier'],
    # [RandomForestClassifier(), 'RandomForestClassifier'],
    [GradientBoostingClassifier(), 'GradientBoostingClassifier']
]
model_score=[]
for a in algo:
    model=a[0]
    model.fit(x_train, y_train.values.ravel())
    y_pred=model.predict(x_test)
    result_x = pd.DataFrame(x_test).reset_index(drop=True)
    result_y = pd.DataFrame(y_test).reset_index(drop=True)
    concat_result = pd.concat([result_x,result_y], axis=1)
    result = concat_result.merge(Master_ID_encoded, left_on='ID', right_on='ENCODED_ID', suffixes=('_Encoded', ''))
    export_result = result.to_csv(r'/mnt/c/Users/####/ML_Models/Churn_Prediction/Churn_Result.csv', index = False)
    score=model.score(x_test, y_test)
    model_score.append([score, a[1]])
    print(f'{a[1]} score = {score}')
    print(metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))
    print('_' * 130)
    print(result.head())

print(pd.DataFrame(model_score, columns = ['score', 'model']).sort_values(by = 'score', ascending = False))


## Plot Tree Graph
# clf = tree.DecisionTreeClassifier()
# model = clf.fit(x_encoded, y_encoded)

# fn = ['OLYMPIC_SEGMENT','TOTAL','GP','BUSKET_SIZE','TOTAL_PRODUCT']

# tree.export_graphviz(model,
#                     out_file = r'/mnt/c/Users/####/ML_Models/Churn_Prediction/D_Tree.dot',
#                     feature_names = fn,
#                     filled = True)

# dot_data = tree.export_graphviz(model,
#                     out_file = None,
#                     feature_names = fn,
#                     filled = True)
# graph = graphviz.Source(dot_data)
# graph


# Status Detail
model_best = sm.Logit(y_encoded,x_encoded).fit()
model_best.summary2()
