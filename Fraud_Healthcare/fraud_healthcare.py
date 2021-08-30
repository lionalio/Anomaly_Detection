import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path + "/common")

from anomaly_detection import *

#t = pd.read_csv("Train-1542865627584.csv")
#b = pd.read_csv("Train_Beneficiarydata-1542865627584.csv")
#i = pd.read_csv("Train_Inpatientdata-1542865627584.csv")
#o = pd.read_csv("Train_Outpatientdata-1542865627584.csv")

#df = pd.concat([i,o])
#df = pd.merge(df, t, on="Provider", how="outer")
#df = df.fillna(0)

#df.to_csv('healthcare.csv', index_label=False)

df = pd.read_csv('healthcare.csv')
print(df)
#eda_ = EDA('healthcare.csv', label='PotentialFraud')
#eda_.dump()

mapping = {
    'PotentialFraud': {'No': 0, 'Yes': 1},
}

preprocessings = {
    #'imputer': SimpleImputer(strategy='most_frequent'),
    #'mapping': mapping,
    #'dim_reduce': False,
    #'remove_high_corr': False,
    'encoder': LabelEncoder(),
    'cat_encoder': None,
    'preprocess': None,
}

classifiers = [
    LogisticRegression(solver='lbfgs'),
    DecisionTreeClassifier(),
    xgb.XGBClassifier(),  # Take insanely long time to run. Why???
    RandomForestClassifier()
]

parameters = [
    # Logistic Regression
    {
        'penalty': ['l2'],
        'C': (0.01, 5., 'log-uniform'),
        'max_iter': (10, 100)
    },
    # Decision Tree
    {
        'criterion':['gini','entropy'],
        'max_depth': (5, 50)
    },
    # XGBoost
    {
        "learning_rate": (0.01, 1., 'log-uniform'),
        "max_depth": (3, 10),
        "min_child_weight": (1, 10),
        "gamma": (0.0, 1, 'uniform'),
        'eval_metric': ['mlogloss']
    },
    # Random Forest
    {
        'bootstrap': [True, False],
         'max_depth': (10, 100),
         'max_features': ['auto', 'sqrt'],
         'min_samples_leaf': (1, 10),
         'min_samples_split': (2, 10),
         'n_estimators':(100, 200, 500, 1000)
    }
]

drops = ['BeneID', 'ClaimID'] #, 'policy_number','policy_bind_date', 'incident_date','incident_location','auto_model', 
        #'fraud_reported', '_c39']

detector = AnomalyDetection('healthcare.csv', 
                        [f for f in df.columns if f != 'PotentialFraud' and f not in drops], 
                        label_col='PotentialFraud')
print(detector.df.info())
detector.set_methods_process(preprocessings)
detector.set_mapping(mapping)
detector.processing()
#detector.find_anomaly_VAE()
detector.find_anomaly_ML()