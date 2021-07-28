import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path + "/common")

from anomaly_detection import *

df = pd.read_csv('creditcard.csv')

preprocessings = {
    'imputer': None,
    'mapping': None,
    'dim_reduce': False,
    #'preprocess': StandardScaler()
}

detector = AnomalyDetection('creditcard.csv', 
                            [f for f in df.columns if f != 'Class' and f != 'Time'], 
                            'Class')
detector.set_methods_process(preprocessings)
detector.processing()
print(detector.find_anomaly_ML())