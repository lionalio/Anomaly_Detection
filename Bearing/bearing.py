import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path + "/common")

from anomaly_detection import *

df = pd.read_csv('merged_dataset_BearingTest_2.csv', index_col=0)

#for f in df.columns:
#    plt.figure(figsize=(15,8))
#    df[f].plot()
#    plt.xlabel(f)
#    plt.show()

preprocessings = {
    'imputer': None,
    'mapping': None,
    'dim_reduce': False,
    #'preprocess': StandardScaler()
}

detector = AnomalyDetection('merged_dataset_BearingTest_2.csv', 
                            [f for f in df.columns], 
                            None,
                            data_type='timeseries')
detector.set_methods_process(preprocessings)
detector.processing()
#print(detector.find_anomaly_stat())
detector.find_anomaly_AE()