import os
import pandas as pd
import numpy as np

def merge_data_from_path(datapath, columns, name):
    merged_data = pd.DataFrame()
    for filename in os.listdir(datapath):
        dataset=pd.read_csv(os.path.join(datapath, filename), sep='\t')
        dataset_mean_abs = np.array(dataset.abs().mean())
        ncols = dataset.shape[1]
        dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1,ncols))
        dataset_mean_abs.index = [filename]
        merged_data = merged_data.append(dataset_mean_abs)

    merged_data.columns = columns
    merged_data.describe()
    merged_data.index = pd.to_datetime(merged_data.index, format='%Y.%m.%d.%H.%M.%S')
    merged_data = merged_data.sort_index()
    merged_data.to_csv(name)
    merged_data.head()

if __name__ == '__main__':
    datapaths = [
        'bearing_datasets/1st_test', 
        'bearing_datasets/2nd_test', 
        'bearing_datasets/4th_test'
    ]
    column_names = [
        ['Bearing {}'.format(i) for i in range(1, 9)],
        ['Bearing {}'.format(i) for i in range(1, 5)],
        ['Bearing {}'.format(i) for i in range(1, 5)]
    ]
    save_names = ['merged_dataset_BearingTest_{}.csv'.format(i) for i in range(1, 4)]

    for path, columns, save_name in zip(datapaths, column_names, save_names):
        print(path, columns, save_name)
        merge_data_from_path(path, columns, save_name)
