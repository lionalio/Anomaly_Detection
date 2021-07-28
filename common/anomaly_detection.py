from libs import *
from data_preparation import *


def is_pos_def(mat):
    return np.all(np.linalg.eigvals(mat) > 0)


class AnomalyDetection(DataPreparation):
    def __init__(self, filename, features, label_col, delimiter=',', 
                single_file=True, data_type='tabular', test_size=0.2):
        super().__init__(filename, features, label_col, delimiter, single_file, data_type, test_size)
        #self.guide()

    def dist_mahalanobis(self):
        data = self.X
        cov_mat = np.array(np.cov(data.T))
        inv_cov_mat = None
        if is_pos_def(cov_mat):
            inv_mat = np.linalg.inv(cov_mat)
            if is_pos_def(inv_mat):
                inv_cov_mat = np.array(inv_mat)
            else:
                raise Exception('Error: Inversed covariance matrix is not positive definite!')
        else:
            raise Exception('Error: The covariance matrix is not positive definite!')

        means = data.mean(axis=0)
        diff = np.array((data - means))
        dist = []
        for i in range(len(diff)):
            dist.append(np.sqrt(diff[i].dot(inv_cov_mat).dot(diff[i])))

        return dist

    def find_anomaly_stat(self):
        outliers = []
        #try:
        d = np.array(self.dist_mahalanobis())
        thresh = np.mean(d) * 3
        for i in range(len(d)):
            if d[i] > thresh:
                outliers.append(d[i])
        plt.figure()
        sns.distplot(d,
                        bins = 50, 
                        kde= False, 
                        color = 'green')
        #plt.xlim([0.0,5])
        #plt.yscale('log')
        plt.xlabel('Mahalanobis dist')
        plt.show()
        #except:
        #    print('Cannot find anomaly using statistics')

        return outliers

    def find_anomaly_ML(self):
        results = {}       
        outlier_fraction = len(self.y[self.y == 1]) / len(self.y[self.y == 0])
        classifiers = [
            IsolationForest(n_estimators=100, max_samples=len(self.X), 
                            contamination=outlier_fraction,random_state=42, 
                            verbose=0),
            LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                leaf_size=30, metric='minkowski',
                                p=2, metric_params=None,
                                novelty=True, 
                                contamination=outlier_fraction)
        ]

        for clf in classifiers:
            clf.fit(self.X)
            y_pred = clf.predict(self.X)
            y_pred[y_pred == 1] = 0
            y_pred[y_pred == -1] = 1
            classname = clf.__class__.__name__
            results[classname] = accuracy_score(self.y, y_pred)
            print('Classification report for {}: '.format(classname))
            print(classification_report(self.y, y_pred))

        return results