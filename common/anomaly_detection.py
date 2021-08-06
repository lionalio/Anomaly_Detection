from pandas.core.reshape.tile import cut
from libs import *
from data_preparation import *
from modules_DL import *


def is_pos_def(mat):
    return np.all(np.linalg.eigvals(mat) > 0)


class AnomalyDetection(DataPreparation):
    def __init__(self, filename, features, label_col=None, delimiter=',', 
                single_file=True, data_type='tabular', test_size=0.2):
        super().__init__(filename, features, label_col, delimiter, single_file, data_type, test_size)
        self.inv_cov_mat = None
        #self.guide()

    def get_inv_cov_mat(self):
        cov_mat = np.array(np.cov(self.X_train.T))
        inv_cov_mat = None
        if is_pos_def(cov_mat):
            inv_mat = np.linalg.inv(cov_mat)
            if is_pos_def(inv_mat):
                inv_cov_mat = np.array(inv_mat)
            else:
                raise Exception('Error: Inversed covariance matrix is not positive definite!')
        else:
            raise Exception('Error: The covariance matrix is not positive definite!')

        self.inv_cov_mat = inv_cov_mat

    def dist_mahalanobis(self, inv_cov_mat, means, data):
        diff = np.array((data - means))
        dist = []
        for i in range(len(diff)):
            dist.append(mahalanobis(diff[i], diff[i], inv_cov_mat))

        return dist

    def find_anomaly_stat(self):
        outliers = []
        # Let's get inversed covariance matrix and mean values from training set
        self.get_inv_cov_mat()
        means = self.X_train.mean(axis=0)
        
        data1 = self.X_train
        d = np.array(self.dist_mahalanobis(self.inv_cov_mat, means, data1))
        thresh = np.mean(d) * 3
        for i in range(len(d)):
            if d[i] > thresh:
                outliers.append(d[i])
        plt.figure()
        sns.distplot(d, bins = 50, kde= False, color = 'green')
        plt.xlabel('Mahalanobis dist')
        plt.show()

        data2 = self.X_test
        df_out1 = pd.DataFrame()
        df_out1['dist'] = d
        df_out1['thresh'] = thresh
        df_out2 = pd.DataFrame()
        df_out2['dist'] = np.array(self.dist_mahalanobis(self.inv_cov_mat, means, data2))
        df_out2['thresh'] = thresh
        df_out = pd.concat([df_out1, df_out2], ignore_index=True)
        df_out.plot(logy=True, figsize=(10, 6), ylim=[1e-1, 1e3])
        plt.show()

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
                                contamination=outlier_fraction),
            #DBSCAN(eps=3, min_samples=2)
            #OneClassSVM(gamma='auto')
        ]

        for clf in classifiers:
            if hasattr(clf, 'fit_predict') and callable(getattr(clf, 'fit_predict')):
                y_pred = clf.fit_predict(self.X)
            else:
                clf.fit(self.X)
                y_pred = clf.predict(self.X)
            y_pred[y_pred == 1] = 0
            y_pred[y_pred == -1] = 1
            classname = clf.__class__.__name__
            results[classname] = accuracy_score(self.y, y_pred)
            print('Classification report for {}: '.format(classname))
            print(classification_report(self.y, y_pred))

        return results

    def find_anomaly_AE(self):
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        autoencoder, decoder = model_auto_encoder2(self.X_train.shape[1])
        history = autoencoder.fit(X_train_scaled, X_train_scaled, batch_size=32, epochs=20, verbose=1)
        pred_train = autoencoder.predict(X_train_scaled)
        print(X_train_scaled[:5])
        print(pred_train[:5])
        pred_train = pd.DataFrame(pred_train, columns=self.df.columns)
        scored_train = pd.DataFrame()
        scored_train['loss_mae'] = np.mean(np.abs(pred_train - X_train_scaled), axis=1)
        scored_train['Threshold'] = 0.5

        sns.distplot(scored_train['loss_mae'], kde=True, bins=10)
        plt.show()

        X_test_scaled = scaler.transform(self.X_test)
        pred_test = autoencoder.predict(X_test_scaled)
        pred_test = pd.DataFrame(pred_test, columns=self.df.columns)
        scored_test = pd.DataFrame()
        scored_test['loss_mae'] = np.mean(np.abs(pred_test - X_test_scaled), axis=1)
        scored_test['Threshold'] = 0.5

        scored = pd.concat([scored_train, scored_test], ignore_index=True)
        scored.plot(logy=True, figsize=(10, 6), ylim=[1e-2, 1e2], color=['blue', 'red'])
        plt.show()