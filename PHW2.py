import pandas as pd
import numpy as np
import random
import warnings
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from pyclustering.cluster.clarans import clarans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from tqdm import tqdm

warnings.filterwarnings(action='ignore')

# Print all the columns of dataframe
pd.set_option('display.max_columns', None)

# Read dataset
df = pd.read_csv('housing.csv')

# Deal with NaN values
feature = list(df)
for col in feature:
    if df[col].isnull().sum() == 0:
        pass
    else:
        df[col].fillna(df[col].mean(), inplace=True)

# df_new for drop median_house_value
df_new = df.copy()
df_new = df_new.drop(['median_house_value'], axis=1)


# feature_selection function is for choosing two random features
# To use 2D plot we will use only 2 features
def feature_selection(data, num):
    feature_name = list(data)
    fit = random.sample(feature_name, num)
    data = data[fit]
    return data


# Encode_Scale function is for Encode and Scale with input data
def Encode_Scale(data, encoder, scaler):
    feature_names = list(data.select_dtypes(object))
    x = data.copy()

    # Check the encoder to do OrdinalEncoding or OneHotEncoding
    if len(feature_names) != 0 and encoder == 'OrdinalEncoder':
        encoder = OrdinalEncoder()
        x[feature_names] = pd.DataFrame(encoder.fit_transform(x[feature_names]))

    elif len(feature_names) != 0 and encoder == 'OneHotEncoder':
        encoder = OneHotEncoder(sparse=False)
        encoded = pd.DataFrame(encoder.fit_transform(x[feature_names]))
        x.drop(columns=feature_names, inplace=True)
        x = pd.concat((x, encoded), axis=1)

    # When finishing Encoding, scale the whole data
    x = scaler.fit_transform(x)

    return x


# Main semi-auto clustering function we make
def cluster_model(x, df, n_times=1):
    # x is a Modified input data (In this case drop median house value)
    # df is a Original input data
    # n_times is a number to try
    # Each of them are save positions for each algorithm
    # score means silhouette_score to compare which clustering model is better
    # Since the silhouette score has a value of -1 to 1, the initial value of scores are -1.
    # Each best models are stored in (algorithm_name)best variables

    KM_score = -1
    KM_data = None
    KM_features = []
    KM_param = None
    kmbest = None

    DB_score = -1
    DB_data = None
    DB_features = []
    dbbest = None

    EM_score = -1
    EM_data = None
    EM_features = []
    EM_param = None
    embest = None

    CLARANS_score = -1
    CLARANS_data = None
    CLARANS_features = []
    CLARANS_param = None
    claransbest = None

    MS_score = -1
    MS_data = None
    MS_features = []
    msbest = None

    # Try n_times to try various parameters and k
    for i in tqdm(range(n_times)):
        # cdict is a place to store random parameters of clustering model to use for every loop

        data = x.copy()
        cdict = {}

        # df_clarans is a dataframe to store result of clarans clustering
        df_clarans = data.copy()
        df_clarans['cluster_result'] = None
        df_clarans = df_clarans['cluster_result']

        # Use feature_selection function choose two random features
        data_selected = feature_selection(data, 2)
        cdict['selected_features'] = list(data_selected)

        # Select scaler to use randomly
        cdict['scaler'] = random.choice(
            [StandardScaler(),
             MinMaxScaler(),
             MaxAbsScaler(),
             RobustScaler()]
        )
        # Select encoder to use randomly
        cdict['encoder'] = random.choice(
            ['OrdinalEncoder',
             'OneHotEncoder']
        )

        # data_selected have a dataset with selected features from original dataset
        # with Encoded and Scaled data
        data_selected = Encode_Scale(data_selected, encoder=cdict['encoder'],
                                     scaler=cdict['scaler'])
        """
        data_selected = Encode_Scale(data_selected, encoder='OrdinalEncoder',
                                     scaler=cdict['scaler'])
        """

        # Select K (number of cluster) for Random 2~12
        cdict['n_clusters'] = random.randint(2, 12)

        # Start making models for each Algorithm
        # -----------------------------------------------------------------------------
        # KMeans
        cdict['kmeans_params'] = {'n_clusters': cdict['n_clusters'],
                                  'verbose': False
                                  }
        KM = KMeans(**cdict['kmeans_params'])
        preds = KM.fit_predict(data_selected)
        score = silhouette_score(data_selected, preds, metric='l2', random_state=42)

        # Repeat and update the model with a higher silhouette score than before.
        if score > KM_score:
            KM_score = score
            KM_data = data_selected
            KM_param = cdict['kmeans_params']
            KM_features = cdict['selected_features']
            kmbest = KM

        # -----------------------------------------------------------------------------
        # DBSCAN
        cdict['DBSCAN_params'] = {'eps': random.uniform(0.1, 10),
                                  'min_samples': random.randint(2, 100),
                                  'metric': 'minkowski',
                                  'p': random.randint(1, data_selected.shape[1]),
                                  }
        DBscan = DBSCAN(**cdict['DBSCAN_params'])
        preds = DBscan.fit_predict(data_selected)
        try:
            score = silhouette_score(data_selected, preds, metric='l2', random_state=42)
        except ValueError:
            score = -1

        # Repeat and update the model with a higher silhouette score than before.
        if score > DB_score:
            DB_score = score
            DB_data = data_selected
            DB_features = cdict['selected_features']
            dbbest = DBscan

        # -----------------------------------------------------------------------------
        # EM(GMM)
        cdict['covariance_type'] = random.choice(['full', 'tied', 'diag', 'spherical'])
        cdict['reg_covar'] = np.log(random.uniform(1, np.e ** 5))
        cdict['EM(GMM)_params'] = {'n_components': cdict['n_clusters'],
                                   'random_state': 42,
                                   'verbose': False,
                                   'reg_covar': cdict['reg_covar'],
                                   'covariance_type': cdict['covariance_type']}
        EM = GaussianMixture(**cdict['EM(GMM)_params'])
        preds = EM.fit_predict(data_selected)
        try:
            score = silhouette_score(data_selected, preds, metric='l2', random_state=42)
        except ValueError:
            score = -1

        # Repeat and update the model with a higher silhouette score than before.
        if score > EM_score:
            EM_score = score
            EM_data = data_selected
            EM_param = cdict['EM(GMM)_params']
            EM_features = cdict['selected_features']
            embest = EM

        # -----------------------------------------------------------------------------
        # CLARANS
        cdict['number_numlocal'] = random.randint(1, 3)
        cdict['number_maxneighbor'] = random.randint(2, 5)
        cdict['CLARANS_params'] = {'number_clusters': cdict['n_clusters'],
                                   'numlocal': cdict['number_numlocal'],
                                   'maxneighbor': cdict['number_maxneighbor']}
        CLARANS = clarans(data_selected.tolist(), **cdict['CLARANS_params'])
        CLARANS.process()

        clarans_pred = -np.ones(shape=data_selected.shape[0])
        for i, idx in enumerate(CLARANS.get_clusters()):
            clarans_pred[idx] = i
        try:
            score = silhouette_score(data_selected, clarans_pred,
                                     metric='l2', random_state=42)
        except ValueError:
            score = -1

        # Repeat and update the model with a higher silhouette score than before.
        if score > CLARANS_score:
            CLARANS_param = cdict['CLARANS_params']
            CLARANS_score = score
            CLARANS_data = data_selected
            CLARANS_features = cdict['selected_features']
            claransbest = CLARANS

        # -----------------------------------------------------------------------------
        # MeanShift
        cdict['meanshift_params'] = {'bandwidth': np.log(random.uniform(1, np.e ** 3)),
                                     'min_bin_freq': random.randint(1, 10),
                                     'n_jobs': 2}
        meanshift = MeanShift(**cdict['meanshift_params'])
        preds = meanshift.fit_predict(data_selected)
        try:
            score = silhouette_score(data_selected, preds, metric='l2', random_state=42)
        except ValueError:
            score = -1

        # Repeat and update the model with a higher silhouette score than before.
        if score > MS_score:
            MS_score = score
            MS_data = data_selected
            MS_features = cdict['selected_features']
            msbest = meanshift

    # Now for loop for n_times is finish

    # -----------------------------------------------------------------------------------------
    # Plot the result of Each clustering Algorithm

    # KMeans_plot
    print("KMeans")
    print(kmbest)
    print(KM_features)
    print(KM_score)
    print("\n")

    labels = kmbest.fit_predict(KM_data)
    n_clusters = int(KM_param['n_clusters'])
    plt.figure(1)
    plt.scatter(KM_data[:, 0], KM_data[:, 1], c=labels)
    plt.title('kmeans' + '(k = ' + str(n_clusters) + ')')
    plt.xlabel(KM_features[0])
    plt.ylabel(KM_features[1])
    plt.figure(2)
    df['quantile_MHV'] = pd.cut(df['median_house_value'],
                                bins=n_clusters, labels=np.arange(n_clusters))
    plt.scatter(KM_data[:, 0], KM_data[:, 1], c=df['quantile_MHV'])
    plt.title('Quantile' + '(k = ' + str(n_clusters) + ')')
    plt.xlabel(KM_features[0])
    plt.ylabel(KM_features[1])

    # -----------------------------------------------------------------------------------------
    # DBscan_plot
    print("DBSCAN")
    print(dbbest)
    print(DB_features)
    print(DB_score)
    print("\n")

    labels = dbbest.fit_predict(DB_data)
    labels_set = set(labels)
    labels_list = list(labels_set)
    n_clusters = len(labels_list)
    plt.figure(3)
    plt.scatter(DB_data[:, 0], DB_data[:, 1], c=labels)
    plt.title('DBSCAN' + '(k = ' + str(n_clusters) + ')')
    plt.xlabel(DB_features[0])
    plt.ylabel(DB_features[1])
    plt.figure(4)
    df['quantile_MHV'] = pd.cut(df['median_house_value'],
                                bins=n_clusters, labels=np.arange(n_clusters))
    plt.scatter(DB_data[:, 0], DB_data[:, 1], c=df['quantile_MHV'])
    plt.title('Quantile' + '(k = ' + str(n_clusters) + ')')
    plt.xlabel(DB_features[0])
    plt.ylabel(DB_features[1])

    # -----------------------------------------------------------------------------------------
    # EM_plot
    print("EM(GMM)")
    print(embest)
    print(EM_features)
    print(EM_score)
    print("\n")

    labels = embest.fit_predict(EM_data)
    n_clusters = int(EM_param['n_components'])
    plt.figure(5)
    plt.scatter(EM_data[:, 0], EM_data[:, 1], c=labels)
    plt.title('EM' + '(k = ' + str(n_clusters) + ')')
    plt.xlabel(EM_features[0])
    plt.ylabel(EM_features[1])
    plt.figure(6)
    df['quantile_MHV'] = pd.cut(df['median_house_value'],
                                bins=n_clusters, labels=np.arange(n_clusters))
    plt.scatter(EM_data[:, 0], EM_data[:, 1], c=df['quantile_MHV'])
    plt.title('Quantile' + '(k = ' + str(n_clusters) + ')')
    plt.xlabel(EM_features[0])
    plt.ylabel(EM_features[1])
    plt.show()

    # -----------------------------------------------------------------------------------------
    # CLARANS_plot
    print("CLARANS")
    print(CLARANS_param)
    print(CLARANS_features)
    print(CLARANS_score)
    print("\n")

    labels = claransbest.get_clusters()
    n_clusters = int(CLARANS_param['number_clusters'])
    for j in range(0, n_clusters):
        a = list(labels[j])
        for index in a:
            df_clarans.loc[index] = j

    clarans_list = list(df_clarans)

    plt.figure(7)
    plt.scatter(CLARANS_data[:, 0], CLARANS_data[:, 1], c=clarans_list)
    plt.title('CLARANS' + '(k = ' + str(CLARANS_param['number_clusters']) + ')')
    plt.xlabel(CLARANS_features[0])
    plt.ylabel(CLARANS_features[1])
    plt.figure(8)
    df['quantile_MHV'] = pd.cut(df['median_house_value'],
                                bins=n_clusters, labels=np.arange(CLARANS_param['number_clusters']))
    plt.scatter(CLARANS_data[:, 0], CLARANS_data[:, 1], c=df['quantile_MHV'])
    plt.title('Quantile' + '(k = ' + str(CLARANS_param['number_clusters']) + ')')
    plt.xlabel(CLARANS_features[0])
    plt.ylabel(CLARANS_features[1])
    plt.show()

    # -----------------------------------------------------------------------------------------
    # MeanShift_plot
    print("MeanShift")
    print(msbest)
    print(MS_features)
    print(MS_score)
    print("\n")

    labels = msbest.fit_predict(MS_data)
    labels_set = set(labels)
    labels_list = list(labels_set)
    n_clusters = len(labels_list)
    plt.figure(9)
    plt.scatter(MS_data[:, 0], MS_data[:, 1], c=labels)
    plt.title('Mean Shift' + '(k = ' + str(n_clusters) + ')')
    plt.xlabel(MS_features[0])
    plt.ylabel(MS_features[1])
    plt.figure(10)
    df['quantile_MHV'] = pd.cut(df['median_house_value'],
                                bins=n_clusters, labels=np.arange(n_clusters))
    plt.scatter(MS_data[:, 0], MS_data[:, 1], c=df['quantile_MHV'])
    plt.title('Quantile' + '(k = ' + str(n_clusters) + ')')
    plt.xlabel(MS_features[0])
    plt.ylabel(MS_features[1])
    plt.show()


# Start cluster_model function
cluster_model(df_new, df)