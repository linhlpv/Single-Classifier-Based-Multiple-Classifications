import pandas as pd 
import numpy as np 
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score

from sklearn import preprocessing

def test_matrix_mul(u, v):
    """
    Just a test function. Never mind!
    """
    n, f = u.shape
    m, _ = v.shape
    u_ = np.expand_dims(u, 2) # n, f, 1
    u_ = np.repeat(u_, m, 2) # n, f, m
    v_ = v.transpose(1, 0) # f, m
    v_ = np.expand_dims(v_, 0)
    v_ = np.repeat(v_, n, 0) # n, f, m
    uv_ = u_ * v_
    uv_ = uv_.sum(1)
    dot = u.dot(v.transpose(1, 0))   
    print(uv_ - dot)

def point_generator(xi, xk, alpha=0.5):
    """
    Params
        xi: reference point
        xk: validate or test point
        alpha: weight for combine points
    Return
        x_hat: new data point
    """
    x_hat = alpha*xi + (1-alpha)*xk
    return x_hat

def measure_valid_pair(xik, yk, classifier):
    """
    Params
        xik: pseudo point that generate from reference xi and valid/test xk
        yk: lable of xk
        classifier: classifier Cx
    Return
        valid_score: 0 or 1 for pair(xi, xk)
    """
    yik = classifier.predict([xik]) # [label]
    return int(yik[0] == yk)

def create_pseudo_data_references_test(X_train, xt, alpha=0.5):
    """
    Params
        X_train: Reference datapoints are choosen for generate pseudo test points
        xt: test point
        alpha: combined weight
    """
    return alpha*X_train + (1-alpha)*(np.repeat(np.expand_dims(xt, 0), X_train.shape[0], 0))

def create_pseudo_data_valid_reference(X_train, X_valid, alpha=0.5):
    """
    *Fast implementation for generate pseudo data point with reference and valid datapoints*
    Params
        X_train: reference points with shape N, F
        X_valid: valid points with shape M, F
        alpha: combine weight
    Return
        pseudo_datapoints: all pseudo datapoints N, M, F
    """
    N, F = X_train.shape
    M, _ = X_valid.shape
    X_train_ = np.repeat(np.expand_dims(X_train, 2), M, 2) #N, F, M
    X_valid_ = np.repeat(np.expand_dims(X_valid.transpose(1, 0), 0), N, 0) #N, F, M
    X_train_ = alpha*X_train_
    X_valid_ = (1-alpha)*X_valid_
    pseudo_datapoints = X_train_ + X_valid_ # N, F, M
    return pseudo_datapoints.transpose(0, 2, 1) #N, M, F


def gen_pseudo_data_points(X_train, X_valid, y_valid, xt, classifier, n=9, m=9, p=0.85, alpha=0.5):
    """
    Params
        X_train: Reference datapoints with shape NxF
        X_valid: Valid datapoints with shape MxF
        y_valid: Label of valid datapoints with shape M
        xt: test datapoint with shape F
        classifier: A single classifier Cx
        n: number of k nearest reference samples of test datapoint
        m: number of k nearest valid samples of test datapoint
        alpha: generate datapoint weight
        p: phi threshold weight
    """
    N, F = X_train.shape
    M, _ = X_valid.shape
    pseudo_datapoints = create_pseudo_data_valid_reference(X_train, X_valid, alpha) #N, M, F

    # calculate valid pairs
    valid_pairs = np.zeros((N, M))
    pseudo_datapoints_labels = classifier.predict(pseudo_datapoints.reshape(-1, F))
    pseudo_datapoints_labels = pseudo_datapoints_labels.reshape(N, -1)
    valid_label = np.repeat(np.expand_dims(y_valid, 0), N, 0) #N, M
    valid_pairs = (pseudo_datapoints_labels == valid_label).astype(np.int) # N, M

    # calculate k nearest points 
    neigh_train = NearestNeighbors(n_neighbors=m+10, metric='euclidean')
    neigh_train = neigh_train.fit(X_train)
    neigh_valid = NearestNeighbors(n_neighbors=m+10, metric='euclidean')
    neigh_valid = neigh_valid.fit(X_valid)
    
    # get index of m nearest points of valid points with test xt
    m_nearest_valid_points_index = neigh_valid.kneighbors([xt], n_neighbors=m, return_distance=False)
    m_nearest_valid_points = X_valid[m_nearest_valid_points_index] 
    n_nearest_train_points_index = neigh_train.kneighbors([xt], n_neighbors=n, return_distance=False)

    # extract pair value for m nearest valid
    valid_pairs_of_m_nearest_valid = valid_pairs[:, m_nearest_valid_points_index].reshape(N, -1) #N, m

    # calculate phi weight
    # calculate distance matrix of reference and valid
    reference_valid_distance_matrix = sklearn.metrics.pairwise_distances(X_train, X_valid) #N, M
    m_nearest_valid_pairs_distance = reference_valid_distance_matrix[:, m_nearest_valid_points_index].reshape(N, -1) #N, m
    reference_testpoint_distance = sklearn.metrics.pairwise.euclidean_distances(X_train, np.repeat(np.expand_dims(xt, 0), N, 0))[:, 0] # N
    valid_testpoint_distance = sklearn.metrics.pairwise.euclidean_distances(X_valid, np.repeat(np.expand_dims(xt, 0), N, 0))[:, 0] # M
    m_nearest_valid_testpoint_distance = valid_testpoint_distance[m_nearest_valid_points_index]
    phi_weights = m_nearest_valid_pairs_distance + m_nearest_valid_testpoint_distance
    phi_weights = phi_weights * (1/reference_testpoint_distance).reshape(N, 1)

    # calculate weight of reference datapoints and threshold 
    weights = valid_pairs_of_m_nearest_valid * phi_weights
    weights = weights.sum(1)
    threshold = p * weights.max()

    # select reference instance for generate pseudo test point
    n_nearest_reference_point_weights = weights[n_nearest_train_points_index]
    reference_datapoints_selected_index = n_nearest_train_points_index[n_nearest_reference_point_weights>=threshold]

    # create pseudo datapoints from reference datapoints that choosen
    pseudo_datapoints = create_pseudo_data_references_test(X_train[reference_datapoints_selected_index], xt)
    return pseudo_datapoints

def predict(pseudo_datapoints, classifier):
    """
    Params
        pseudo_datapoints: pseudo test datapoints 
        classifier: single classifier Cx
    """
    preds = classifier.predict_proba(pseudo_datapoints)
    preds = preds.mean(0)
    # print(preds, np.argmax(preds))
    return np.argmax(preds)

def predict_all(X_train, X_valid, y_valid, X_test, classifier, n=9, m=9, p=0.85, alpha=0.5):
    """
    Params
        X_train: Reference datapoints with shape NxF
        X_valid: Valid datapoints with shape MxF
        X_test: Test datapoints
        y_valid: Label of valid datapoints with shape M
        classifier: A single classifier Cx
        n: number of k nearest reference samples of test datapoint
        m: number of k nearest valid samples of test datapoint
        alpha: generate datapoint weight
        p: phi threshold weight
    """
    y_preds = []
    for xt in X_test:
        pseudo_datapoints = gen_pseudo_data_points(X_train, X_valid, y_valid, xt, classifier)
        y_pred = predict(pseudo_datapoints, classifier)
        y_preds.append(y_pred)
    return np.array(y_preds)



if __name__ == '__main__':
    # u = np.random.rand(2, 3)
    # v = np.random.rand(4, 3)
    # test_matrix_mul(u, v)

    # Load dataset
    data = pd.read_csv("iris.csv")
    le = preprocessing.LabelEncoder()
    data['label'] = le.fit_transform(data['category'].values)
    X, y = data[['feature1', 'feature2', 'feature3', 'feature4']].values, data['label'].values

    # split train test valid dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1, test_size=0.33)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify=y_train, random_state=1, test_size=0.5)  

    # Prepaired classifier
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train) 
    mlp = MLPClassifier(random_state=1, max_iter=300)
    mlp.fit(X_train, y_train)

    # Acc of single classifier with-out 
    y_pred = neigh.predict(X_test)
    print(f'acc score of knn in test dataset is {accuracy_score(y_pred, y_test)}')
    y_pred = mlp.predict(X_test)
    print(f'acc score of mlp in test dataset is {accuracy_score(y_pred, y_test)}')

    # Acc of single classifier with
    y_preds = predict_all(X_train, X_valid, y_valid, X_test, mlp)
    print(f'acc score of single based multiple classifier - mlp in test dataset is {accuracy_score(y_preds, y_test)}')
    y_preds = predict_all(X_train, X_valid, y_valid, X_test, neigh)
    print(f'acc score of single based multiple classifier - knn in test dataset is {accuracy_score(y_preds, y_test)}')
