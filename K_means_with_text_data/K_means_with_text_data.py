from numpy.lib.function_base import vectorize
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances

def k_means_with_text_data():
    save_file('https://homes.cs.washington.edu/~hschafer/cse416/a6/people_wiki.csv', 'people_wiki.csv')
    wiki = pd.read_csv('people_wiki.csv')
    wiki = wiki.sample(frac=0.1, random_state=0)

    vectorizer = TfidfVectorizer(max_df=0.95)
    tf_idf = vectorizer.fit_transform(wiki['text'])
    words = vectorizer.get_feature_names()

    tf_idf = csr_matrix(tf_idf)

    tf_idf = normalize(tf_idf)

    queries = tf_idf[100:102, :]
    
    # compute pairwise distances from every data point to each query vector
    dist = pairwise_distances(tf_idf, queries, metric='euclidean')

    distances = pairwise_distances(tf_idf, tf_idf[0:3, :].toarray(), metric='euclidean')

    dist = distances[430, 1]

    # test cell
    if distances.shape==(5907, 3) and np.allclose(dist, pairwise_distances(tf_idf[430,:], tf_idf[1.:])):
        print('Pass')
    else:
        print('Check code again')

    closest_cluster = np.array([np.argmin(distances[i,:]) for i in range(0, tf_idf.shape[0])]).reshape(1, 5907)

    # test cell
    reference = [list(row).index(min(row)) for row in distances]
    if np.allclose(closest_cluster, reference):
        print('Pass')
    else:
        print('Check code again')

    distances = pairwise_distances(tf_idf, tf_idf[0:3, :].toarray(), metric='euclidean')
    min_value = 2
    cluster_assignment = []
    for i in range(0, tf_idf.shape[0]):
        cluster_assignment.append(np.argmin(distances[i,:]))

    if len(cluster_assignment)==5907 and np.array_equal(np.bincount(cluster_assignment), np.array([515, 440, 4952])):
        print('Pass')
    else:
        print('Check code again')

    if np.allclose(assign_cluster(tf_idf[0:100:10], tf_idf[0:8:2]), np.array([0, 3, 3, 3, 3, 2, 2, 1, 1, 1])):
        print('Pass')
    else:
        print('Check code again')

    data = np.array([[1., 2., 0.],
                    [0., 0., 0.],
                    [2., 2., 0.]])

    centroids = np.array([[0.5, 0.5, 0.],
                            [0., -0.5, 0.]])

    cluster_assignment = assign_cluster(data, centroids)

    result = revise_centroids(tf_idf[0:100:10], 3, np.array([0, 1, 1, 0, 0, 2, 0, 2, 2, 1]))

    k = 3
    heterogeneity = []
    initial_centroids = get_initial_centroids(tf_idf, k, seed=0)
    centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400, record_heterogeneity=heterogeneity, verbose=True)

    plot_heterogeneity(heterogeneity, k)

    np.bincount(cluster_assignment)

    k = 10
    heterogeneity = {}
    for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
        initial_centroids = get_initial_centroids(tf_idf, k, seed)
        centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400, record_heterogeneity=None, verbose=False)

        heterogeneity[seed] = compute_heterogeneity(tf_idf, k, centroids, cluster_assignment)

        sys.stdout.flush()

    plt.figure(figsize=(10,5))
    plt.boxplot([list(heterogeneity.values()), list(heterogeneity_smart.values())], vert=False)
    plt.yticks([1, 2], ['k-means', 'k-means++'])
    plt.rcPrarms.update({'font.size':16})
    plt.xlabel('Heterogeneity')
    plt.tight_layout()


    all_centroids = {}
    all_cluster_assignment = {}
    heterogeneity_values = []
    seeds = [20000, 40000, 80000]
    k_list = [2, 10, 25, 50, 100]

    for k in k_list:
        print(f'Running k = {k}')
        heterogeneity = []
        all_centroids[k], all_cluster_assignment[k] = kmeans_multiple_runs(tf_idf, k, maxiter=400, seeds=seeds, verbose=True)
        score = compute_heterogeneity(tf_idf, k, all_centroids[k], all_cluster_assignment[k])
        heterogeneity_values.append(score)

    plot_k_vs_heterogeneity(k_list, heterogeneity_values)

    k = 2
    visualize_document_clusters(wiki, tf_idf, all_centroids[k], all_cluster_assignment[k], k, words)

    k = 10
    visualize_document_clusters(wiki, tf_idf, all_centroids[k], all_cluster_assignment[k], k, words)

    np.bincount(all_cluster_assignment[10])

    k = 25
    visualize_document_clusters(wiki, tf_idf, all_centroids[k], all_cluster_assignment[k], k, words, display_docs=0)



def visualize_document_clusters(wiki, tf_idf, cnetroids, cluster_assignment, k, words, display_docs=5):
    """
    given a set of clustered documents, prints information about the ceentroids including
    - the title and starting sentence of the closest 5 points to each centroid
    - the five words that are contained in the clusters documents with the highest TF_IDF

    parameters:
    - wiki: original dataframe
    - tf_idf: data matrix containing TF-IDF vectors for each document
    - centroids: a np.array of length k that contains the centroids for the clustering
    - cluster_assignments: a np.array of length N that has the cluster assignmnets for each row
    - k: what vlaue of k is used
    - words: list of words in the corpus (should match tf_idf)
    - display_odcs: how many documents to show for each cluster (default 5)
    """

    print('=' * 90)
    # visualize each cluster c
    for c in range(k):
        # cluster heading
        print(f'Cluster {c} ({(cluster_assignment==c)/sum()} docs)')

        idx = centroids[c].argsort()[::-1]

        for i in range(5):
            print(f'{words[idx[i']]}: {centroids[c, idx[i]]:.3f}', end=' ')


        if display_docs > 0:
            # compute distances from the centroid to al data points in the cluster, and compute nearest neighbors of the centroids within the cluster
            distances = pairwise_distances(tf_idf, centroids[c].reshape(1,-1), metric='euclidean').flatten()
            distances[cluster_assignment!=c] = float('inf')
            nearest_neighbors = distances.argsort()

            # for the nearest neighbors, print the title as well as first 180 characters of text.
            for i in range(display_docs):
                text = ' '.join(wiki.iloc[nearest_neightbors[i]]['text'].split(None, 25)[0:25])
                print(f'* {wiki.iloc[nearest_neighbors[i]]['name']:50s} {distances[nearest_neightbors[i]]:.5f}')
                print(f' {text[:90]}')
                if len(text) > 90:
                    print(f'    {text[90:180]}')
        print('=' * 90)




def kmeans_multiple_runs(data, k, maxiter, seeds, verbose=False):
    """
    runs kmeans multiple times

    parameters:
        - data: is an np.array of float values of length N
        - k: number of centroids
        - maxiter: maximum number of iterations to run the algorithm
        - seeds: either number of seeds to try (generated randomly) or a list of seed values
        - verbose: set to True to display progress. Defaults to False and won't display progress.

    returns:
        - final_centroids: a np.array of length k for the centroids upon termination of the algorithm.
        - final_cluster_assignment: a np.array of length N where the i-th index represents which centroid data[i] was assigned to.
                                    the assignemtns range between the values 0, ..., k-1 upon termination of the algorithm.
    """

    min_heterogeneity_achieved = float('inf')
    final_centroids = None
    final_cluster_assignment = None
    if type(seeds)==int:
        seeds = np.random.randint(low=0, high=10000, size=seeds)

    num_runs = len(seeds)

    for seed in seeds:
        # use k-means++ initialization with the provided seed
        # set reocrd_heterogeneity=None because we will compute that once at the end

        initial_centroids = smart_initialize(data, k, seed)

        # run k-means
        centroids, cluster_assignment = kmeans(data, k, initial_centroids, maxiter=400, record_heterogeneity=None, verbose=False)

        # to save time, compute heterogeneity only once in the end
        seed_heterogeneity = compute_heterogeneity(data, k, centroids, cluster_assignment)

        if verbose:
            print(f'seed={seed:06d}, heterogeneity={seed_heterogeneity:.5f}')

        # if current measurement of heterogeneity is lower than previously seen, update the minimum record of heterogeneity
        if seed_heterogeneity < min_heterogeneity_achieved:
            min_heterogeneity_achieved = seed_heterogeneity
            final_centroids = centroids
            final_cluster_assignment = cluster_assignment

    return final_centroids, final_cluster_assignment


def plot_k_vs_heterogeneity(k_values, heterogeneity_values):
    """
    given list of k-values and their heterogeneities, will make a plot showing how heterogeneity varies with k
    """

    plt.fiture(figsize=(7,4))
    plt.plot(k_values, heterogeneity_values, linewidth=4)
    plt.xlabel('K')
    plt.ylabel('Heterogeneity')
    plt.title('K vs. Heterogeneity')
    plt.rcParams.upate({'font.size': 16})
    plt.tight_layout()

def smart_initialize(data, k, seed=None):
    """
    use k-means to initialize a good set of centroids
    """

    if seed is not None:
        np.random.seed(seed)

    centroids[0] = np.zeros((k, data.shape[1]))

    # randomly choose the first centroid.
    # since we have no prior knowledge, choose uniformly at random
    idx = np.random.randint(data.shape[0])
    centroids[0] = data[idx,:].toarray()

    # compute distances from the first centroid chosen to all the other data points
    distances = pairwise_distances(data, centroids[0:1], metric='euclidean').flatten()

    for i in range(1, k):
        # choose the next centroid randomly, so that the probability for each data point to be chosen is directly propertional to its squared distance from the nearest centroid.
        # roughtly speaking, a new centroid should be as far as from other centroids as possible.

        idx = np.random.choice(data.shape[0], 1, p=distances/sum(distances))
        centroids[i] = data[idx,:].toarray()

        # now compute distances form the centroids to al data points
        distances = np.min(pairwise_distances(data, centroids[0:i+1], metric='euclidean'), axis=1)

    return centroids


def compute_heterogeneity(data, k, centroids, cluster_assignment):
    """
    computes the heterogeneity metric of the data using the given centroids and cluster assignments.
    """

    heterogeneity = 0.0
    for i in range(k):

        # select all data points that belong to cluster i. 
        member_data_points = data[cluster_assignment==i,:]

        if member_data_points.shape[0] > 0:
            # check if i-th cluster is non-empty
            # compute distances from centroid to data point
            distances = pairwise_distances(member_data_points, [centroids[i]], metric='euclidean')
            squared_distances = distances**2
            heterogeneity += np.sum(squared_distances)

    return heterogeneity


def kmeans(data, k, initial_centroids, maxiter, record_heterogeneity=None, verbose=False):
    """
    this function runs k-means on given data and initial set of centroids.

    Parameters:
        - data: is an np.array of float values of length N.
        - k: number of centroids
        - initial_centroids: is an np.array of float values of length k.
        - maxiter: maximum number of iterations to run the algorithm
        - record_heterogeneity: if provided an empty list, it will compute the heterogeneity at each iteration and append it to the list.
                                defaults to None and won't record heterogeneity.
        - verbose: set to True to display progress. Defaults to Flase and won't display progress.

    Returns:
        - entroids: A np.array of liength k for the centroids upon termination of the algorithm.
        - cluster_assignment: A np.array of length N where the i-th index represents which centroid data[i] was assigned to. 
                              The assignments range between the values 0, ..., k-1 upon termination of the algorithm.
    """

    centroids = initial_centroids[:]
    prev_cluster_assignment = None

    for itr in range(maxiter):
        if verbose:
            print(itr)
        
        # 1. make cluster assignments using nearest centroids
        cluster_assignment = assign_cluster(data, centroids)

        # 2. compute a new centroid for each of the k clusters, averaging all data points assigned to that cluster.
        centroids = revise_centroids(data, k, cluster_assignment)

        # check for convergence: if none of the assignments changed, stop
        if prev_cluster_assignment is not None and (prev_cluster_assignment==cluster_assignment).all():
            break

        # print number of new assignments
        if prev_cluster_assignment is not None:
            num_changed = sum(abs(prev_cluster_assignment-cluster_assignment))
            if verbose:
                print(f'    {num_changed:5d} elements changed their cluster assignment.')

        # record heterogeneity convergence metric
        if record_heterogeneity is not None:
            score = compute_heterogeneity(data, k, centroids, cluster_assignment)
            record_heterogeneity.append(score)

        prev_cluster_assignment = cluster_assignment[:]

    return centroids, cluster_assignment


def plot_heterogeneity(heterogeneity, k):
    """
    plots how the heterogeneity changes as the number of iterations increases.
    """

    plt.figure(figsize=(7,4))
    plt.plot(heterogeneity, linewidth=4)
    plt.xlabel('# Iterations')
    plt.ylabel('Heterogeneity')
    plt.title(f'Heterogeneity of clustering over time, K={k}')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()


def revise_centroids(data, k, cluster_assignment):
    """
    Parameter:
        - data: is an np.array of float values of length N.
        - k: number of centroids
        - cluster_assignment: np.array of length N where the ith index represents which centroid data[i] was assigned to. The assignments range between the values 0, ..., k-1.

    Returns
        - A np.array of length k for the new centroids.
    """

    new_centroids = []
    for i in range(k):
        # select all data points that belong to cluster i.
        member_data_points = data[cluster_assignment==i]

        # compute the mean of the data points.
        centroid = member_data_points.mean(axis=0)

        # convert numpy.matrix type to numpy.ndarray type
        centroid = centroid.getA1()
        new_centroids.append(centroid)

    return np.array(new_centroids)



def  assign_cluster(data, centroids):

    """
    Parameters:
        - data: is an np.array of float values of length N.
        - centroids: is an np.array of float values of length k.

    Returns
        - A np.array of length N where the ith index represents which centroid data[i] was assigned to. The assignments range between the value 0, ..., k-1.
    """

    distances = pairwise_distances(data, centroids, metric='euclidean')
    cluster_assignment = []
    for i in range(0, data.shape[0]):
        cluster_assignment.append(np.argmin(distances[i,:]))

    return np.array(cluster_assignment)



def get_initial_centroids(data, k, seed=None):

    # randomly choose k data points as initial centroids

    if seed is not None:
        np.random.seed(seed)

    n = data.shape[0]

    # pick K indices from range [0, N]
    rand_indices = np.random.randint(0, n, size=k)

    centroids = data[rand_indices, :].toarray()

    return centroids



def save_file(url, file_name):
    r = requests.get(url)
    with open(file_name, 'wb') as f:
        f.write(r.content)


if __name__=='__main__':
    k_means_with_text_data()

