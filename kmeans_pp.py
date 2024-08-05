import sys
import numpy as np
import pandas as pd
import mykmeanssp


def validate_args(args):
    if len(args) < 5:
        raise ValueError("An Error Has Occurred")

    K = float(args[1])
    if len(args) == 5:
        iter = 300
        eps = float(args[2])
        file_name_1 = args[3]
        file_name_2 = args[4]

    if len(args) == 6:
        iter = float(args[2])
        eps = float(args[3])
        file_name_1 = args[4]
        file_name_2 = args[5]

    try:
        if K.is_integer():
            K = int(K)
        else:
            print("Invalid number of clusters!")
            sys.exit(1)
        if not (1 < K):
            raise ValueError("Invalid number of clusters!")
    except ValueError:
        raise ValueError("Invalid number of clusters!")

    try:
        if iter.is_integer():
            iter = int(iter)
        else:
            print("Invalid maximum iteration!")
            sys.exit(1)
        if not (1 < iter < 1000):
            raise ValueError("Invalid maximum iteration!")
    except ValueError:
        raise ValueError("Invalid maximum iteration!")

    try:
        if eps < 0:
            raise ValueError("Invalid epsilon!")
    except ValueError:
        raise ValueError("Invalid epsilon!")

    return K, iter, eps, file_name_1, file_name_2


def load_data(file_name):
    return pd.read_csv(file_name, header=None)


def kmeans_pp_initialization(data, K):
    np.random.seed(1234)
    N, d = data.shape
    centroids = np.zeros((K, d))
    indices = np.zeros(K, dtype=int)

    indices[0] = np.random.choice(np.arange(1, N + 1)) - 1
    centroids[0] = data[indices[0]]

    for k in range(1, K):
        D2 = np.array([np.min([np.linalg.norm(c - x) for c in centroids[:k]]) for x in data])
        # divide every value cell by the sum of all cells
        sum_D2 = np.sum(D2)
        probs = D2 / sum_D2
        # put probability zero in all the indexes which are already in the centroids
        for i in range(k):
            probs[indices[i]] = 0
        indices[k] = np.random.choice(N, p=probs)

        centroids[k] = data[indices[k]]

    return indices, centroids


def main():
    try:
        K, iter, eps, file_name_1, file_name_2 = validate_args(sys.argv)

        data_1 = load_data(file_name_1)
        data_2 = load_data(file_name_2)
        merged_data = pd.merge(data_1, data_2, on=data_1.columns[0])
        sorted_data = merged_data.sort_values(by=merged_data.columns[0])

        # update indexes and delete the first col
        sorted_data.reset_index(drop=True, inplace=True)

        observations = sorted_data.iloc[:, 1:].values

        # Validate that clusters < observations (K < N)
        N, d = observations.shape
        if not (K < N):
            raise ValueError("Invalid number of clusters!")

        initial_indices, initial_centroids = kmeans_pp_initialization(observations, K)

        # turn centroids and obsevrations to a list of lists
        initial_centroids = initial_centroids.tolist()
        observations = observations.tolist()

        final_centroids = mykmeanssp.fit(initial_centroids, observations, K, iter, eps)

        print(",".join(map(str, initial_indices)))
        for centroid in final_centroids:
            print(",".join(map(lambda x: f"{x:.4f}", centroid)))

    except ValueError as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
