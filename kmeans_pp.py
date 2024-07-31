import sys
import numpy as np
import pandas as pd
import mykmeanssp

def validate_args(args):
    if len(args) < 5:
        raise ValueError("Insufficient arguments provided!")
    
    try:
        K = int(args[1])
        if not (1 < K < int(args[2])):
            raise ValueError("Invalid number of clusters!")
    except ValueError:
        raise ValueError("Invalid number of clusters!")
    
    try:
        iter = int(args[2]) if len(args) > 5 else 300
        if not (1 < iter < 1000):
            raise ValueError("Invalid maximum iteration!")
    except ValueError:
        raise ValueError("Invalid maximum iteration!")
    
    try:
        eps = float(args[3])
        if eps < 0:
            raise ValueError("Invalid epsilon!")
    except ValueError:
        raise ValueError("Invalid epsilon!")
    
    file_name_1 = args[4]
    file_name_2 = args[5]
    
    if not (file_name_1.endswith('.txt')):
        raise ValueError("Invalid file format for file_name_1!")
    
    if not (file_name_2.endswith('.txt')):
        raise ValueError("Invalid file format for file_name_2!")
    
    return K, iter, eps, file_name_1, file_name_2

def load_data(file_name):
    if file_name.endswith('.txt'):
        return pd.read_csv(file_name, delimiter='\t')
    else:
        raise ValueError("Unsupported file format!")

def kmeans_pp_initialization(data, K):
    np.random.seed(1234)
    N, d = data.shape
    centroids = np.zeros((K, d))
    indices = np.zeros(K, dtype=int)
    
    indices[0] = np.random.choice(N)
    centroids[0] = data[indices[0]]
    
    for k in range(1, K):
        D2 = np.array([min([np.inner(c-x, c-x) for c in centroids[:k]]) for x in data])
        probs = D2 / D2.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()
        
        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break
        
        indices[k] = i
        centroids[k] = data[i]
    
    return indices, centroids

def main():
    try:
        K, iter, eps, file_name_1, file_name_2 = validate_args(sys.argv)
        
        data_1 = load_data(file_name_1)
        data_2 = load_data(file_name_2)
        
        merged_data = pd.merge(data_1, data_2, on=data_1.columns[0])
        sorted_data = merged_data.sort_values(by=merged_data.columns[0])
        observations = sorted_data.iloc[:, 1:].values
        
        initial_indices, initial_centroids = kmeans_pp_initialization(observations, K)
        
        final_centroids = mykmeanssp.fit(initial_centroids, observations, K, iter, eps)
        
        print(','.join(map(str, initial_indices)))
        for centroid in final_centroids:
            print(','.join(map(lambda x: f'{x:.4f}', centroid)))
    
    except ValueError as e:
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
