import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error

class NMFRecommender:

    def __init__(self, random_state=15, rank=2, maxiter=200, tol=1e-3):
        """
        Save the parameter values as attributes.
        """
        # Initialize parameters
        self.random_state = random_state
        self.rank = rank
        self.maxiter = maxiter
        self.tol = tol


    def _initialize_matrices(self, m, n):
        """
        Initialize the W and H matrices.
        
        Parameters:
            m (int): the number of rows
            n (int): the number of columns
        Returns:
            W ((m,k) array)
            H ((k,n) array)
        """
        np.random.seed(self.random_state)  # Set random seet
        # Initialize matrices to values between 0 and 1
        W = np.random.rand(m, self.rank)
        H = np.random.rand(self.rank, n)
        return W, H


    def _compute_loss(self, V, W, H):
        """
        Compute the loss of the algorithm according to the 
        Frobenius norm.
        
        Parameters:
            V ((m,n) array): the array to decompose
            W ((m,k) array)
            H ((k,n) array)
        Returns:
            Frobenius norm of V - WH (float)
        """
        # Calculate the frobenius norm
        frobenius_norm = np.linalg.norm(V - W @ H, ord='fro')
        return frobenius_norm


    def _update_matrices(self, V, W, H):
        """
        The multiplicative update step to update W and H
        Return the new W and H (in that order).
        
        Parameters:
            V ((m,n) array): the array to decompose
            W ((m,k) array)
            H ((k,n) array)
        Returns:
            New W ((m,k) array)
            New H ((k,n) array)
        """
        new_H = H * ((W.T @ V)/(W.T @ W @ H))   # 25.1
        new_W = W * ((V @ new_H.T)/(W @ new_H @ new_H.T))  # 25.2
        return new_W, new_H


    def fit(self, V):
        """
        Fits W and H weight matrices according to the multiplicative 
        update algorithm. Save W and H as attributes and return them.
        
        Parameters:
            V ((m,n) array): the array to decompose
        Returns:
            W ((m,k) array)
            H ((k,n) array)
        """
        # Initialize things
        m, n = np.shape(V)
        W, H = self._initialize_matrices(m, n)
        
        # Loop until done
        for i in range(self.maxiter):
            W, H = self._update_matrices(V, W, H)  # 25.1 and 25.2
            loss = self._compute_loss(V, W, H)  # Stop is loss is small enough
            if loss < self.tol: return W, H
            
        # Save results as attributes
        self.W = W
        self.H = H
        
        return W, H


    def reconstruct(self):
        """
        Reconstruct and return the decomposed V matrix for comparison against 
        the original V matrix. Use the W and H saved as attrubutes.
        
        Returns:
            V ((m,n) array): the reconstruced version of the original data
        """
        return self.W @ self.H


def prob4(rank=2):
    """
    Run NMF recommender on the grocery store example.
    
    Returns:
        W ((m,k) array)
        H ((k,n) array)
        The number of people with higher component 2 than component 1 scores (float)
    """
    V = np.array([[0, 1, 0, 1, 2, 2],
                [2, 3, 1, 1, 2, 2],
                [1, 1, 1, 0, 1, 1],
                [0, 2, 3, 4, 1, 1],
                [0, 0, 0, 0, 1, 0]])
    
    # Get the recommender and fit it
    clf = NMFRecommender(rank=2)
    W, H = clf.fit(V)
    
    # Determine the number of people with higher component 2 than component 1 scores (float)
    num_people = np.sum(H[1] > H[0])
    
    return W, H, num_people


def prob5(filename='artist_user.csv'):
    """
    Read in the file `artist_user.csv` as a Pandas dataframe. Find the optimal
    value to use as the rank as described in the lab pdf. Return the rank and the reconstructed matrix V.
    
    Returns:
        rank (int): the optimal rank
        V ((m,n) array): the reconstructed version of the data
    """
    # Read in the data
    X = pd.read_csv(filename, index_col=0)
    
    # Get benchmark value
    # frob_norm = np.linalg.norm(X, ord='fro')   # TODO: FIX
    # bench_mark = frob_norm * 0.0001   # TODO: FIX
    
    rank = 10
    # for rank in [10, 11, 12, 13, 14, 15]:  # TODO: FIX
    for rank in [13]:
        # Train the model
        model = NMF(n_components=rank, init="random", random_state=0)
        W = model.fit_transform(X)
        H = model.components_
        V = W @ H  # Reconstruct V
        
        # RMSE = np.sqrt(mean_squared_error(X, V))  # Calculate the RMSE  # TODO: FIX
        # if RMSE < bench_mark: break   # Stop if close enough  # TODO: FIX
    
    return rank, V


def discover_weekly(userid, V):
    """
    Create the recommended weekly 30 list for a given user.
    
    Parameters:
        userid (int): which user to do the process for
        V ((m,n) array): the reconstructed array
        
    Returns:
        recom (list[str]): a list of strings that contains the names of the recommended artists
    """
    # Read in the data
    artist_user = pd.read_csv("artist_user.csv", index_col=0)
    artists = pd.read_csv("artists.csv", index_col=0)
    row = list(artist_user.index)
    user_index = row.index(userid)
    
    # Determine the artists already listed
    already_listed = set(artist_user.loc[userid][artist_user.loc[userid] != 0].index)
    
    # Get output for the corresponding user and sort it from largest to smallest
    out = V[user_index]   
    sorted_indices = np.argsort(out) 
    
    all_recommendations = list(artist_user.columns[sorted_indices])
    all_recommendations.reverse()
        
    count = 0
    recom = []
    
    for i in all_recommendations:  # Iterate through values in row from largest to smallest
        if i not in already_listed:
            recom.append(artists.loc[int(i)][0])
            count += 1
        if count >= 30: break
        
    return recom
    
    
if __name__=="__main__":
    
    # clf = NMFRecommender()
    
    # Prob 1
    # m, n = 4, 3
    # W, H = clf._initialize_matrices(m, n)
    # V = np.random.rand(m, n)
    # print(W, H)
    # frob_loss = clf._compute_loss(V, W, H)
    # print(frob_loss)
    
    # Prob 2
    # m, n = 4, 3
    # W, H = clf._initialize_matrices(m, n)
    # V = np.random.rand(m, n)
    # new_W, new_H = clf._update_matrices(V, W, H)
    # print(new_W, new_H)
    
    # Prob 3
    # m, n = 4, 3
    # V = np.array([[7, 12, 16], [15, 26, 36], [23, 40, 56], [35, 61, 86]])
    # W, H = clf.fit(V)
    # print("W", W)
    # print("H", H)
    # print("V", V)
    # print("WH", clf.reconstruct())
    # print(clf._compute_loss(V, W, H))
    
    # Prob 4
    # W, H, num_people = prob4()
    # print("W:", W)
    # print("H:", H)
    # print("num_people:", num_people)
    
    # Prob 5 and 6
    # rank, V = prob5()

    # # print("rank", rank)
    # # print("V", V)
    
    # recom = discover_weekly(2, V)
    # print(recom)
    
    pass
