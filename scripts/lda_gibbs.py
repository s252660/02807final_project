"""Latent Dirichlet Allocation (LDA) fitted with Gibbs sampling.

Follows Griffiths Steyvers' paper: https://web-static-aws.seas.harvard.edu/courses/cs281/papers/griffiths-steyvers-2004.pdf

Notation (as in the paper)
--------------------------
    T              number of topics
    W              number of distinct words (vocabulary size)
    D              number of documents (songs)
    w_i            the i-th word token in the corpus (one entry per occurrence)
    d_i            the document that token i belongs to
    z_i            the topic assigned to token i
    alpha, beta    Dirichlet hyperparameters (paper defaults: alpha = 50/T, beta = 0.1)

    n_{-i,j}^{(w_i)}   number of times word w_i is assigned to topic j       -> n_wj[w_i, j]
    n_{-i,j}^{(.)}     total number of words assigned to topic j             -> n_j[j]
    n_{-i,j}^{(d_i)}   number of words in document d_i assigned to topic j   -> n_dj[d_i, j]
    n_{-i,.}^{(d_i)}   total number of words in document d_i                 -> n_d[d_i]
    The "-i" means: not counting the token i that is currently being re-sampled.

Equation 2 (used to monitor convergence)
----------------------------------------
    P(w | z) = ( Gamma(W*beta) / Gamma(beta)^W )^T
               * prod_j [ prod_w Gamma(n_j^{(w)} + beta) ] / Gamma(n_j^{(.)} + W*beta)

Equation 5 (the Gibbs sampling update)
--------------------------------------
                                   n_{-i,j}^{(w_i)} + beta       n_{-i,j}^{(d_i)} + alpha
    P(z_i = j | z_{-i}, w)  ∝  ---------------------------  *  ---------------------------
                                 n_{-i,j}^{(.)}  + W * beta      n_{-i,.}^{(d_i)} + T * alpha

    first factor : probability of word w_i under topic j
    second factor: probability of topic j in document d_i

Estimates of phi and theta from a sample of z (the paper, right after equation 5)
------------------------------------------------------------------------------
    phi_j^{(w)}   = (n_j^{(w)} + beta)  / (n_j^{(.)} + W * beta)      # topic j's word distribution
    theta_j^{(d)} = (n_j^{(d)} + alpha) / (n_.^{(d)} + T * alpha)     # document d's topic mix
"""

import numpy as np
from numba import njit
from scipy.special import gammaln

class LDA:
    """LDA with Gibbs sampling

    Fitted attributes:
        topic_word : array (T, W)   phi, row j is topic j's distribution over words
        doc_topic  : array (D, T)   theta, row d is document d's distribution over topics
        log_likelihood : list       log P(w | z) (equation 2) after each sweep
    """

    def __init__(self, n_topics=10, alpha=None, beta=0.1, n_iter=200, random_state=None):
        self.n_topics = n_topics                                            # Number of topics T
        
        self.alpha = 50.0 / n_topics if alpha is None else float(alpha)     # Default value as in the paper
        self.beta = beta
        
        self.n_iter = n_iter
        self.random_state = random_state

    @staticmethod
    def log_p_w_given_z(n_wj, n_j, beta):
        """ Equation 2 from the Gibbs paper, to monitor convergence, logged (so we sum instead of multiplying)
        """
        W, T = n_wj.shape
        return float(T * (gammaln(W * beta) - W * gammaln(beta))
                    + gammaln(n_wj + beta).sum()
                    - gammaln(n_j + W * beta).sum())
    
    def fit(self, X):
        """
            To learn the topics from a doc word count matrix X.
            Steps:
                1. Initialise the topic assignments z_i at random and build the count tables.
                2. Gibbs sweeps: for each token i, sample a new topic z_i using equation 5.
                3. Estimate phi (word probabilities for each topic) and theta (song topic mixtures) from the final sample of z.
        """

        # 1. Init
        rng = np.random.default_rng(self.random_state)

        T = self.n_topics                                               # Number of topics you ask for
        D, W = X.shape                                                  # Number of songs, vocabulary size

        alpha = self.alpha
        beta = self.beta                                               # Default value as in the paper

        # 1.1 Convert the count matrix into a pair of lists of word tokens, document id 
        # w[i] = which word token i is, d[i] = which song token i is in. See to_tokens() for details.
        w, d = self.to_tokens(X)

        # 1.2 Initialise topic assigment (z_i) at random
        z = rng.integers(T, size=len(w)) # Recall T is number of topics

        # 1.3 Build word & document to topic count tables.
        n_wj = np.zeros((W, T), dtype=np.int64)   # word-topic counts
        n_dj = np.zeros((D, T), dtype=np.int64)   # document-topic counts

        np.add.at(n_wj, (w, z), 1)                # Tallies of how many times each word is assigned to each topic
        np.add.at(n_dj, (d, z), 1)
        n_j = n_wj.sum(axis=0)                    # number of words to each topic
        n_d = n_dj.sum(axis=1)                    # number of tokens in each song

        # 2. Gibbs sweeps
        self.log_likelihood = []
        for seed in rng.integers(2**31, size=self.n_iter):
            gibbs_sweep(w, d, z, n_wj, n_j, n_dj, n_d, alpha, beta, seed)       # Performs sweep
            self.log_likelihood.append(self.log_p_w_given_z(n_wj, n_j, beta))   # Compute log P(w|z) after this sweep. To monitor convergence, see if it stabilises after a few sweeps.

        # 3. Estimates of phi (word probabilities for each topic) and theta (song topic mixture)
        phi = (n_wj + beta) / (n_j + W * beta)   
        self.topic_word = phi.T     

        theta = (n_dj + alpha) / (n_d[:, None] + T * alpha)                                          
        self.doc_topic = theta                                    

        return self

    def save(self, path, vocab, songs):
        """Save the settings and fitted values to an .npz file
            path : where to save the file
            vocab : word of each column of X        (needed to name the topics' words)
            songs : song index of each row of X     (which song each row of doc_topic belongs to)
        """
        np.savez_compressed(path,
            n_topics=self.n_topics, 
            alpha=self.alpha, 
            beta=self.beta,
            n_iter=self.n_iter, 
            random_state=self.random_state,

            topic_word=self.topic_word, 
            doc_topic=self.doc_topic,
            log_likelihood=np.array(self.log_likelihood),
            vocab=np.asarray(vocab, dtype=str), 
            songs=np.asarray(songs)
        )

    @classmethod
    def load(cls, path):
        """Rebuild a fitted model saved with save()."""
        f = np.load(path)

        model = cls(n_topics=int(f["n_topics"]), alpha=float(f["alpha"]), beta=float(f["beta"]), n_iter=int(f["n_iter"]), random_state=f["random_state"].item())
        model.topic_word = f["topic_word"]
        model.doc_topic = f["doc_topic"]
        model.log_likelihood = list(f["log_likelihood"])

        model.vocab = f["vocab"]
        model.songs = f["songs"]
        return model

    def top_words(self, vocab, n=10):
        """Top n words of each topic. Useful for inspecting the topics after fitting."""
        results = []
        for row in self.topic_word:
            # 1. Get top word indices for this topic, sorted by probability
            sorted_indices = np.argsort(row)[::-1]  

            # 2. Fetch the specified words from 1. in vocab list
            words = vocab[sorted_indices[:n]]       
            results.append(list(words))

        return results
    
    @staticmethod
    def to_tokens(X):
        """Turn a count matrix into one entry per word token: w[i] (word) and d[i] (document).
            Input: doc word count matrix
                e.g:
                    Song 1:
                        "quick quick slow"
                    Song 2:
                        "slow rock step"

                    Vocab: ["quick", "slow", "rock", "step"] # 0, 1, 2, 3 in W
                    X = [[2, 1, 0, 0],
                        [0, 1, 1, 1]]

            Output: w, d arrays of length = number of tokens
                e.g: 
                    w = [0, 0, 1, 1, 2, 3]
                    d = [0, 0, 0, 1, 1, 1] # 0: song 1, 1: song 2
        """
        # 1. Find every cell that is not zero: song rows[k] contains word cols[k].
        rows, cols = X.nonzero()

        # 2. Look up how many times that word appears in that song.
        # rint: round to nearest int 
        counts = np.rint(np.asarray(X[rows, cols]).ravel()).astype(np.int64) # Flatten to 1D array

        # 3. Write each (song, word) pair down `count` times: one entry per token.
        d = np.repeat(rows, counts).astype(np.int64)
        w = np.repeat(cols, counts).astype(np.int64)

        return w, d

@njit
def gibbs_sweep(w, d, z, n_wj, n_j, n_dj, n_d, alpha, beta, seed):
    """
        Does 1 full pass on every word in every song.

    """
    # 1. Init
    np.random.seed(seed)
    
    W, T = n_wj.shape
    p = np.empty(T) # score for each topic, unnormalised probability of topic j for token i

    # 2. For each word token i, sample a new topic for it
    for i in range(len(w)):
        wi, di, j_old = w[i], d[i], z[i] # word of token i: vocab id; song of token i; topic of token before update

        # 2.1 Remove the current token i from the counts before calculating the probabilities for each topic.
        n_wj[wi, j_old] -= 1        # Remove the current word from the original topic counts
        n_j[j_old] -= 1             # Remove from old topic's total count
        n_dj[di, j_old] -= 1        # Remove this song's count for the old topic
        n_d_minus_i = n_d[di] - 1   # Remove from length of current song

        # 2.2 Plug in equation 5 from paper
        for j in range(T):
            p[j] = ((n_wj[wi, j] + beta) / (n_j[j] + W * beta)
                    * (n_dj[di, j] + alpha) / (n_d_minus_i + T * alpha))

        # 2.3 Sample new topic j_new from p =
        j_new = sample_discrete(p)

        # 2.4 Add token i back with its new topic.
        z[i] = j_new
        n_wj[wi, j_new] += 1
        n_j[j_new] += 1
        n_dj[di, j_new] += 1

@njit
def sample_discrete(p):
    """
        Draw topic based on probabilities defined in p.
        Each topic j has probability: p[j] / sum(p).
    """
    # 1. Get the random number
    u = np.random.random() * p.sum()

    # 2. See where the random number lands in the cum dist. of p
    cumulative = 0.0
    for j in range(len(p)):
        cumulative += p[j]
        if u < cumulative:
            return j

    # 3. Return chosen topic j
    return len(p) - 1
