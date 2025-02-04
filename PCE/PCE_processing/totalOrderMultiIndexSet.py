import numpy as np 
from numpy.matlib import repmat

def totalOrderMultiIndices(m, p):
    """
    Returns a matrix containing all multi-indices of size m of total order p
    (implementation due to prior course member)

    Parameters
    ----------
    m : int 
        Number of index components.
    p : int
        Maximum total degree (sum of degrees across a multiindex)

    Returns
    -------
    np.array
        Matrix containing multi-indices on rows.

    """
    Mk = np.zeros((1, m))
    M = Mk
    for k in range(p):
        Mk = repmat(Mk, m, 1) + np.kron(np.eye(m), np.ones((Mk.shape[0], 1)))
        Mk = np.unique(Mk, axis=0)
        M = np.vstack((M, Mk))
    return M.astype(int)