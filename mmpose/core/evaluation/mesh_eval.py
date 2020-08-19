import numpy as np


def compute_similarity_transform(S1, S2):
    """Computes a similarity transform (sR, t) that takes a set of 3D points S1
    (N x 3) closest to a set of 3D points S2, where R is an 3x3 rotation
    matrix, t 3x1 translation, s scale. And return the transformed 3D points
    S1_hat (N x 3). i.e. solves the orthogonal Procrutes problem.

    Notes:
        Points number: N

    Args:
        S1 (np.ndarray([N, 3])): Source point set.
        S2 (np.ndarray([N, 3])): Target point set.

    Returns:
        S1_hat (np.ndarray([N, 3])): Transformed source point set.
    """

    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Transform the source points:
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat
