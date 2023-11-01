import numpy as np


def fill(mat, ix, jx, Iq, Jq, dataq, counter, sym=False):
    ii, jj = np.meshgrid(ix, jx, indexing="ij")
    assert mat.size == ii.size
    dataq[counter : counter + ii.size] = mat.flatten()
    Iq[counter : counter + ii.size] = ii.flatten()
    Jq[counter : counter + ii.size] = jj.flatten()
    counter += ii.size
    if sym:
        counter = fill(mat.T, jx, ix, Iq, Jq, dataq, counter, sym=False)
    return counter


def fill_eye(dim, i_start, j_start, Iq, Jq, dataq, counter):
    dataq[counter : counter + dim] = np.ones(dim)
    Iq[counter : counter + dim] = range(i_start, i_start + dim)
    Jq[counter : counter + dim] = range(j_start, j_start + dim)
    counter += dim
    return counter


def get_R_matrix(prob, dim, total_dim, regularization):
    import scipy.sparse as sp

    nnz = prob.N * dim**2 + 2 * (prob.N - 1) * dim**2
    I = np.empty(nnz)
    J = np.empty(nnz)
    data = np.empty(nnz)
    counter = 0

    for n in range(prob.N):
        i = n * dim
        R_nn = prob.get_R_nn(n, dim, regularization)
        counter = fill(
            R_nn,
            range(i, i + dim),
            range(i, i + dim),
            I,
            J,
            data,
            counter,
        )
        if n > 0:
            R_nm = prob.get_R_nm(n, dim, regularization)
            counter = fill(
                R_nm,
                range(i - dim, i),
                range(i, i + dim),
                I,
                J,
                data,
                counter,
                sym=True,
            )
    assert counter == nnz, ("R", counter, nnz)
    R = sp.csc_array((data, (I, J)), shape=(total_dim, total_dim))
    return R


def get_A_inv(prob, regularization, reduced=True):
    import scipy.sparse as sp

    dim = prob.get_dim(regularization)
    if reduced:
        total_shape = ((prob.N - 1) * dim, prob.N * dim)
        nnz = (prob.N - 1) * dim**2 + (prob.N - 1) * dim
    else:
        total_shape = (prob.N * dim, prob.N * dim)
        nnz = (prob.N - 1) * dim**2 + (prob.N) * dim

    I = np.empty(nnz)
    J = np.empty(nnz)
    data = np.empty(nnz)
    counter = 0
    start_i = 0

    if not reduced:
        counter = fill_eye(dim, 0, 0, I, J, data, counter)
        start_i += dim

    for n in range(1, prob.N):
        j = (n - 1) * dim
        counter = fill(
            -prob.get_phi(n, regularization),
            range(start_i + j, start_i + j + dim),
            range(j, j + dim),
            I,
            J,
            data,
            counter,
        )
        counter = fill_eye(dim, start_i + j, j + dim, I, J, data, counter)
    assert counter == nnz, (counter, nnz)
    A_inv = sp.csc_array((data, (I, J)), shape=total_shape)
    return A_inv


def get_prob_matrices(prob, regularization):
    import scipy.sparse as sp

    k = prob.get_dim(regularization)
    total_dim = prob.N * (k + 1) + 1
    A_0 = sp.csc_array(
        ([1.0], ([total_dim - 1], [total_dim - 1])), shape=(total_dim, total_dim)
    )

    qnnz = prob.N * (k + 1) ** 2 + 2 * prob.N * (k + 1) + 1
    Iq = np.empty(qnnz)
    Jq = np.empty(qnnz)
    dataq = np.empty(qnnz)
    counterq = 0

    A_list = []
    q_0 = 0.0
    for n in range(prob.N):
        i = n * (k + 1)

        I_n = list(range(i, i + prob.d)) + [i + k] + [total_dim - 1]
        J_n = list(range(i, i + prob.d)) + [total_dim - 1] + [i + k]
        data_n = [1.0] * prob.d + [-0.5] * 2
        A_n = sp.csc_array((data_n, (I_n, J_n)), shape=(total_dim, total_dim))
        A_list.append(A_n)

        Q_nn, q_n, q_0n = prob.get_Q_matrices(n, dim=k + 1)

        assert i + k + 1 <= total_dim, f"too high index: {i+k+1}"
        counterq = fill(
            Q_nn, range(i, i + k + 1), range(i, i + k + 1), Iq, Jq, dataq, counterq
        )
        counterq = fill(
            q_n,
            range(i, i + k + 1),
            range(total_dim - 1, total_dim),
            Iq,
            Jq,
            dataq,
            counterq,
            sym=True,
        )

        q_0 += q_0n

    dataq[-1] = q_0
    Iq[-1] = total_dim - 1
    Jq[-1] = total_dim - 1
    counterq += 1

    assert counterq == qnnz, ("Q", counterq, qnnz)
    Q = sp.csc_array((dataq, (Iq, Jq)), shape=(total_dim, total_dim))
    if regularization == "no":
        return Q, A_0, A_list, None

    R = get_R_matrix(
        prob, dim=k + 1, total_dim=total_dim, regularization=regularization
    )
    return Q, A_0, A_list, R


def get_H(Q_all, A_0, A_list, rho_est, lambdas_est):
    if type(A_0) == list:
        assert len(rho_est) == len(A_0)
        assert len(A_list) == len(A_0)
        H = Q_all
        for n in range(len(rho_est)):
            H += rho_est[n] * A_0[n] + lambdas_est[n] * A_list[n]
    else:
        H = Q_all + rho_est * A_0
        for n in range(len(lambdas_est)):
            H += lambdas_est[n] * A_list[n]
    return H


# TODO: replace with cert_matrix.get_original_f
def get_f(x, dim=None):
    print("get_f is deprecated, use cert_matrix.get_original_f instead.")
    if np.ndim(x) == 1:
        assert dim is not None
        X = x.reshape((-1, dim))  # N x d
    else:
        X = x
        if dim is not None:
            assert X.shape[1] == dim
        else:
            dim = X.shape[1]
    if dim in [4, 6]:
        d = dim // 2
    else:
        d = dim
    F = np.c_[X, np.linalg.norm(X[:, :d], axis=1) ** 2]
    f = np.r_[F.flatten(), 1.0]
    return f
