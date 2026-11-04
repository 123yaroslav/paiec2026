"""Hierarchical logit adaptation shared by the simulator and the submission.

For one benchmark the logit of subject s on item i is

    gamma * z_i + c + d_s

z_i is the base prior logit, c is an offset shared by all subjects on this
benchmark, d_s is the offset of subject s. Priors are c ~ N(0, tau_c^2) and
d_s ~ N(0, tau_d^2). Labels of every subject on the same benchmark update c,
labels of subject s also update d_s. The posterior is a Laplace approximation,
and the prediction integrates its variance with the probit approximation.
"""

import math

import numpy as np


def posterior(z, subj, y, target, tau_c, tau_d, gamma, iters=25):
    """Return mean and variance of c + d_target.

    z      base logits of labeled items (1-d array)
    subj   integer subject index of each label, target subject included or not
    y      0/1 labels
    target integer index of the target subject
    """
    subjects = sorted(set(int(s) for s in subj) | {int(target)})
    pos = {s: k + 1 for k, s in enumerate(subjects)}
    dim = len(subjects) + 1
    prec = np.full(dim, 1.0 / tau_d ** 2)
    prec[0] = 1.0 / tau_c ** 2
    theta = np.zeros(dim)
    if len(y):
        cols = np.array([pos[int(s)] for s in subj])
        z = np.asarray(z, dtype=float)
        y = np.asarray(y, dtype=float)
        for _ in range(iters):
            eta = gamma * z + theta[0] + theta[cols]
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
            r = y - p
            w = p * (1 - p)
            grad = -prec * theta
            grad[0] += r.sum()
            np.add.at(grad, cols, r)
            hess = np.diag(prec)
            hess[0, 0] += w.sum()
            wc = np.zeros(dim)
            np.add.at(wc, cols, w)
            hess[0, 1:] += wc[1:]
            hess[1:, 0] += wc[1:]
            hess[np.arange(1, dim), np.arange(1, dim)] += wc[1:]
            step = np.linalg.solve(hess, grad)
            theta += step
            if np.abs(step).max() < 1e-7:
                break
        eta = gamma * z + theta[0] + theta[cols]
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        w = p * (1 - p)
        hess = np.diag(prec)
        hess[0, 0] += w.sum()
        wc = np.zeros(dim)
        np.add.at(wc, cols, w)
        hess[0, 1:] += wc[1:]
        hess[1:, 0] += wc[1:]
        hess[np.arange(1, dim), np.arange(1, dim)] += wc[1:]
    else:
        hess = np.diag(prec)
    cov = np.linalg.inv(hess)
    t = pos[int(target)]
    mean = theta[0] + theta[t]
    var = cov[0, 0] + cov[t, t] + 2 * cov[0, t]
    return float(mean), float(max(var, 0.0))


def predict(z_target, mean, var, gamma):
    eta = (gamma * np.asarray(z_target, dtype=float) + mean) / math.sqrt(1 + math.pi * var / 8)
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))


def item_shift(x_target, x_labeled, residual, weight, bandwidth, sigma2):
    """Item-level logit shift from labels of similar items on the same benchmark.

    x_target   (n_t, d) normalized embeddings of target items
    x_labeled  (n_l, d) normalized embeddings of labeled items
    residual   y - p of each labeled item under the level model
    weight     p (1 - p) of each labeled item
    Similarity is exp((cos - 1) / bandwidth), so an identical item has weight 1.
    The shift is one Newton step for a shared offset with prior variance sigma2.
    """
    if len(residual) == 0 or sigma2 <= 0:
        return np.zeros(len(x_target))
    k = np.exp((np.asarray(x_target, dtype=np.float32) @ np.asarray(x_labeled, dtype=np.float32).T - 1.0) / bandwidth)
    return sigma2 * (k @ residual) / (1.0 + sigma2 * (k @ weight))
