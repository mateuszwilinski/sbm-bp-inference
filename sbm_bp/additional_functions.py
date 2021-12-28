# -*- coding: utf-8 -*-
import numpy as np

# functions


def noise_links(N, m, model="ER", eps=0.0, randomize_nodes=False):
    if model == "ER":
        ids_links = np.random.choice(int(N * (N - 1) / 2), m, replace=False)
        if N % 2:
            links = np.array([ids_links % (N), ids_links // (N)]).T
            links[:, 1] = (links[:, 0] + links[:, 1] + 1) % N
        else:
            links = np.array([(ids_links % (N - 1)) + 1, ids_links // (N - 1)]).T
            links[:, 1] = links[:, 1] + ((links[:, 0] - 1) // (N / 2)) * N / 2
            links[:, 0] = links[:, 0] - ((links[:, 0] - 1) // (N / 2)) * N / 2
            links[:, 0] = (links[:, 1] + links[:, 0]) % N
    elif model == "SBM":
        prob_between = eps / (1.0 + eps)
        m_inside = np.min([np.sum(np.random.rand(m) > prob_between), int(N * (N - 2) / 4)])
        m_between = m - m_inside
        ids_between = np.random.choice(int(N * N / 4), m_between, replace=False)
        links_between = np.array([ids_between % (N / 2), (N / 2) + (ids_between // (N / 2))], dtype=int).T
        ids_inside = np.random.choice(int(N * (N - 2) / 4), m_inside, replace=False)
        if (N / 2) % 2:
            links_inside = np.array([ids_inside % (N / 2), (ids_inside // (N / 2))], dtype=int).T
            links_inside[:, 0] = links_inside[:, 0] + (links_inside[:, 1] // ((N - 2) / 4)) * (N / 2)
            links_inside[:, 1] = links_inside[:, 1] % ((N - 2) / 4)
            links_inside[:, 1] = (links_inside[:, 0] + links_inside[:, 1] + 1) % (N / 2)
            links_inside[:, 1] = links_inside[:, 1] + (links_inside[:, 0] // (N / 2)) * (N / 2)
        else:
            links_inside = np.array([ids_inside % (N / 4), (ids_inside // (N / 4))], dtype=int).T
            links_inside[:, 0] = links_inside[:, 0] + (links_inside[:, 1] // ((N / 2) - 1)) * (N / 2)
            links_inside[:, 1] = links_inside[:, 1] % ((N / 2) - 1)
            links_inside[:, 0] = links_inside[:, 0] + (links_inside[:, 1] // (N / 4)) * N / 4
            links_inside[:, 1] = links_inside[:, 1] % (N / 4)
            links_inside[:, 1] = (links_inside[:, 0] + links_inside[:, 1] + 1) % (N / 2)
            links_inside[:, 1] = links_inside[:, 1] + (links_inside[:, 0] // (N / 2)) * (N / 2)
        links = np.concatenate([links_inside, links_between])
    if randomize_nodes:
        permutation = np.random.permutation(N)
        links = permutation[links]
    return links
