import os
from shutil import copyfile
import shutil

import torch
import yaml
import numpy as np
import networkx as nx

import math
#from tqdm import tqdm
from bisect import bisect_left, bisect_right
import numpy as np


def gen_slice_tree_weight(out_dim):
    G = nx.DiGraph()

    N = int(2*out_dim)

    root_node = 'node0'
    nodes_inter = []
    nodes_leaf = []
    weight = np.zeros(N)

    for ii in range(N):
        node_name = 'node%d' % (ii)

        if ii % 2 == 0:
            nodes_inter.append(node_name)
            if node_name not in G:
                G.add_node(node_name)
                G.add_edge('node%d' % (ii-2), 'node%d' % ii)
        else:
            nodes_leaf.append(node_name)
            if node_name not in G:
                G.add_node(node_name)
                G.add_edge('node%d' % (ii-1), 'node%d' % ii)


    for ii in range(out_dim):
        weight[ii] = 1/out_dim
        weight[ii+out_dim] = 1 - ii/out_dim

            # print('%d   %d' % (int(ii/K),(ii+1)))
    B, nodes_tree = get_B_matrix_networkx_no_inter(G, root_node, nodes_inter, nodes_leaf)

    Bw = B.transpose()*weight
    # Equally weight
    Bw = Bw[:, 1:]
    return Bw, nodes_tree

def gen_slice_tree(out_dim):
    G = nx.DiGraph()

    N = int(2*out_dim)

    root_node = 'node0'
    nodes_inter = []
    nodes_leaf = []
    weight = np.zeros(N)

    for ii in range(N):
        node_name = 'node%d' % (ii)

        if ii % 2 == 0:
            nodes_inter.append(node_name)
            if node_name not in G:
                G.add_node(node_name)
                G.add_edge('node%d' % (ii-2), 'node%d' % ii)
        else:
            nodes_leaf.append(node_name)
            if node_name not in G:
                G.add_node(node_name)
                G.add_edge('node%d' % (ii-1), 'node%d' % ii)


    for ii in range(out_dim):
        weight[ii] = 1/out_dim
        weight[ii+out_dim] = 1 - ii/out_dim

            # print('%d   %d' % (int(ii/K),(ii+1)))
    B, nodes_tree = get_B_matrix_networkx_no_inter(G, root_node, nodes_inter, nodes_leaf)

    Bw = B.transpose()
    # Equally weight
    Bw = Bw[:, 1:]
    return Bw, nodes_tree


def gen_cluster_tree_sample(K, D,out_dim):
    G = nx.DiGraph()

    N = 0

    for k in range(D + 1):
        N += K ** k

    root_node = 'node0'
    nodes_inter = [root_node]
    nodes_leaf = []
    G.add_node(root_node)
    weight = np.zeros(N)
    weight_val = 1



    for ii in range(N - 1):
        node_name = 'node%d' % (ii + 1)

        if ii < (N - K ** D - 1):
            nodes_inter.append(node_name)
        else:
            nodes_leaf.append(node_name)

        if node_name not in G:
            G.add_node(node_name)
            G.add_edge('node%d' % int(ii / K), 'node%d' % (ii + 1))


            # print('%d   %d' % (int(ii/K),(ii+1)))
    B, nodes_tree = get_B_matrix_networkx_no_inter(G, root_node, nodes_inter, nodes_leaf)

    # Equally weight
    Bw = 1 / D * B[1:,:].transpose()

    np.random.seed(0)
    perm_ind = np.random.permutation(Bw.shape[0])

    Bw = Bw[perm_ind[:out_dim]]
    return Bw, nodes_tree



def gen_quad_tree_sample(D,out_dim):
    G = nx.DiGraph()

    K = 4
    N = 0

    if out_dim > K**D:
        print('out_dim is too large')

    weight_th = []
    val = 0
    for k in range(D + 1):
        N += K ** k
        val += K**(k+1)
        weight_th.append(val)

    root_node = 'node0'
    nodes_inter = [root_node]
    nodes_leaf = []
    G.add_node(root_node)
    weight = np.zeros(N)
    weight_val = 1
    weight_cnt = 0
    weight_sum = 1
    for ii in range(N-1):
        node_name = 'node%d' % (ii + 1)

        if ii < (N - K ** D - 1):
            nodes_inter.append(node_name)
        else:
            nodes_leaf.append(node_name)

        if node_name not in G:
            G.add_node(node_name)
            G.add_edge('node%d' % int(ii / K), 'node%d' % (ii + 1))

        weight[ii] = weight_val
        if ii == weight_th[weight_cnt]:
            weight_val = weight_val/2
            weight_cnt += 1
            weight_sum += weight_val
    weight[N-1] = weight_val
            # print('%d   %d' % (int(ii/K),(ii+1)))
    B, nodes_tree = get_B_matrix_networkx_no_inter(G, root_node, nodes_inter, nodes_leaf)

    Bw = B.transpose()*weight
    # Equally weight
    Bw = Bw[:, 1:]/weight_sum

    np.random.seed(0)
    perm_ind = np.random.permutation(Bw.shape[0])

    Bw = Bw[perm_ind[:out_dim]]
    return Bw, nodes_tree

def gen_inv_quad_tree(D):
    G = nx.DiGraph()

    K = 4
    N = 0

    weight_th = []
    val = 0
    for k in range(D + 1):
        N += K ** k
        val += K**(k+1)
        weight_th.append(val)

    root_node = 'node0'
    nodes_inter = [root_node]
    nodes_leaf = []
    G.add_node(root_node)
    weight = np.zeros(N)
    weight_val = 1
    weight_cnt = 0
    weight_sum = 1
    for ii in range(N-1):
        node_name = 'node%d' % (ii + 1)

        if ii < (N - K ** D - 1):
            nodes_inter.append(node_name)
        else:
            nodes_leaf.append(node_name)

        if node_name not in G:
            G.add_node(node_name)
            G.add_edge('node%d' % int(ii / K), 'node%d' % (ii + 1))

        weight[ii] = weight_val
        if ii == weight_th[weight_cnt]:
            weight_val = weight_val*2
            weight_cnt += 1
            weight_sum += weight_val
    weight[N-1] = weight_val
            # print('%d   %d' % (int(ii/K),(ii+1)))
    B, nodes_tree = get_B_matrix_networkx_no_inter(G, root_node, nodes_inter, nodes_leaf)

    Bw = B.transpose()*weight
    # Equally weight
    Bw = Bw[:, 1:]/weight_sum
    return Bw, nodes_tree

def gen_quad_tree(D):
    G = nx.DiGraph()

    K = 4
    N = 0

    weight_th = []
    val = 0
    for k in range(D + 1):
        N += K ** k
        val += K**(k+1)
        weight_th.append(val)

    root_node = 'node0'
    nodes_inter = [root_node]
    nodes_leaf = []
    G.add_node(root_node)
    weight = np.zeros(N)
    weight_val = 1
    weight_cnt = 0
    weight_sum = 1
    for ii in range(N-1):
        node_name = 'node%d' % (ii + 1)

        if ii < (N - K ** D - 1):
            nodes_inter.append(node_name)
        else:
            nodes_leaf.append(node_name)

        if node_name not in G:
            G.add_node(node_name)
            G.add_edge('node%d' % int(ii / K), 'node%d' % (ii + 1))

        weight[ii] = weight_val
        if ii == weight_th[weight_cnt]:
            weight_val = weight_val/2
            weight_cnt += 1
            weight_sum += weight_val
    weight[N-1] = weight_val
            # print('%d   %d' % (int(ii/K),(ii+1)))
    B, nodes_tree = get_B_matrix_networkx_no_inter(G, root_node, nodes_inter, nodes_leaf)

    Bw = B.transpose()*weight
    # Equally weight
    Bw = Bw[:, 1:]/weight_sum
    return Bw, nodes_tree

def gen_weight_cluster_tree(K,D,inv_weight=1.0):
    G = nx.DiGraph()

    N = 0

    weight_th = []
    val = 0
    for k in range(D + 1):
        N += K ** k
        val += K**(k+1)
        weight_th.append(val)

    root_node = 'node0'
    nodes_inter = [root_node]
    nodes_leaf = []
    G.add_node(root_node)
    weight = np.zeros(N)
    weight_val = 1
    weight_cnt = 0
    weight_sum = 1
    for ii in range(N-1):
        node_name = 'node%d' % (ii + 1)

        if ii < (N - K ** D - 1):
            nodes_inter.append(node_name)
        else:
            nodes_leaf.append(node_name)

        if node_name not in G:
            G.add_node(node_name)
            G.add_edge('node%d' % int(ii / K), 'node%d' % (ii + 1))

        weight[ii] = weight_val
        if ii == weight_th[weight_cnt]:
            weight_val = weight_val/inv_weight
            weight_cnt += 1
            weight_sum += weight_val
    weight[N-1] = weight_val
            # print('%d   %d' % (int(ii/K),(ii+1)))
    B, nodes_tree = get_B_matrix_networkx_no_inter(G, root_node, nodes_inter, nodes_leaf)

    Bw = B.transpose()*weight
    # Equally weight
    Bw = Bw[:, 1:]/weight_sum
    return Bw, nodes_tree

def gen_cluster_tree(K, D):
    G = nx.DiGraph()

    N = 0

    for k in range(D + 1):
        N += K ** k

    root_node = 'node0'
    nodes_inter = [root_node]
    nodes_leaf = []
    G.add_node(root_node)
    weight = np.zeros(N)
    weight_val = 1



    for ii in range(N - 1):
        node_name = 'node%d' % (ii + 1)

        if ii < (N - K ** D - 1):
            nodes_inter.append(node_name)
        else:
            nodes_leaf.append(node_name)

        if node_name not in G:
            G.add_node(node_name)
            G.add_edge('node%d' % int(ii / K), 'node%d' % (ii + 1))


            # print('%d   %d' % (int(ii/K),(ii+1)))
    B, nodes_tree = get_B_matrix_networkx_no_inter(G, root_node, nodes_inter, nodes_leaf)

    # Equally weight
    Bw = 1 / D * B[1:,:].transpose()
    return Bw, nodes_tree



def get_B_matrix_networkx_no_inter(T, root_node, nodes_inter,nodes_leaf, nodes_tree=[]):
    """
    Usage:
    #G is a Graph (networkx format)

    T = nx.dfs_tree(G, root_node)
    B,nodes_tree = get_matrix_networkx(G,root_node,nodes_tree=labels)
    Bsp = sparse.csc_matrix(B)

    :param root_node:
    :return: B, nodes_tree
    """
    if len(nodes_tree) == 0:
        nodes_tree = nodes_inter + nodes_leaf

    dict_nodes = {}
    ii = 0
    for node in nodes_tree:
        dict_nodes[node] = ii
        ii += 1

    B = np.zeros((len(nodes_tree), len(nodes_leaf)))
    ii = 0
    for node in nodes_leaf:
        node_current = node
        B[dict_nodes[node_current], ii] = 1
        B[dict_nodes[root_node], ii] = 1
        while node_current is not root_node:
            try:
                node_current = list(T.predecessors(node_current))[0]
                B[dict_nodes[node_current], ii] = 1
            except:
                node_current = root_node
        ii += 1

    return B, nodes_tree

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def split_softmax(input_matrix, num_split=2):
    # ??????????
    num_columns,d = input_matrix.size()

    # d/2??????????
    split_size = d // num_split
    chunks = torch.chunk(input_matrix, chunks=split_size, dim=1)

    # ?????????????softmax???
    softmax_chunks = [torch.softmax(chunk, dim=1) for chunk in chunks]

    # softmax??????????
    output_matrix = torch.cat(softmax_chunks, dim=1)

    return output_matrix

def barycenter(a_list, B_list, max_iter=1500, a=0.05, c=1 / 4, init_x="l2", debug_mode=False):
        """
        FastPSD

        Parameters
        ----------
        a_list : list of numpy.ndarray (shape is (n_doc, n_words))
            probability measures
        max_iter : int
            a maximum number of iterations
        a : float
            step size
        init_x : str
            if init_x is "random", initial value is random. If init_x is "l2", L2 barycenter is used as the initial value

        Return
        ----------
        min_x : numpy.ndarray (shape is (n_words))
            the FS-TSWB
        """

        n = len(a_list)
        n_leaf = a_list[0].shape[0]

        if init_x == "random":
            while True:
                x = np.random.randn(a_list[0].shape[0]).astype(np.float32)
                x = np.where(x < 0, 0, x)

                if x.sum() != 0:
                    break
            x = x / x.sum()
        elif init_x == "l2":
            x = sum(a_list) / n
        else:
            print("ERROR : random or l2")

        # Ba_list = []
        Ba_list_sum = []
        sorted_Ba = []
        cum_list = []

        for B in B_list:
            Ba_list = np.concatenate([B.dot(a_list[i].astype(np.float32))[:, np.newaxis] for i in range(n)], axis=1)
            # Ba_list.append(np.concatenate([B.dot(a_list[i].astype(np.float32))[:, np.newaxis] for i in range(n)], axis=1))
            sorted_Ba.append(np.sort(Ba_list, axis=1))
            Ba_list_sum.append(Ba_list.sum())
            cum_list.append(
                np.concatenate([np.zeros(Ba_list.shape[0])[:, np.newaxis], np.cumsum(sorted_Ba[-1], axis=1)], axis=1))

        min_val = np.inf
        min_x = x
        loss_list = []

        for k in tqdm(range(max_iter)):

            grad = np.zeros(n_leaf)
            loss = 0.0
            sort_idx_list = []
            for i in range(len(B_list)):
                Bx = B_list[i].dot(x)
                sort_idx_list.append(
                    np.array([bisect_right(sorted_Ba[i][j], Bx[j]) for j in range(self.B_list[i].shape[0])]))
                grad_b = 2 * sort_idx_list[-1] - n
                grad += Bt_list[i].dot(grad_b) / n

                # Compute loss
                loss += Ba_list_sum[i] + ((2 * sort_idx_list[i] - n) * Bx).sum()
                for j in range(len(sort_idx_list[i])):
                    loss -= 2 * cum_list[i][j][sort_idx_list[i][j]]

            grad /= self.n_slice
            loss /= n

            gamma = a / math.pow(k + 1.0, c) / np.linalg.norm(grad)
            x = x - gamma * grad
            x = self.projection_simplex_sort(x)

            if min_val > loss:
                min_val = loss
                min_x = x
            if debug_mode:
                loss_list.append(loss)

        if debug_mode:
            return min_x, loss_list

        return min_x


def projection_simplex_sort(v, z=1):
    # https://gist.github.com/mblondel/6f3b7aaad90606b98f71
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


def barycenter_naive(a_list, B, max_iter=1500, a=0.05, c=1 / 4, init_x="l2", debug_mode=False):
    """
    PSD
    n_slice=1
    """
    n = len(a_list)

    if init_x == "random":
        while True:
            x = np.random.randn(a_list[0].shape[0]).astype(np.float32)
            x = np.where(x < 0, 0, x)

            if x.sum() != 0:
                break
        x = x / x.sum()
    elif init_x == "l2":
        x = a_list.mean(1)
    else:
        print("ERROR : random or l2")

    Ba = B.dot(a)
    min_val = np.inf
    min_x = x
    loss_list = []

    for k in tqdm(range(max_iter)):
        Bx = B.dot(x)
        tmp = Bx[:, np.newaxis] - Ba
        # grad = self.Bt_list[0].dot(np.sign(Bx[:, np.newaxis] - Ba_list)).sum(1) / n
        grad = B.transpose().dot(np.sign(tmp)).sum(1) / n

        gamma = a / math.pow(k + 1.0, c) / np.linalg.norm(grad)
        x = x - gamma * grad
        x = projection_simplex_sort(x)

        # compute loss
        # loss = np.abs(Ba_list - Bx[:, np.newaxis]).sum()
        loss = np.abs(tmp).sum()

        if min_val > loss:
            min_val = loss
            min_x = x

        if debug_mode:
            loss_list.append(loss)
    # print("loss", self.get_obj_func(self.B, a_list)(x), min_val)
    if debug_mode:
        return min_x, loss_list
    return min_x

def _create_model_training_folder(writer, files_to_same):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))