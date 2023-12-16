#!/usr/bin/env python3

'''
Find the maximum likelihood model transition rate matrix Q
for provided single-cell lineage data, cell locations, and tree topology

'''

import argparse
import numpy as np
import pandas as pd
import numpy.linalg as linalg
import networkx as nx
from ete3 import Tree, NodeStyle, TreeStyle
from TreeSolver.Cassiopeia_Tree import Cassiopeia_Tree

import scipy.stats as scs
from collections import defaultdict
from TreeSolver.simulation_tools import dataset_generation as data_gen
from Analysis import reconstruct_states
from Analysis import small_parsimony
import TreeSolver.compute_meta_purity as cmp


def likelihood_migration(m, ordering, Evals, T, T_inv, priors):
    '''
    Calculates the log-likelihood of a model transition rate matrix Q
    which has been decomposed into Evals, T, T_inv
    :param m: number of tissue locations
    :param ordering: list of nodes in post-traverse ordering
    :param Evals: Eigenvalues of matrix Q
    :param T: Matrix with columns eigenvectors of Q
    :param T_inv: inverse of T
    :param fp: prior of the model
    :return: log likelihood of the tree on the model
    '''
    def logP(i,j,t):
        Pij = (T[i,:]*np.exp(Evals*t)@T_inv[:,j]).real
        if Pij == 0:
            return -np.inf
        else:
            return np.log(Pij)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    base_map = {alphabet[i]:i for i in range(m)}
    np.seterr(divide = 'ignore')
    pi = np.log(np.array(priors))
    np.seterr(divide = 'warn')
    LL = 0
    for node in ordering:
        node.probs = [-np.inf for _ in range(m)]
        childs = node.children
        if len(childs) == 0:    #leaves; probs pre-initialized to 0
            node.probs[base_map[node.state]] = 0    #empirical state
        else:
            for k in range(m):  #ancestor base
                sum_left = -np.inf
                sum_right = -np.inf
                for n in range(m):  #descendant base
                    sum_left = sumLogProbs(sum_left, childs[0].probs[n]+logP(k,n,childs[0].branch))
                    sum_right = sumLogProbs(sum_right, childs[1].probs[n]+logP(k,n,childs[1].branch))
                node.probs[k] = sum_left+sum_right
    LLi = -np.inf
    root = ordering[-1]
    for n in range(m):
        LLi = sumLogProbs(LLi,pi[n]+root.probs[n])
    LL += LLi
    return LL
def likelihood_mutation(ordering, char_mat):
    '''
    Calculates the log-likelihood of a provided tree using the
    TideTree model.
    NOTE:  This is not yet used in this project
    :param ordering: list of nodes in post-traverse order
    :param char_mat: character matrix of single-cell lineage data
    :return: log-likelihood using TideTree model
    '''
    N = len(char_mat) # number of cells
    n = len(char_mat[0])   #number of target sites (states)
    k = n  #number of mutations in target site... set equal as in meta_sim (characters)
    t = 1
    l = 0.17       #matches simulations
    s = .03         #matches simulations
    P = tidetree_P(k,t,l,s)  #assume all branch lengths are the same
    np.seterr(divide = 'ignore')
    logP = np.log(P)
    np.seterr(divide = 'warn')
    base_map = {str(i):i for i in range(2,k+1)}  #char to prob index
    base_map['0'] = 0
    base_map['-1'] = 1
    # pi[0] = 1.0         #root is unedited; leaves are wysiwyg
    pi = [-np.inf for foo in range(0,k+2)]
    pi[0] = 0.0
    LL = 0
    for i in range(n):
        for node in ordering:
            node.probs = [-np.inf for foo in range(k+2)]
            childs = node.children
            if len(childs) == 0:    #leaves; probs pre-initialized to 0
                node.probs[base_map[node.char_vec[i]]] = 0.0    #empirical state
            else:
                for j in range(k+2):  #ancestor base
                    sum_left = -np.inf
                    sum_right = -np.inf
                    for p in range(k+2):  #descendant base
                        sum_left = sumLogProbs(sum_left, childs[0].probs[p]+logP[j,p])
                        sum_right = sumLogProbs(sum_right, childs[1].probs[p]+logP[j,p])
                    node.probs[j] = sum_left+sum_right
        LLi = -np.inf
        root = ordering[-1]
        #for j in range(k+2):
            # Li += pi[j]*root.probs[j]
        LLi = root.probs[0]
        LL += LLi
    return LL
def tidetree_P(k,t,l=0.1,s=1):  #for now assume all s are equal
    '''
    Generates the analytical, time-dependent probability
    transition matrix based on the TideTree model.
    NOTE:  This is not yet used in this project
    :param k: Number of characters
    :param t: time
    :param l: branch length
    :param s: rate parameter per site, currently assumed constant across all sites
    :return: probability transition matrix
    '''
    si = np.ones(k)*s
    r = l+np.sum(si)
    at = np.exp(-r*t)
    bt = np.exp(-l*t)
    ct = (bt-at)*si/np.sum(si)
    P = np.zeros((k+2,k+2))
    P = P + np.eye(k+2)*bt
    P[0,0] = at
    P[:,1] = np.ones(k+2)*(1-bt)
    P[1,1] = 1
    P[0,2:k+2] = ct
    return P
def sumLogProbs(a,b):
    if a == -np.inf and b == -np.inf:
        return -np.inf
    elif a > b:
        return a + np.log1p(np.exp(b-a))
    else:
        return b + np.log1p(np.exp(a-b))
def get_rate_matrix(m, t_mat = None, model='equal'):
    '''
    Returns various transition rate matrices for play analyses
    :param m: number of tissue locations
    :param t_mat: the true, simulated transition rate matrix
    :param model:
    if equal: returns a jukes-cantor-esque model (of size m)
    if rand: returns a randomly assigned matrix (with appropriate diagonals)
    if truth: returns simulated transition matrix
    :return:
    '''
    def equal(m):
        Q = np.ones((m,m))/(m-1)
        for i in range(m):
            Q[i][i] = -1
        return Q

    def rand(m):
        Q = np.random.rand(m,m)
        for i in range(m):
            agg = 0
            for j in range(m):
                if i != j: agg += Q[i,j]
            Q[i,:] = Q[i,:]/agg
            Q[i,i] = -1
        return Q

    def truth(m):
        Q = t_mat.copy()
        for i in range(m):
            Q[i][i] = -np.sum(Q[i,:])
        return Q

    if model == 'rand':
        Q = rand(m)
    elif model == 'truth':  Q = truth(m)
    else:  Q = equal(m)

    return Q
def P_t(Evals,T,Ti,t=1):
    '''
    Takes decomposed rate matrix Q and returns probability transition matrix
    :param Evals: Eigenvalues of matrix Q
    :param T: Matrix with columns eigenvectors of Q
    :param T_inv: inverse of T
    :param t: time
    :return: probability transition matrix (w.r.t. t)
    '''
    eD = np.diag(np.exp(Evals*t))
    Pt = (T@eD@Ti).real
    return Pt
def get_Q_params(Q):
    '''
    Takes matrix Q and returns its eigendecomposition
    :param Q: transition rate matrix
    :return Evals: Eigenvalues of matrix Q
    :return T: Matrix with columns eigenvectors of Q
    :return T_inv: inverse of T
    #:return fp: prior of the model
    '''
    Evals, T = np.linalg.eig(Q)
    Ti = linalg.inv(T)
    eD = np.diag(np.exp(Evals))  #take t = 1 for now, refine later
    fp = None
    # eps = 1e-4
    # eDD = eD.copy()
    # for i in range(100):      #defunct priors calculation
    #     eDD_old = eDD.copy()
    #     eDD = eDD*eDD_old
    #     #print(linalg.norm(eDD-eDD_old))
    #     if linalg.norm(eDD-eDD_old) < eps:
    #         fp = ((T@eDD@Ti)[0]).real
    #         if linalg.norm(fp,ord=1) == 0: break
    #         fp = fp / linalg.norm(fp,ord=1)
    #         return Evals, T, Ti, fp
    #     if linalg.norm(eDD-eDD_old) > 1e50:
    #         break
    # fp = np.ones(len(T))
    # fp = fp / linalg.norm(fp,ord=1)
    return Evals, T, Ti, fp
def count_metas(root):
    '''
    Counts the total metastases in a tree
    :param root: root node of the tree
    :return: number of metastases
    '''
    metas = 0
    for node in root.iter_search_nodes():
        for child in node.children:
            if node.state != child.state:
                metas += 1
    return metas
def assign_states(root):
    '''
    Updates the node.state value for each node in the tree
    based on the tissue with maximum probability
    :param root: root of the tree
    '''
    alphabet  = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for node in root.iter_search_nodes():
        node.state = alphabet[np.argmax(np.array(node.probs))]
def set_node_style(nodes, m):
    '''
    sets style of ete3 nodes (assuming m <= 6)
    :param nodes: should be root of the tree
    :param m: number of tissues, 6 or less
    '''
    nstyles = [NodeStyle() for i in range(m)]
    for ns in nstyles:
        ns["shape"] = "sphere"
        ns["size"] = 5
        ns["hz_line_type"] = 3
    nstyles[0]["fgcolor"] = "red"
    nstyles[1]["fgcolor"] = "orange"
    nstyles[2]["fgcolor"] = "green"
    nstyles[3]["fgcolor"] = "blue"
    nstyles[4]["fgcolor"] = "purple"
    nstyles[5]["fgcolor"] = "brown"
    nstyles[0]["hz_line_color"] = "red"
    nstyles[1]["hz_line_color"] = "orange"
    nstyles[2]["hz_line_color"] = "green"
    nstyles[3]["hz_line_color"] = "blue"
    nstyles[4]["hz_line_color"] = "purple"
    nstyles[5]["hz_line_color"] = "brown"
    for node in nodes.iter_search_nodes():
        if node.state == 'A': node.set_style(nstyles[0])
        if node.state == 'B': node.set_style(nstyles[1])
        if node.state == 'C': node.set_style(nstyles[2])
        if node.state == 'D': node.set_style(nstyles[3])
        if node.state == 'E': node.set_style(nstyles[4])
        if node.state == 'F': node.set_style(nstyles[5])
def dis_similarity_matrix(char_mat):
    '''
    Computes a similarity and dissimilarity matrix of the cells
    represented by a character matrix using hamming distances.
    :param char_mat: character matrix from single-cell lineage data
    :return: similarity matrix, dissimilarity matrix
    '''
    n_cells = len(char_mat[:,0])
    n_states = len(char_mat[0,:])
    sim_mat = np.zeros((n_cells, n_cells))
    for i in range(n_cells):
        for j in range(i,n_cells):
            for k in range(n_states):
                if char_mat[i,k] == char_mat[j,k]: sim_mat[i,j] += 1
            sim_mat[j,i] = sim_mat[i,j]
    sim_mat = sim_mat / n_states
    dissim_mat = 1 - sim_mat
    return sim_mat, dissim_mat
def compute_priors_mets(sim_data_mets, m = 6):
    '''
    Computes the percentage of each tissue in which
    cells are found
    :param sim_data_mets: list of tissues in which cells are found (length = N number of cells)
    :param m: number of tissue locations
    :return: priors
    '''
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    base_map = {alphabet[i]:i for i in range(m)}
    priors = np.zeros(m)
    for state in sim_data_mets:
        priors[base_map[state]] += 1
    priors = priors/len(sim_data_mets)
    return priors
def compute_Q_empirical_knn(sim_data_mets, dissim_mat, priors, k, m=6):
    '''
    Computes a transition rate matrix Q based on heuristics described
    below as well as some experimentation
    :param sim_data_mets: list of tissue location for each cell, in order of cell alignment in dssim_mat
    :param dissim_mat: dissimilarity matrix of cells
    :param priors: priors (defunct)
    :param k: k parameter for k-nearest neighbors
    :param m: number of tissue locations
    :return: Initial transition rate matrix Q
    '''
    #inputs organized by cell (index)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    base_map = {alphabet[i]:i for i in range(m)}
    Q = np.zeros((m,m))
    d = np.zeros((m,m)) #avg distances between locations
    count = np.zeros((m,m))
    count1 = np.zeros(m)
    N = len(sim_data_mets)
    for i in range(N):  #total cells
        state1 = sim_data_mets[i]
        count1[base_map[state1]] += 1  #total number of cell type i counted
        X = np.argpartition(dissim_mat[i],k)[:k]    #count k nearest neighbors
        for j in X:
            state2 = sim_data_mets[j]
            count[base_map[state1],base_map[state2]] += 1  #total number of cell type i near to cell type j


    #1. if you're near to stuff, rate from them to you high
    #2. if you're near to stuff, rate from you to them is high
    #3. if you've lots, rate to you is high
    #4. if you've lots, rate from you is low
    #5. if you've little, rate to you is low
    #6. if you've little, rate from you is high
    #7. if you're spread, rate from you is high
    #8. if you're spread, rate to you is high
    #9. if you're tight, rate from you is low
    #10. if you're tight, rate to you is low

    #1. i!=j prop to count
    #2. i!=j prop to count
    #3. i!=j col. prop to i=j
    #4. i!=j row inv. prop to i=j
    #5. i!=j prop to count1d(j)
    #6. i!=j inv. prop to count1d(i)
    #7,9. i!=j inv. prop. to q[i,i]/count1d(i)        #tightness
    #8,10. i!=j inv. prop. to q[i,i]/count1d(i)        #tightness

    Q = count.copy()
    for i in range(6):
        for j in range(6):
            if i!=j:
                if count1[j] != 0:
                    Q[i,j] = Q[i,j]*count1[i]/count1[j]
    Q = Q/np.max(Q)         #better normalization done later
    for i in range(m):
        Q[i,i] = 0
        Q[i,i] = -np.sum(Q[i,:])      #Q[i,i] dependent on the rest of the row

    return Q
def likelihood_wrt_Q(Q, ordering, priors):
    '''
    Wrapper to call likelihood_migration without decomposing Q first
    :param Q: transition rate matrix
    :param ordering: list of nodes in post-traversal ordering
    :param priors: priors
    :return: log-likelihood of the model on the tree
    '''
    m = 6
    #m = len(Q[0])
    Evals, T, T_inv, fp = get_Q_params(Q)
    LL = likelihood_migration(m, ordering, Evals, T, T_inv, priors)
    return LL
def refine_Q(Q,ordering,kk,priors,m=6):
    '''
    First go at stochastic optimization to refine Q.
    IMPROVE THIS METHOD NEXT!
    :param Q: Transition rate matrix Q to refine
    :param ordering: list of nodes in post-traversal order
    :param kk: number of times to perturb each element of Q
    :param priors: priors
    :param m: number of tissue locations
    :return: refined Q
    '''
    def sample_norm(x,sigma=0.05):     #x in range [0,1].  Must sample btwn [0,1].
        '''
        Junky sampling for stochastic optimization.  IMPROVE THIS.
        Tries to keep numbers copacetic
        :param x: element in Q in range [0,1]
        :param sigma: measure of amount to perturb
        :return: perturbed element
        '''
        sample = np.random.normal(x,sigma)
        if sample < 0: sample = sample/2
        if sample > 1: sample = 2-sample
        if sample > 1: sample = 0.99
        if sample < 0: sample = 0.99
        return sample
    Q_best = None
    LL_best = -np.inf
    LL_last = -np.inf
    Q1 = Q.copy()
    count = 0   #lets stop if no change after 5 k's
    for k in range(kk):
        for i in range(m):
            for j in range(m):
                if i != j:
                    x = sample_norm(Q1[i,j])
                    dx = x - Q1[i,j]
                    Q2 = Q1.copy()
                    Q2[i,j] = x
                    Q2[i,i] -= dx
                    #LL1 = likelihood_wrt_Q(Q1,ordering,priors)
                    LL2 = likelihood_wrt_Q(Q2,ordering,priors)
                    if LL2 > LL_last:
                        Q1 = Q2.copy()
                        LL_last = LL2
                    if LL2 > LL_best:
                        LL_best = LL2
                        Q_best = Q2.copy()
                        count = 0
                    if count > 5: break
        print("k, LL_best: ",k,LL_best)
        refine_bl(ordering, Q, priors, m=6)
        count += 1
    return Q_best
def refine_bl(post_ordering, Q, priors, m=6): #MAKE THIS MORE REFINED
    '''
    Searches for optimal universal branch length between [0.04,2)
    and updates all node branches to this universal length.
    This could use major improvements on accuracy and efficiency...
    ...and also should be superceded by an individual branch length optimizer
    :param post_ordering: list of nodes in post-order
    :param Q: transition rate matrix
    :param priors: priors
    :param m: number of tissue locations
    :return:
    '''
    best_bl = 0
    best_LL = -np.inf
    Evals, T, Ti, fp = get_Q_params(Q)
    for num in range(2,100):
        bl = num/50
        for node in post_ordering: node.branch = bl
        LL = likelihood_migration(m, post_ordering, Evals, T, Ti, priors)
        if LL > best_LL:
            best_bl = bl
            best_LL = LL
    for node in post_ordering: node.branch = best_bl
def mets_freq(root):
    '''
    Calculates the metastasis frequency(count) matrix of a reconstructed tree
    :param root: root node of tree with all nodes assigned a location
    :return: matrix of counts of transitions between each tissue location,
            total number of metastases
    '''
    met_freqs = defaultdict(dict)
    num_mets = 0
    alphabet = "ABCDEF"
    for i in range(6):
        for j in range(6):
            met_freqs[alphabet[i]][alphabet[j]] = 0

    for node in root.iter_search_nodes():
        childs = node.children
        m_p = node.state
        for child in childs:
            m_c = child.state
            if m_p != m_c:
                num_mets += 1
                met_freqs[m_p][m_c] += 1
    return met_freqs, num_mets
def simulate_tree_get_stats(t_depth):
    '''
    Much of this is a copy of code retrieved from Quinn et. al. 2021
    and has been modified for the purposes herein.

    Most of this method calls dataset_generation.py and methods therein
    (TreeSolver, Cassiopeia_Tree, etc.) to create the simulated dataset

    This also calculated the FitchCount Spearman Correlation (scorr_fitcher)
    which is used for comparative analysis.

    This has been copied/modified mostly to easily return and manipulate many
    of the parametersof the simulation

    :param t_depth: depth of tree to simulate
    :return:
    data_muts: the character matrix of the simulated data
    data_mets: the location of each leaf node (in chronological list order)
    tmat: the simulated true conditional transition rate probability matrix
    tree: true Cassiopeia_Tree of the simulation
    leaf_names: list of leaf names for lookup/reference
    mu: migration rate
    depth: depth of tree
    metsdf: true empirical conditional transition rate probability matrix
    scorr_fitcher: the spearman correlation coefficient for FitchCount
    scorr_naive: scorr for naive model
    scorr_mv: scorr for majority vote model
    '''
    def compute_priors(C, S, p, mean=0.01, disp=0.1, empirical = np.array([])):
        sp = {}
        prior_probabilities = {}
        for i in range(0, C):
            if len(empirical) > 0:
                sampled_probabilities = sorted(empirical)
            else:
                sampled_probabilities = sorted([np.random.negative_binomial(mean,disp) for _ in range(1,S+1)])
            mut_rate = p
            prior_probabilities[i] = {'0': (1-mut_rate)}
            total = np.sum(sampled_probabilities)

            sampled_probabilities = list(map(lambda x: x / (1.0 * total), sampled_probabilities))

            for j in range(1, S+1):
                prior_probabilities[i][str(j)] = (mut_rate)*sampled_probabilities[j-1]

        return prior_probabilities, sp
    def get_transition_stats(tree):
        n_transitions = 0
        transitions = defaultdict(dict)
        freqs = defaultdict(int)
        alphabet = "ABCDEF"
        for i in range(6):
            for j in range(6):
                transitions[alphabet[i]][alphabet[j]] = 0

        root = [n for n in tree if tree.in_degree(n) == 0][0]
        for e in nx.dfs_edges(tree, source=root):

            p,c = e[0], e[1]
            m_p, m_c = tree.nodes[p]['meta'], tree.nodes[c]['meta']
            if m_p != m_c:
                n_transitions += 1
                if m_c not in transitions[m_p]:
                    transitions[m_p][m_c] = 0
                transitions[m_p][m_c] += 1

            if tree.out_degree(c) == 0:
                freqs[m_c] += 1

        return n_transitions, transitions, freqs
    no_mut_rate = 0.985
    number_of_states = 20#40
    dropout = 0.17
    depth = 8
    number_of_characters = 20#40

    pp, sp = compute_priors(number_of_characters, number_of_states, 0.025, mean=1, disp=0.1)

    # define parameters:
    N_clones = 1 #number of clones to simulate
    max_mu = 0.3 #max rate of metastasis
    min_alpha = 0.75 #min rate of doubling
    t_range = [t_depth,t_depth]#[12,16]#[12,16] #range of time-steps [x,y)
    sigma = 6 #number of tumor samples
    beta = 0.00 #extinction rate

    # make samples
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    sample_list = [alphabet[i] for i in range(sigma)]

    def assign_majority_vote(t, root):

        def majority_vote(tree, rt):

            children = [node for node in nx.dfs_preorder_nodes(tree, rt) if tree.out_degree(node) == 0]
            children_vals = [tree.nodes[n]["S1"] for n in children]

            uniq_count = np.unique(children_vals, return_counts=True)
            label = uniq_count[0][np.argmax(uniq_count[1])]

            return label

        for n in nx.dfs_preorder_nodes(t, root):

            if t.out_degree(n) == 0:
                t.nodes[n]['label'] = t.nodes[n]['S1'][0]

            t.nodes[n]['label'] = majority_vote(t, n)

        return t
    def compute_transitions_majority_vote(t, meta):

        possible_labels = meta.unique()

        M = len(possible_labels)
        C = np.zeros((M, M))
        label_to_j = dict(zip(possible_labels, range(len(possible_labels))))

        root = [n for n in t if t.in_degree(n) == 0][0]
        t = small_parsimony.assign_labels(t, meta)

        t = cmp.set_depth(t, root)

        t = assign_majority_vote(t, root)

        # now count transitions
        for v in nx.dfs_postorder_nodes(t, source=root):

            v_lab = t.nodes[v]['label']
            i = label_to_j[v_lab]

            children = list(t.successors(v))
            for c in children:

                c_lab = t.nodes[c]['label']
                j = label_to_j[c_lab]

                C[i, j] += 1

        count_mat = pd.DataFrame(C)
        count_mat.columns = possible_labels
        count_mat.index = possible_labels
        return count_mat
    def kl_divergence(a, b):

        kl_a = np.sum([a[i]*np.log(a[i]/b[i]) for i in range(len(a))])
        kl_b = np.sum([b[i] * np.log(b[i]/a[i]) for i in range(len(b))])

        return kl_a + kl_b

    alpha = 1
    tmat = pd.DataFrame(np.zeros((6,6)), index=sample_list,  columns=sample_list)

    for i in tmat.index:
        thetas = np.random.dirichlet([alpha]*(sigma-1))
        tmat.loc[i, [j for j in tmat.columns if j != i]] = thetas
    print(tmat)
    depth = np.random.randint(t_range[0],t_range[1]+1)

    tree, params, mu = data_gen.generate_simulated_experiment_plasticity(pp,
                                                                         [0.18]*number_of_characters,
                                                                         characters=number_of_characters,
                                                                         subsample_percentage=0.5,
                                                                         dropout=True,
                                                                         sample_list = sample_list,
                                                                         max_mu = max_mu,
                                                                         min_alpha = min_alpha,
                                                                         depth = depth,
                                                                         beta = 0,
                                                                         transition_matrix = tmat
                                                                         )

    n_mets, mets, freqs = get_transition_stats(tree.network)

    leaves = [n for n in tree.network if tree.network.out_degree(n) == 0]
    t = tree.network
    t2 = t.copy()

    meta = pd.DataFrame.from_dict(dict(zip([n.name for n in leaves], [tree.network.nodes[n]['meta'] for n in leaves])), orient='index')
    est_freqs_naive = reconstruct_states.naive_fitch(t2, meta.loc[:,0])
    est_freqs = reconstruct_states.fitch_count(t, meta.loc[:,0])#, count_unique = False)
    est_freqs_mv = compute_transitions_majority_vote(t, meta.iloc[:, 0])

    metsdf = pd.DataFrame.from_dict(mets, orient='index').loc[sample_list, sample_list]



    def fill_out_df(df, list):
        shape = df.shape
        index = df.index
        for item in list:
            if item not in index:
                df.loc[item] = [0 for _ in range(shape[1])]
        columns = df.columns
        for item in list:
            if item not in columns:
                df.loc[:,[item]] = [0 for _ in range(len(list))]
        return df

    fill_out_df(est_freqs,sample_list)
    fill_out_df(est_freqs_naive,sample_list)
    fill_out_df(est_freqs_mv,sample_list)

    est_freqs = est_freqs.loc[sample_list, sample_list]
    est_freqs_naive = est_freqs_naive.loc[sample_list, sample_list]
    est_freqs_mv = est_freqs_mv.loc[sample_list, sample_list]

    np.fill_diagonal(est_freqs.values,0)
    est_freqs = est_freqs.fillna(value = 0)
    est_freqs = est_freqs.apply(lambda x: x / max(1, x.sum()), axis=1)

    np.fill_diagonal(est_freqs_naive.values,0)
    est_freqs_naive = est_freqs_naive.fillna(value = 0)
    est_freqs_naive = est_freqs_naive.apply(lambda x: x/max(1, x.sum()), axis=1)

    np.fill_diagonal(est_freqs_mv.values,0)
    est_freqs_mv = est_freqs_mv.fillna(value = 0)
    est_freqs_mv = est_freqs_mv.apply(lambda x: x/max(1, x.sum()), axis=1)

    np.fill_diagonal(metsdf.values,0)
    metsdf = metsdf.fillna(value = 0)
    metsdf = metsdf.apply(lambda x: x / max(1, x.sum()), axis=1)

    scorr_fitcher = scs.spearmanr(metsdf.values.ravel(), est_freqs.values.ravel())[0]

    scorr_naive = scs.spearmanr(metsdf.values.ravel(), est_freqs_naive.values.ravel())[0]

    scorr_mv = scs.spearmanr(metsdf.values.ravel(), est_freqs_mv.values.ravel())[0]

    data_muts = []
    data_mets = []
    leaf_names = []
    for i in range(len(leaves)):
        data_muts.append(leaves[i].char_vec)
        data_mets.append(tree.network.nodes[leaves[i]]['meta'])
        leaf_names.append(leaves[i])
    for j in range(len(data_muts[0])):
        for i in range(len(data_muts)):
            if data_muts[i][j] == '-':
                data_muts[i][j] = '-1'

    return data_muts, data_mets, tmat, tree, leaf_names, mu, depth, metsdf, scorr_fitcher, scorr_naive, scorr_mv
def get_all_stats(Q_iter,t_depth,render = False):
    '''
    Workhorse of the model.
        - Calls to simulate a tree
        - Creates initial Q matrix
        - Refined Q matrix
        - Calculates tissue probabilities of each cell
        - Calculates spearman coefficients of returned model

    :param Q_iter: Number of times to iterate through Q elements in the refinement stage
    :param t_depth: depth of tree to simulate
    :param render: whether or not to create a pdf image of the true and simulated tree
    :return:
        depth: depth of tree
        mu: migration rate
        LL: log likelihood of refined tree
        scorr_MLE_Q_knn: spearman correlation of the initial Q
        scorr_MLE_Q_best: spearman correlation of the highest LL Q
        scorr_fitcher: the spearman correlation coefficient for FitchCount
        scorr_naive: scorr for naive model
        scorr_mv: scorr for majority vote model
    '''
    m = 6
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    sample_list = [alphabet[i] for i in range(m)]

    sim_data_muts, sim_data_mets, tmat_cond, tree_truth, leaf_names, mu, depth, metsdf, scorr_fitcher, scorr_naive, scorr_mv = simulate_tree_get_stats(t_depth)

    nx = tree_truth.get_network()
    tree_truth_states = {}  #dictionary of node.name to meta
    for node in nx.nodes:
        tree_truth_states[node.name] = tree_truth.network.nodes[node]['meta']
    sim_data_muts = pd.DataFrame(
        sim_data_muts,
        columns=[f"r{i}" for i in range(1, len(sim_data_muts[0]) + 1)],
        index=[f"c{i}" for i in range(1, len(sim_data_muts) + 1)]
    ) #cell names for reference
    char_mat = sim_data_muts.to_numpy()
    root_truth = Tree(tree_truth.get_newick(), format=1)    #True Simulated Tree
    root_true_topo = Tree(tree_truth.get_newick(), format=1)    #True topology, empty otherwise
    for node in root_true_topo.iter_search_nodes():
        node.add_features(branch=1, probs=[], state=None, char_vec=None)
    for node in root_truth.iter_search_nodes():
        node.add_features(branch=1, probs=[], state=tree_truth_states[node.name], char_vec=None)
    sim_data_muts['state'] = sim_data_mets
    for leaf in root_true_topo.get_leaves():
        leaf.state = tree_truth_states[leaf.name]
    post_ordering = []
    for node in root_true_topo.traverse("postorder"):
        post_ordering.append(node)

    #time to get a Q and do some MLE
    priors = compute_priors_mets(sim_data_mets, m)
    sim_mat, dissim_mat = dis_similarity_matrix(char_mat)

    Q_knn = compute_Q_empirical_knn(sim_data_mets, dissim_mat, priors, 8)

    #set up metas in nodes
    Evals, T, Ti, fp = get_Q_params(Q_knn)
    LLt = likelihood_migration(m, post_ordering, Evals, T, Ti, priors)

    #quick interlude to get better branch lengths
    refine_bl(post_ordering,Q_knn,priors)

    LL = likelihood_migration(m, post_ordering, Evals, T, Ti, priors)  #get node.probs with best_bl
    assign_states(root_true_topo)   #use node.probs to assign states
    sample_list = [alphabet[i] for i in range(m)]
    mfreq_truth, nmets_truth = mets_freq(root_truth)
    mfreq_topo, nmets_topo = mets_freq(root_true_topo)
    metsdf_truth = pd.DataFrame.from_dict(mfreq_truth, orient='index').loc[sample_list, sample_list]
    metsdf_topo = pd.DataFrame.from_dict(mfreq_topo, orient='index').loc[sample_list, sample_list]
    mets_freq_truth = metsdf_truth.apply(lambda x: x / max(1, x.sum()), axis=1)
    mets_freq_topo = metsdf_topo.apply(lambda x: x / max(1, x.sum()), axis=1)

    scorr_MLE_Q_knn = scs.spearmanr(mets_freq_truth.values.ravel(), mets_freq_topo.values.ravel())[0]


    true_topo_knn_tiss = []
    truth_tiss = []
    for node in root_true_topo.traverse("postorder"):
        if not node.is_leaf():
            true_topo_knn_tiss.append(node.state)
    for node in root_truth.traverse("postorder"):
        if not node.is_leaf():
            truth_tiss.append(node.state)
    count_correct_knn = 0
    for i in range(len(true_topo_knn_tiss)):
        if true_topo_knn_tiss[i] == truth_tiss[i]:
            count_correct_knn += 1

    Q_best = refine_Q(Q_knn,post_ordering,Q_iter,priors,m=6)

    Evals, T, Ti, fp = get_Q_params(Q_best)
    LL = likelihood_migration(m, post_ordering, Evals, T, Ti, priors)  #get node.probs with best_bl
    assign_states(root_true_topo)   #use node.probs to assign states
    mfreq_topo_best, nmets_topo = mets_freq(root_true_topo)
    metsdf_topo_best = pd.DataFrame.from_dict(mfreq_topo_best, orient='index').loc[sample_list, sample_list]
    mets_freq_topo_best = metsdf_topo_best.apply(lambda x: x / max(1, x.sum()), axis=1)
    scorr_MLE_Q_best = scs.spearmanr(mets_freq_truth.values.ravel(), mets_freq_topo_best.values.ravel())[0]
    print("Q_knn:")
    print(Q_knn)
    print("Q_best:")
    print(Q_best)

    true_topo_tiss = []
    for node in root_true_topo.traverse("postorder"):
        if not node.is_leaf():
            true_topo_tiss.append(node.state)

    count_correct = 0
    for i in range(len(true_topo_tiss)):
        if true_topo_tiss[i] == truth_tiss[i]:
            count_correct += 1

    #print("PERCENTAGE CORRECT (Q_knn, Q_best): ",count_correct_knn/len(true_topo_knn_tiss),count_correct/len(true_topo_tiss))
    #Note:  Q refinement does not always increase percentage correct!  Revise Q refinement method.

    if render == True:
        set_node_style(root_truth,6)
        set_node_style(root_true_topo,6)

        ts = TreeStyle()
        ts.mode = "c"
        ts.show_leaf_name = False

        root_truth.render("truth.png",tree_style=ts,w=3,units='in',dpi=300)
        root_true_topo.render("true_topo.png",tree_style=ts,w=3,units='in',dpi=300)


    return depth, mu, LL, scorr_MLE_Q_knn, scorr_MLE_Q_best, scorr_fitcher, scorr_naive, scorr_mv



def main():
    '''
    Generates trees based on fields Q_iter and t_depth (number of
    times to iterate the Q refinement and depth of tree to simulate)
    saves results in datafiles.
    '''

    Q_iter = 20
    t_depth = 9
    num_trees = 30
    file_name = "test_data_"+str(Q_iter)+"_"+str(t_depth)+".txt"

    with open(file_name, "a") as file:
        for i in range(num_trees):
            print("tree reconstruction #: ",i)
            data = get_all_stats(Q_iter,t_depth,False)
            print(data)
            for dat in data:
                file.write(str(dat)+", ")
            file.write("\n")


if __name__ == "__main__":
    main()


