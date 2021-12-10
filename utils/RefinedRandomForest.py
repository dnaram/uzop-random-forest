import numpy as np

from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
from utils.TreeWrapper import TreeStruct


class RefinedRandomForest():
    def __init__(self, rf, C = 1.0, prune_pct = 0.1, n_prunings = 1):
        self.rf_ = rf
        self.C = C
        self.prune_pct = prune_pct
        self.n_prunings = n_prunings
        self.trees_ = [TreeStruct(tree.tree_) for tree in rf.estimators_] # create TreeWrapper around every estimator tree
        self.leaves()

    def leaves(self):
        # store number of leaves per tree and total number of leaves
        self.n_leaves_ = [tree.leaves.shape[0] for tree in self.trees_]
        
        # store cumulative sum of leaves per tree starting from 0
        self.M = np.sum(self.n_leaves_)
        self.offsets_ = np.zeros_like(self.n_leaves_)
        self.offsets_[1:] = np.cumsum(self.n_leaves_)[:-1]
        
        # prepare lists for leaves and their corresponding trees (size M)
        self.ind_trees_ = np.zeros(self.M,dtype=np.int32)
        self.ind_leaves_ = np.zeros(self.M,dtype=np.int32)
        
        # for every estimator tree map its leaves and index of tree they are belonging to
        for tree_ind, tree in enumerate(self.trees_):
            start = self.offsets_[tree_ind]
            end = self.offsets_[tree_ind+1] if tree_ind+1<len(self.trees_) else self.M
            # store leaf nodes (in ind_leaves_) and tree indeces (in ind_trees) they are belonging to
            self.ind_trees_[start:end] = tree_ind
            self.ind_leaves_[start:end] = tree.leaves

    def get_indicators(self, X):
        # for each element x in X and for each tree in the forest
        # return the index of the leaf x ends up in and store in `leaf` matrix
        leaf = self.rf_.apply(X)
        
        # index for each sample from 0 to N-1
        sample_ind = np.arange(X.shape[0])
        row_ind = []
        col_ind = []
        for tree_ind, tree in enumerate(self.trees_):
            # get leaf l example x ended up in one particular tree (column in leaf matrix) for every x in X
            X_leaves = leaf[:,tree_ind]
            # map example's id with its corresponding leaf (+ offset defining tree)
            row_ind.append(sample_ind)
            col_ind.append(self.offsets_[tree_ind]+tree.leaf_pos[X_leaves])
            
        row_ind = np.concatenate(row_ind)
        col_ind = np.concatenate(col_ind)
        
        data = np.ones_like(row_ind) # find which elements in sparse matrix are to be equal to 1
        
        indicators = csr_matrix((data, (row_ind, col_ind)), shape=(X.shape[0],self.M))
        return indicators

    def prune_trees(self):
        # for every leaf in every esitmator tree get the index of its leaf sibling
        ind_siblings = np.zeros_like(self.ind_leaves_)
        for tree_ind, tree in enumerate(self.trees_):
            offset = self.offsets_[tree_ind]
            sibl_ind = tree.sibling_leaf_positions()
            sibl_ind[sibl_ind>=0] += offset
            start = self.offsets_[tree_ind]
            end = self.offsets_[tree_ind+1] if tree_ind+1<len(self.trees_) else self.M
            ind_siblings[start:end] = sibl_ind
            
        # get coefficient corresponding to each of the leaves in the estimator trees
        coef = self.lr.coef_
        if type(self.rf_) == RandomForestClassifier:
            sibl_coef = coef[:,ind_siblings]
            sibl_coef[:,ind_siblings < 0] = np.inf # so that it does not happen that leaf is being merged with branch
        else:
            sibl_coef = coef[ind_siblings]
            sibl_coef[ind_siblings < 0] = np.inf # so that it does not happen that leaf is being merged with branch
        
        # it is possible now to compare coefficients between leaves and its siblings
        if type(self.rf_) == RandomForestClassifier:
            sum_coef = np.sum(coef**2 + sibl_coef**2,axis=0)
        else:
            sum_coef = coef**2 - sibl_coef**2
        
        # sorting the difference in coefficients in ascending order by arguments
        ind = np.argsort(sum_coef)
        
        # we want to prune 10% of the least significant leaves
        if type(self.rf_) == RandomForestClassifier:
            n_prunings = np.floor(coef.shape[1] * self.prune_pct).astype(int)
        else:
            n_prunings = np.floor(len(coef) * self.prune_pct).astype(int)
        
        # let's start pruning
        pruned = 0
        i = 0
        while pruned < n_prunings:
            # get the least significant leaf and its corresponding tree
            tree_ind = self.ind_trees_[ind[i]]
            leaf_ind = self.ind_leaves_[ind[i]]
            
            # merge leaf and its sibling
            res = self.trees_[tree_ind].merge_leaves(leaf_ind)
            if res:
                pruned += 1
            # go to the next least significant leaf
            i += 1
            
        # check for trees which are not relevant any more for predictions 
        to_delete = []
        for tree_ind, tree in enumerate(self.trees_):
            if tree.update_leaves():
                to_delete.append(tree) # if there is only root left in the tree - remove it from the estimators
        
        # remove irrelevant trees from estimators
        for tree in to_delete:
            treeind = self.trees_.index(tree)
            del self.rf_.estimators_[treeind]
            self.trees_.remove(tree)
        self.leaves() # update number of leaves per tree, total number of leaves and other info

    def fit(self, X, y):
        n_pruned = 0
        while n_pruned <= self.n_prunings:
            indicators = self.get_indicators(X)
            #print('Model size: {} leaves'.format(indicators.shape[1]))
            #self.svr = SVR(C=self.C,fit_intercept=False,epsilon=0.)
            if type(self.rf_) == RandomForestClassifier:
                self.lr = LinearSVC(C=self.C, 
                                fit_intercept=True,
                                loss='hinge',
                                penalty='l2',
                                multi_class='ovr',
                                max_iter=1000)
            else:
                self.lr = Ridge(alpha=1/(2*self.C))
            self.lr.fit(indicators,y)
            if n_pruned < self.n_prunings:
                self.prune_trees()
            n_pruned += 1
        for tree_ind, tree in enumerate(self.trees_):
            offset = self.offsets_[tree_ind]
            if type(self.rf_) == RandomForestClassifier:
                tree.value[tree.leaves,0,:] = self.lr.coef_[:,offset:offset + tree.leaves.shape[0]].T
            else:
                tree.value[tree.leaves,0,:] = self.lr.coef_[offset:offset + tree.leaves.shape[0]].T.reshape(-1,1)


    def predict_proba(self, X):
        return self.lr.predict_proba(self.get_indicators(X))
        
    def predict(self, X):
        return self.lr.predict(self.get_indicators(X))