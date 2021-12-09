import numpy as np

'''
    - TreeStruct is a simple wrapper around sklearn.tree.DecisionTreeClassifier class 
    - tree's strcture is encoded using children_left and children_right lists
    - number of elements in each list corresponds to the number of nodes in a tree
    - if node is a leaf its children are encoded with -1
    - if node is deleted its children are encoded with -5
    - tree nodes are being numerated in PREORDER fashion
    - it is possible to construct a tree from children_left and children_right and vice versa
''' 

class TreeStruct():
    TREE_LEAF = -1
    DELETED_LEAF = -5

    def __init__(self, tree):
        self.children_left = tree.children_left
        self.children_right = tree.children_right
        self.value = tree.value
        self.update_leaves() # collect additional information about tree structure related to leaves

    def update_leaves(self):
        self.leaves = np.nonzero(self.children_left==TreeStruct.TREE_LEAF)[0] # get nodes that do not have children (leaves)
        if self.leaves.shape[0] == 1:
            # this tree has been pruned to the root, we should delete it from the list of estimators
            return True # there is only root left - this tree should be removed from estimators
        
        # for each leaf in tree find its sibling and store it in leaf_siblings
        # example: if   children_left  = [1,2,-1,4,-1,-1,7,-1,-1]
        #               children_right = [6,3,-1,5,-1,-1,8,-1,-1]
        #               leaves         = [2,4,5,7,8]
        #          then leaf_siblings  = [3,5,4,8,7]
        self.leaf_siblings = np.array([self.find_sibling_node(leaf) for leaf in self.leaves])
        
        # define position of each node in leaves list (-1 if node is not leaf) and store it in leaf_position list
        # NOTE: it will be useful in later tree manipulation
        self.leaf_pos = np.zeros(self.children_left.shape[0],dtype=np.int32) - 1
        positions = np.arange(self.leaves.shape[0])
        self.leaf_pos[self.leaves] = positions
        
        return False # this tree has more than one node - it does not have to be removed from estimators

    def find_sibling_node(self, node):
        # try to find leaf in children_left
        left = np.nonzero((self.children_left==node))[0]
        if left.shape[0] > 0:
            # this leaf is left child - add its right sibling to the list
            return self.children_right[left[0]]
            
        # try to find leaf in children_right
        right = np.nonzero((self.children_right==node))[0]
        # this leaf is right child - add its left sibling to the list
        return self.children_left[right[0]]

    # checks whether the given node is leaf
    def is_leaf(self, node):
        return self.children_left[node] == TreeStruct.TREE_LEAF
        
    # checks whether the given node is deleted in merging process    
    def is_pruned(self, node):
        return self.children_left[node] == TreeStruct.DELETED_LEAF

    # for every leaf returns the position of its siblings in leaves list and -1 if sibling is not leaf
    def sibling_leaf_positions(self):
        return self.leaf_pos[self.leaf_siblings]

    # merge target leaf and its sibling to their parent
    def merge_leaves(self, leaf):
        if self.is_pruned(leaf):
            return False # this leaf has been already merged
        else:
            assert self.is_leaf(leaf) # it is not possible to merge internal node
            
            sib = self.leaf_siblings[self.leaf_pos[leaf]]
            if not self.is_leaf(sib):
                return False # it is not possible to merge leaf and internal node
                
            # mark target leaf and its sibling as deleted
            self.children_left[[leaf, sib]] = TreeStruct.DELETED_LEAF
            self.children_right[[leaf, sib]] =  TreeStruct.DELETED_LEAF
            
            # get parent of target leaf and its sibling
            parent = np.nonzero(np.logical_or(self.children_left==leaf,self.children_right == leaf))[0][0]
            
            # set parent of target leaf and its sibling as a leaf
            self.children_left[parent] = TreeStruct.TREE_LEAF
            self.children_right[parent] = TreeStruct.TREE_LEAF
            
            return True