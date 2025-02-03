import numpy as np
from collections import Counter

class Node:
    def __init__(self,feature=None,threshold=None,left=None,right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


    def is_leaf_node(self):
        return self.value is not None



class My_DecisionTree:



    def __init__(self,min_samples_split=2,max_depth=100,n_features=None):
        self.min_samples_split =min_samples_split   #minimum number of rows required to split 
        self.max_depth = max_depth   #max depth of tree
        self.n_features = n_features   #number of features to consider for best split
        self.root=None


    def fit(self,X,y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features,X.shape[1])  #if n_features is not given then take all the features
        self.root = self._build_tree(X,y)
    

    def _build_tree(self,X,y,depth=0):

        n_samples,n_feats = X.shape     #number of samples and features
        n_labels = len(np.unique(y))    #number of unique labels in target

        #check the stopping criteria

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        feature_indexs = np.random.choice(n_feats,self.n_features,replace=False)   #random sample the  features 

        #Find the best split
        best_thresh,best_feature=self._best_split(X,y,feature_indexs) #Find the best column/ feature to split 

        # and also the best value in that column(best threshold) to split on

        left_indexs,right_indexs = self._split(X[:,best_feature],best_thresh)    #split based on the best feature and threshold


        left = self._build_tree(X[left_indexs,:],y[left_indexs],depth+1)
        right = self._build_tree(X[right_indexs,:],y[right_indexs],depth+1)
        
        return Node(best_feature,best_thresh,left,right)


    def _most_common_label(self,y):
        if len(y) == 0:
            return 0
        counter = Counter(y)
        if not counter:
            return 0
        return counter.most_common(1)[0][0] 


    def _best_split(self,X,y,feat_idxs):
        best_gain=-1
        split_idx,split_thresh = None,None

        for feat_idx in feat_idxs:  #for each feature in features array
            X_column = X[:,feat_idx]   # take a particular column
            thresholds = np.unique(X_column)    #unique values in a column

            for threshold in thresholds:  #for each threshold
                gain = self._information_gain(y,X_column,threshold)  #find the information gain for that threshold

                if gain > best_gain:     # if the gain we get is getter 
                    best_gain = gain             #update the best gain
                    split_idx = feat_idx         #update the best feature/column to split on
                    split_thresh = threshold     #update the best threshold  in that feature to split on

        return split_thresh,split_idx

    def _information_gain(self,y,X_column,threshold):  # we pass a column/feature, target and the thresholdfrom that column

        parent_entropy = self._entropy(y)       #calculate entropy
        left_idx,right_idx=self._split(X_column,threshold)   #split the column based on the threshold
        if len(left_idx) == 0 or len(right_idx) == 0:           #if the split is not possible 
            return 0
        
        n=len(y)   #total number of rows
        n_l,n_r=len(left_idx),len(right_idx)   #number of rows in left and right child
        e_l,e_r=self._entropy(y[left_idx]),self._entropy(y[right_idx])   #entropy of left and right child
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r   #weighted average of entropy of left and right child
        ig = parent_entropy - child_entropy    #information gain of that split 
        return ig
    

    def _split(self,X_column,threshold):
        left = np.argwhere(X_column <= threshold).flatten()  #get the indexs of rows where the value <= threshold
        right = np.argwhere(X_column > threshold).flatten()  #get the indexs of rows where the value > threshold
        return left,right
    
    def _entropy(self,y):
        hist = np.bincount(y)   #count the frequency of each unique value in the target
        ps = hist / len(y)       #calculate the probability  count/total
        return -np.sum([p * np.log2(p) for p in ps if p > 0])   #log 0 is undefined so filter
    
    def predict(self,X):
        return np.array([self._classify(x,self.root) for x in X])   
    
    def _classify(self,x,node):
        if node.is_leaf_node():   
            return node.value   #if it is a leaf node return the value which is the most common label in that node
        
        if x[node.feature] <= node.threshold:         # for the row we passing if the value of that node/split feature                                        #
            return self._classify(x,node.left)
                # is less than the threshold then go to left child
        return self._classify(x,node.right)            # else go to right child
        
    