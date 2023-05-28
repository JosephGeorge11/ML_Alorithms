{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f7b401b5",
   "metadata": {},
   "outputs": [],
   "source": [
    "###################################################################################################################################"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ff04192e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from collections import Counter\n",
    "from random import random\n",
    "from sklearn import datasets\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "67d40277",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Node:\n",
    "    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):\n",
    "        self.feature = feature\n",
    "        self.threshold = threshold\n",
    "        self.left = left\n",
    "        self.right = right\n",
    "        self.value = value\n",
    "        \n",
    "    def is_leaf_node(self):\n",
    "        return self.value is not None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "63731d83",
   "metadata": {},
   "outputs": [],
   "source": [
    "class DecisionTree:\n",
    "    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):\n",
    "        self.min_samples_split=min_samples_split\n",
    "        self.max_depth=max_depth\n",
    "        self.n_features=n_features\n",
    "        self.root=None\n",
    "\n",
    "    def fit(self, X, y):\n",
    "        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)\n",
    "        self.root = self._grow_tree(X, y)\n",
    "\n",
    "    def _grow_tree(self, X, y, depth=0):\n",
    "        n_samples, n_feats = X.shape\n",
    "        n_labels = len(np.unique(y))\n",
    "\n",
    "        # check the stopping criteria\n",
    "        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):\n",
    "            leaf_value = self._most_common_label(y)\n",
    "            return Node(value=leaf_value)\n",
    "\n",
    "        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)\n",
    "\n",
    "        # find the best split\n",
    "        best_feature, best_thresh = self._best_split(X, y, feat_idxs)\n",
    "\n",
    "        # create child nodes\n",
    "        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)\n",
    "        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)\n",
    "        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)\n",
    "        return Node(best_feature, best_thresh, left, right)\n",
    "\n",
    "\n",
    "    def _best_split(self, X, y, feat_idxs):\n",
    "        best_gain = -1\n",
    "        split_idx, split_threshold = None, None\n",
    "\n",
    "        for feat_idx in feat_idxs:\n",
    "            X_column = X[:, feat_idx]\n",
    "            thresholds = np.unique(X_column)\n",
    "\n",
    "            for thr in thresholds:\n",
    "                # calculate the information gain\n",
    "                gain = self._information_gain(y, X_column, thr)\n",
    "\n",
    "                if gain > best_gain:\n",
    "                    best_gain = gain\n",
    "                    split_idx = feat_idx\n",
    "                    split_threshold = thr\n",
    "\n",
    "        return split_idx, split_threshold\n",
    "\n",
    "\n",
    "    def _information_gain(self, y, X_column, threshold):\n",
    "        # parent entropy\n",
    "        parent_entropy = self._entropy(y)\n",
    "\n",
    "        # create children\n",
    "        left_idxs, right_idxs = self._split(X_column, threshold)\n",
    "\n",
    "        if len(left_idxs) == 0 or len(right_idxs) == 0:\n",
    "            return 0\n",
    "        \n",
    "        # calculate the weighted avg. entropy of children\n",
    "        n = len(y)\n",
    "        n_l, n_r = len(left_idxs), len(right_idxs)\n",
    "        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])\n",
    "        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r\n",
    "\n",
    "        # calculate the IG\n",
    "        information_gain = parent_entropy - child_entropy\n",
    "        return information_gain\n",
    "\n",
    "    def _split(self, X_column, split_thresh):\n",
    "        left_idxs = np.argwhere(X_column <= split_thresh).flatten()\n",
    "        right_idxs = np.argwhere(X_column > split_thresh).flatten()\n",
    "        return left_idxs, right_idxs\n",
    "\n",
    "    def _entropy(self, y):\n",
    "        hist = np.bincount(y)\n",
    "        ps = hist / len(y)\n",
    "        return -np.sum([p * np.log(p) for p in ps if p>0])\n",
    "\n",
    "\n",
    "    def _most_common_label(self, y):\n",
    "        counter = Counter(y)\n",
    "        value = counter.most_common(1)[0][0]\n",
    "        return value\n",
    "\n",
    "    def predict(self, X):\n",
    "        return np.array([self._traverse_tree(x, self.root) for x in X])\n",
    "\n",
    "    def _traverse_tree(self, x, node):\n",
    "        if node.is_leaf_node():\n",
    "            return node.value\n",
    "\n",
    "        if x[node.feature] <= node.threshold:\n",
    "            return self._traverse_tree(x, node.left)\n",
    "        return self._traverse_tree(x, node.right)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "d7c1b190",
   "metadata": {},
   "outputs": [],
   "source": [
    "class RandomForest:\n",
    "    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None):\n",
    "        self.n_trees = n_trees\n",
    "        self.max_depth=max_depth\n",
    "        self.min_samples_split=min_samples_split\n",
    "        self.n_features=n_feature\n",
    "        self.trees = []\n",
    "\n",
    "    def fit(self, X, y):\n",
    "        self.trees = []\n",
    "        for _ in range(self.n_trees):\n",
    "            tree = DecisionTree(max_depth=self.max_depth,\n",
    "                            min_samples_split=self.min_samples_split,\n",
    "                            n_features=self.n_features)\n",
    "            X_sample, y_sample = self._bootstrap_samples(X, y)\n",
    "            tree.fit(X_sample, y_sample)\n",
    "            self.trees.append(tree)\n",
    "\n",
    "    def _bootstrap_samples(self, X, y):\n",
    "        n_samples = X.shape[0]\n",
    "        idxs = np.random.choice(n_samples, n_samples, replace=True)\n",
    "        return X[idxs], y[idxs]\n",
    "\n",
    "    def _most_common_label(self, y):\n",
    "        counter = Counter(y)\n",
    "        most_common = counter.most_common(1)[0][0]\n",
    "        return most_common\n",
    "\n",
    "    def predict(self, X):\n",
    "        predictions = np.array([tree.predict(X) for tree in self.trees])\n",
    "        tree_preds = np.swapaxes(predictions, 0, 1)\n",
    "        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])\n",
    "        return predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "565a751a",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = datasets.load_breast_cancer()\n",
    "X = data.data\n",
    "y = data.target\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=1234\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "fdcd77a9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1.288e+01 1.822e+01 8.445e+01 ... 1.096e-01 2.582e-01 8.893e-02]\n",
      " [1.113e+01 2.244e+01 7.149e+01 ... 6.413e-02 3.169e-01 8.032e-02]\n",
      " [1.263e+01 2.076e+01 8.215e+01 ... 1.105e-01 2.226e-01 8.486e-02]\n",
      " ...\n",
      " [1.247e+01 1.860e+01 8.109e+01 ... 1.015e-01 3.014e-01 8.750e-02]\n",
      " [1.822e+01 1.870e+01 1.203e+02 ... 1.325e-01 3.021e-01 7.987e-02]\n",
      " [1.272e+01 1.378e+01 8.178e+01 ... 6.343e-02 2.369e-01 6.922e-02]]\n"
     ]
    }
   ],
   "source": [
    "print(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "4ede14e6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 1 1 0 1 0 1 1 1 1 0 1 0 0 1 1 1 1 0 1 0 1 1 1 1 0 1 0 0 1 0 1 0 1 0 0 1\n",
      " 1 0 1 1 0 1 0 0 1 1 1 0 1 1 0 1 1 1 1 0 1 1 0 0 1 1 1 1 0 1 1 1 0 1 0 1 1\n",
      " 1 1 0 1 1 0 0 1 0 0 1 1 1 1 1 0 1 1 0 1 1 1 1 0 1 1 1 1 1 1 0 1 1 1 0 0 1\n",
      " 0 1 0 1 1 1 1 1 1 0 1 0 1 1 1 0 0 0 1 1 1 1 1 1 1 0 1 1 1 1 0 0 0 0 1 0 1\n",
      " 1 0 0 1 1 1 0 1 1 1 1 0 1 1 0 0 0 1 1 0 1 1 1 1 1 1 1 0 0 0 0 0 1 1 0 1 1\n",
      " 0 1 0 0 0 1 1 1 0 0 1 0 0 1 1 0 1 0 0 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 0 1 1\n",
      " 1 0 1 0 1 0 1 1 0 1 0 0 1 0 1 0 1 1 1 1 1 1 0 1 1 1 0 1 1 0 1 1 1 1 1 1 0\n",
      " 1 0 1 1 1 1 0 0 1 1 1 1 0 0 0 1 1 0 0 1 0 1 1 1 0 0 1 1 1 0 1 0 0 1 1 1 1\n",
      " 0 1 0 0 0 1 0 0 0 1 1 1 1 0 1 0 0 0 1 0 1 1 0 0 1 0 0 1 0 1 0 1 0 1 1 0 1\n",
      " 1 1 1 1 1 1 1 1 1 0 1 0 0 0 0 1 0 1 1 1 1 0 1 0 1 1 0 1 1 0 1 0 1 1 1 0 1\n",
      " 1 1 0 1 0 0 0 0 0 1 0 0 1 1 1 0 0 0 1 1 0 1 1 1 1 1 1 1 1 0 0 1 0 1 0 0 0\n",
      " 1 0 1 0 0 1 0 1 0 1 0 1 1 1 1 1 0 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n",
      " 1 1 1 0 1 0 1 0 1 0 1]\n"
     ]
    }
   ],
   "source": [
    "print(y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "8e22a36f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1.167e+01 2.002e+01 7.521e+01 ... 8.120e-02 3.206e-01 8.950e-02]\n",
      " [1.080e+01 9.710e+00 6.877e+01 ... 4.603e-02 2.090e-01 7.699e-02]\n",
      " [1.245e+01 1.641e+01 8.285e+01 ... 1.342e-01 3.231e-01 1.034e-01]\n",
      " ...\n",
      " [1.327e+01 1.476e+01 8.474e+01 ... 1.001e-01 2.027e-01 6.206e-02]\n",
      " [1.343e+01 1.963e+01 8.584e+01 ... 1.160e-01 2.884e-01 7.371e-02]\n",
      " [2.742e+01 2.627e+01 1.869e+02 ... 2.625e-01 2.641e-01 7.427e-02]]\n"
     ]
    }
   ],
   "source": [
    "print(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "80f33f01",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 1 1 1 1 1 0 1 0 0 0 1 1 1 1 0 1 1 1 0 1 0 0 0 0 1 0 1 1 1 1 1 0 1 1 1 1\n",
      " 0 0 1 0 1 0 1 1 1 1 0 0 1 1 1 1 0 0 1 1 1 1 0 1 1 1 1 1 0 0 1 1 0 1 0 1 0\n",
      " 1 1 1 0 1 0 1 1 1 0 0 0 0 0 1 0 1 0 1 0 0 1 1 1 1 1 0 0 1 1 0 1 1 0 0 1 0\n",
      " 1 0 0]\n"
     ]
    }
   ],
   "source": [
    "print(y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "47674dac",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 1 1 1 1 1 1 1 0 0 0 1 1 0 1 0 1 1 1 0 1 0 0 0 0 1 0 1 1 1 1 1 0 1 1 1 1\n",
      " 0 1 1 0 1 0 1 1 1 1 1 0 1 0 1 1 0 0 1 1 1 1 0 1 1 1 1 1 0 0 1 1 1 1 0 1 0\n",
      " 1 1 1 0 1 0 1 1 1 0 0 0 0 0 1 1 1 1 1 0 0 1 1 1 1 1 0 0 1 1 0 1 1 0 0 1 0\n",
      " 1 0 0]\n"
     ]
    }
   ],
   "source": [
    "clf = RandomForest(n_trees=20)\n",
    "clf.fit(X_train, y_train)\n",
    "predictions = clf.predict(X_test)\n",
    "print(predictions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "b1c2c820",
   "metadata": {},
   "outputs": [],
   "source": [
    "def accuracy(y_true, y_pred):\n",
    "    accuracy = np.sum(y_true == y_pred) / len(y_true)\n",
    "    return accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "900b95a8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9298245614035088\n"
     ]
    }
   ],
   "source": [
    "acc =  accuracy(y_test, predictions)\n",
    "print(acc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dfca3ebe",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
