# Unsupervised machine learning

## Supervised vs. Unsupervised Learning

Supervised learning involves training a model using labeled input and output data. Since the model is provided with
a labeled training set, it knows the correct output for each example. This allows the algorithm to learn and
generalize patterns, which it can then apply to new, unseen data. With a labeled dataset, the model can measure
its accuracy and improve over time by adjusting its predictions.

In contrast, unsupervised learning does not rely on labeled data. The machine learning algorithm is not given any
predefined labels; instead, it identifies hidden patterns and structures in the data without human intervention.
These algorithms do not make predictions; they simply group similar data points together.


Supervised Learning Examples:
* Classification.
* Regression.

Unsupervised Learning Examples:
* Clustering. 
* Association.
* Dimensionality reduction.

## Task description

* Using the UCI Mushroom dataset, use k-means and a suitable cluster evaluation metric to determine the optimal number of clusters in the dataset. Note that this may not necessarily be two (edible versus not-edible).
* Plot this metric while increasing the number of clusters, e.g., $k=2..30$.
* Visualise the data using the number of clusters and a suitable projection or low-dimensional embedding.

