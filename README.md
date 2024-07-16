[![Build Status](https://github.com/algo-hhu/fair-kmeans/actions/workflows/mypy-flake-test.yml/badge.svg)](https://github.com/algo-hhu/fair-kmeans/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Supported Python version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Stable Version](https://img.shields.io/pypi/v/fair-kmeans?label=stable)](https://pypi.org/project/fair-kmeans/)

# Fair K-Means

Fair K-Means produces a fair clustering assignment according to the fairness definition of Chierichetti et al. [1]. Each point has a binary color assigned to it. The goal is to assign the points to clusters such that the number of points with different colors in each cluster is the same. The algorithm also works with weights, so each point can participate with a different weight in the coloring.

The algorithm works as follows, assuming that the binary colors are red and blue:
1. A matching between the red and blue points is computed such that the cost (the point distances) of the matching is minimized.
2. The mean of each matched pair is computed.
3. A K-Means++ clustering of all the means is computed, and the point pairs are assigned to the clusters of their means.

The matching between the red and blue points is computed using the [Lemon C++ Library](https://lemon.cs.elte.hu/trac/lemon). The library is included in the package and does not need to be installed separately. Only the needed files were included, and a complete version of the library can be found [here](https://lemon.cs.elte.hu/trac/lemon). A copyright notice is included [here](fair_kmeans/lemon/LICENSE).

### References

[1] Flavio Chierichetti, Ravi Kumar, Silvio Lattanzi, and Sergei Vassilvitskii, Fair clus-
tering through fairlets, Proceedings of the 30th Annual Conference on Neural Infor-
mation Processing Systems (NIPS), 2017, pp. 5036–5044.

## Installation

```bash
pip install fair-kmeans
```

## Example

```python
from fair_kmeans import FairKMeans

example_data = [
    [1.0, 1.0, 1.0],
    [1.1, 1.1, 1.1],
    [1.2, 1.2, 1.2],
    [2.0, 2.0, 2.0],
    [2.1, 2.1, 2.1],
    [2.2, 2.2, 2.2],
]

example_colors = [1, 1, 1, 0, 0, 0]

km = FairKMeans(n_clusters=2, random_state=0)
km.fit(example_data, color=example_colors)
labels = km.labels_
centers = km.cluster_centers_

print(labels) # [1, 0, 0, 1, 0, 0]
print(centers) # [[1.65, 1.65, 1.65], [1.5, 1.5, 1.5]]
```

## Example with Weights

```python
from fair_kmeans import FairKMeans

example_data = [
    [1.0, 1.0, 1.0],
    [1.1, 1.1, 1.1],
    [1.2, 1.2, 1.2],
    [2.0, 2.0, 2.0],
    [2.1, 2.1, 2.1],
    [2.2, 2.2, 2.2],
]

example_colors = [1, 1, 1, 0, 0, 0]
example_weights = [2, 2, 1, 1, 1, 3]

km = FairKMeans(n_clusters=2, random_state=0)
km.fit(example_data, color=example_colors, sample_weight=example_weights)
labels = km.labels_
centers = km.cluster_centers_

print(labels) # [1, 1, 0, 1, 1, 0]
print(centers) # [[0.85, 0.85, 0.85], [1.28, 1.28, 1.28]]
```

## Development

Install [poetry](https://python-poetry.org/docs/#installation)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Install clang
```bash
sudo apt-get install clang
```

Set clang variables
```bash
export CXX=/usr/bin/clang++
export CC=/usr/bin/clang
```

Install the package
```bash
poetry install
```

If the installation does not work and you do not see the C++ output, you can build the package to see the stack trace
```bash
poetry build
```

Run the tests
```bash
poetry run python -m unittest discover tests -v
```

## Citation

If you use this code, please cite [the following paper](https://arxiv.org/abs/2406.02739v1):

```
Melanie Schmidt, Chris Schwiegelshohn and Christian Sohler.
"Fair Coresets and Streaming Algorithms for Fair k-Means Clustering" (2020).
Approximation and Online Algorithms. WAOA 2019.
Lecture Notes in Computer Science, vol 11926. Springer.
```
