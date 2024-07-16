[![Build Status](https://github.com/algo-hhu/fair-kmeans/actions/workflows/mypy-flake-test.yml/badge.svg)](https://github.com/algo-hhu/fair-kmeans/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Supported Python version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Stable Version](https://img.shields.io/pypi/v/fair-kmeans?label=stable)](https://pypi.org/project/fair-kmeans/)

# Fair K-Means++



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
