import unittest
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.utils._param_validation import InvalidParameterError

from fair_kmeans import FairKMeans


def manual_transform(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.sqrt(((X - center) ** 2).sum(axis=1)) for center in centers], axis=-1
    )


def calculate_costs(
    points: np.ndarray, centers: np.ndarray, weights: np.ndarray, labels: np.ndarray
) -> float:
    cost = 0
    for i, p in enumerate(points):
        c = centers[labels[i]]
        cost += weights[i] * ((p - c) ** 2).sum()
    return cost


def check_fairness(
    labels: np.ndarray, colors: np.ndarray, weights: np.ndarray = None
) -> None:
    if weights is None:
        weights = np.ones(len(labels))

    color_count = np.zeros((max(colors) + 1, max(labels) + 1))

    for i, c in enumerate(colors):
        color_count[c, labels[i]] += weights[i]

    assert (
        color_count == color_count[0]
    ).all(), f"Some labels are not fair: {color_count}"


def assert_equals_computed(
    km: FairKMeans,
    data: np.ndarray,
    labels: np.ndarray,
    colors: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> None:
    if weights is None:
        weights = np.ones(len(data))
    cost = calculate_costs(np.array(data), km.cluster_centers_, weights, labels)
    assert km.inertia_ is not None and np.isclose(
        km.inertia_, cost
    ), f"Inertia: {km.inertia_} vs. cost {cost}"

    score = km.score(data, color=colors)
    assert np.isclose(km.inertia_, -score), f"Inertia: {km.inertia_} vs. score {-score}"

    pred_labels = km.predict(data, color=colors)
    assert all(pred_labels == labels)


class TestFairKMeans(unittest.TestCase):

    example_data = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.1, 1.1, 1.1],
            [1.2, 1.2, 1.2],
            [2.0, 2.0, 2.0],
            [2.1, 2.1, 2.1],
            [2.2, 2.2, 2.2],
        ]
    )
    example_colors = np.array([1, 1, 1, 0, 0, 0])

    def test_n_clusters(self) -> None:
        km = FairKMeans(n_clusters=0)

        self.assertRaises(InvalidParameterError, km.fit, self.example_data)

    def test_fit(self) -> None:

        km = FairKMeans(n_clusters=2, random_state=0)
        km.fit(self.example_data, color=self.example_colors)

        assert_equals_computed(
            km, self.example_data, km.labels_, self.example_colors, weights=None
        )

        check_fairness(km.labels_, self.example_colors)

    def test_fast_fit(self) -> None:
        km = FairKMeans(n_clusters=2, random_state=0)
        km.fit(self.example_data, color=self.example_colors, fast=True)

        assert_equals_computed(
            km, self.example_data, km.labels_, self.example_colors, weights=None
        )

        check_fairness(km.labels_, self.example_colors)

    def test_fit_predict(self) -> None:
        km = FairKMeans(n_clusters=2, random_state=0)
        labels = km.fit_predict(self.example_data, color=self.example_colors)
        assert all(labels == [1, 0, 0, 1, 0, 0])

        check_fairness(labels, self.example_colors)

    def test_fit_transform(self) -> None:
        km = FairKMeans(n_clusters=2, random_state=0)
        km.set_output(transform="pandas")
        transformed = km.fit_transform(self.example_data, color=self.example_colors)
        transformed_manual = manual_transform(self.example_data, km.cluster_centers_)

        check_fairness(km.labels_, self.example_colors)

        assert np.allclose(transformed, transformed_manual)

    def test_score(self) -> None:
        km = FairKMeans(n_clusters=2, random_state=0, max_iter=2)
        km.fit(self.example_data, color=self.example_colors)
        score = km.score(self.example_data, color=self.example_colors)

        assert score is not None and np.isclose(score, -4.53)
        new_arr = np.random.rand(75, 3)
        weights = [1] * 50 + [2] * 25
        colors = [0] * 50 + [1] * 25

        score = km.score(new_arr, sample_weight=weights, color=colors)
        labels = km.predict(new_arr, sample_weight=weights, color=colors)
        cost = calculate_costs(new_arr, km.cluster_centers_, weights, labels)

        check_fairness(labels, colors, weights)

        assert np.isclose(-score, cost), f"Score: {-score}, Cost: {cost}"

    def test_transform(self) -> None:
        km = FairKMeans(n_clusters=2, random_state=0)
        km.fit(self.example_data, color=self.example_colors)

        transformed = km.transform(self.example_data)

        transformed_manual = manual_transform(self.example_data, km.cluster_centers_)
        assert np.allclose(transformed, transformed_manual)

    def test_dataframe(self) -> None:
        feature_names = ["A", "B", "C"]
        data = pd.DataFrame(self.example_data, columns=feature_names)

        km = FairKMeans(n_clusters=2, random_state=0)
        km.fit(data, color=self.example_colors)

        assert_equals_computed(km, data, km.labels_, self.example_colors, weights=None)

        assert km.feature_names_in_ is not None and all(
            km.feature_names_in_ == feature_names
        )

        _ = km.get_feature_names_out()

        _ = km.get_params()

        check_fairness(km.labels_, self.example_colors)

    def test_datasets(self) -> None:
        for dataset_path in sorted(Path("tests/datasets").glob("*.data")):
            data = pd.read_csv(dataset_path, header=None, skiprows=1)
            array = data.iloc[:, 2:].values
            colors = data.iloc[:, 1].values
            weights = data.iloc[:, 0].values.astype(int)
            del data
            dataset_name, k = dataset_path.name.split(".")[0].split("_")
            with (dataset_path.parent / f"{dataset_name}_{k}.output").open() as f:
                c_obj = float(f.readline())
                c_iters = int(f.readline())
                c_fair = bool(f.readline())
            assert (
                c_fair
            ), f"Dataset {dataset_path.name} is not fair in C++ implementation."
            c_centers = pd.read_csv(
                dataset_path.parent / f"{dataset_name}_{k}.output",
                header=None,
                skiprows=3,
            )
            with self.subTest(msg=f"{dataset_name}_k={k}"):
                km = FairKMeans(n_clusters=int(k), random_state=0, tol=0)
                km.fit(array, color=colors, sample_weight=weights)

                is_close = np.isclose(c_centers, km.cluster_centers_)

                assert c_iters == km.n_iter_, f"Iterations: {km.n_iter_} vs. {c_iters}"

                assert np.isclose(
                    c_obj, km.inertia_
                ), f"Cost: {km.inertia_} vs. {c_obj}"

                assert is_close.all(), km.cluster_centers_

                check_fairness(km.labels_, colors, weights)


if __name__ == "__main__":
    unittest.main()
