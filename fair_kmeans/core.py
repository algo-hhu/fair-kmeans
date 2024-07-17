import ctypes
from numbers import Integral, Real
from time import time
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn._config import get_config
from sklearn.cluster import KMeans
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    check_random_state,
)

import fair_kmeans._core  # type: ignore

_DLL = ctypes.cdll.LoadLibrary(fair_kmeans._core.__file__)


class FairKMeans(KMeans):

    _parameter_constraints: dict = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "init": [StrOptions({"k-means++", "random"})],
        "n_init": [
            StrOptions({"auto"}),
            Interval(Integral, 1, None, closed="left"),
        ],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_clusters: int,
        init: str = "k-means++",
        n_init: Union[int, str] = "auto",
        max_iter: int = 100,
        tol: float = 1e-4,
        verbose: int = 0,
        random_state: Optional[int] = None,
    ):
        self._seed = int(time()) if random_state is None else random_state
        if not isinstance(init, str):
            raise NotImplementedError(
                "In scikit-learn, the initialization can also be done "
                "with an array-like object. "
                "This will be implemented in future versions."
            )

        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=check_random_state(self._seed),
        )

    def check_metatadata_routing(self) -> None:
        if get_config().get("enable_metadata_routing", False):
            raise NotImplementedError(
                "FairKMeans has not yet been tested with metadata routing."
            )

    def _warn_mkl_vcomp(self, n_active_threads: int) -> None:
        """Warn when vcomp and mkl are both present"""
        raise NotImplementedError(
            "scikit-learn is known to have a memory leak on Windows "
            "with MKL, when there are less chunks than available "
            "threads. You can avoid it by setting the environment"
            f" variable OMP_NUM_THREADS={n_active_threads}. "
            "This has not been tested with FairKMeans yet, and "
            "we do not know if it will cause the same problems."
        )

    def _check_X(self, X: Sequence[Sequence[float]]) -> Any:
        _X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
            copy=False,
        )

        self._check_params_vs_input(_X)

        return _X

    def _check_color(
        self, X: Sequence[Sequence[float]], color: Optional[Sequence[int]]
    ) -> np.ndarray:
        if color is None:
            raise ValueError(
                "Please provide a list or array of colors for each point. "
                "In total, at most two colors are allowed."
            )

        if len(color) != len(X):
            raise ValueError(
                "Length of colors array should be equal to the number of samples in X."
            )

        self.n_colors_ = len(np.unique(color))

        if self.n_colors_ > 2:
            raise NotImplementedError("For now, FairKMeans only supports two colors.")

        return np.array(color, dtype=np.int32, order="C", copy=False)

    def _check_sample_weight_constraints(
        self, X: Sequence[Sequence[float]], sample_weight: Optional[Sequence[int]]
    ) -> Any:
        # For now, we only accept integers, because the code was
        # only tested with integer weights
        if sample_weight is not None and any(
            isinstance(w, float) for w in sample_weight
        ):
            raise NotImplementedError(
                "For now, FairKMeans only supports integer weights greater than 1."
            )

        return _check_sample_weight(sample_weight, X, only_non_negative=True)

    def _run_fair_clustering(
        self,
        X: np.ndarray,
        sample_weight: np.ndarray,
        color: np.ndarray,
        update_centers: bool = False,
    ) -> Tuple[float, np.ndarray]:
        n_samples = X.shape[0]

        # Declare c types
        c_array = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_sample_weight = sample_weight.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_color = color.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

        c_n = ctypes.c_uint(n_samples)
        c_d = ctypes.c_uint(self.n_features_in_)
        c_k = ctypes.c_uint(self.n_clusters)
        c_n_colors = ctypes.c_uint(self.n_colors_)

        c_random_state = ctypes.c_size_t(self._seed)

        c_labels = (ctypes.c_int * n_samples)()
        c_centers = self.cluster_centers_.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        )

        # If we want to update the centers, we do many iterations (max_iter)
        if update_centers:
            c_max_iter = ctypes.c_uint(self.max_iter)
        else:
            c_max_iter = ctypes.c_uint(1)

        c_tolerance = ctypes.c_double(self._tol)
        c_update_centers = ctypes.c_bool(update_centers)
        c_iter = ctypes.c_uint()

        _DLL.fairKMeans.restype = ctypes.c_double
        cost = _DLL.fairKMeans(
            c_array,
            c_sample_weight,
            c_color,
            c_n,
            c_d,
            c_k,
            c_n_colors,
            c_max_iter,
            c_tolerance,
            c_random_state,
            c_labels,
            c_centers,
            c_update_centers,
            ctypes.byref(c_iter),
        )

        # In case it ran already
        if not hasattr(self, "n_iter_"):
            self.n_iter_ = 0

        self.n_iter_ += c_iter.value

        if cost == -1:
            raise ValueError("The weights of the colors are not balanced.")
        elif cost == -2:
            raise ValueError("The produced assignment is not fair.")
        elif cost == -3:
            raise ValueError("The CapacityScaling algorithm was infeasible.")
        elif cost == -4:
            raise ValueError("The CapacityScaling algorithm was unbounded.")

        labels = np.ctypeslib.as_array(c_labels)

        return cost, labels

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Any = None,
        sample_weight: Optional[Sequence[int]] = None,
        color: Optional[Sequence[int]] = None,
        fast: bool = False,
    ) -> "FairKMeans":
        self._validate_params()
        self._check_feature_names(X, reset=True)

        _X = self._check_X(X)
        _sample_weight = self._check_sample_weight_constraints(_X, sample_weight)
        _color = self._check_color(_X, color)

        if fast:
            super().fit(_X, y, _sample_weight)
        else:
            self._n_threads = _openmp_effective_n_threads()
            self.n_features_in_ = _X.shape[1]

            # precompute squared norms of data points
            x_squared_norms = row_norms(X, squared=True)

            self.cluster_centers_ = self._init_centroids(
                _X,
                x_squared_norms=x_squared_norms,
                init=self.init,
                sample_weight=_sample_weight,
                random_state=self.random_state,
            )

        # Overwrite the tolerance to be the one given by the user
        # In sklearn it is based on the variance of the data
        self._tol = self.tol

        cost, labels = self._run_fair_clustering(
            _X, _sample_weight, _color, update_centers=True
        )

        self.inertia_ = cost

        self.labels_ = labels

        self._n_features_out = len(self.cluster_centers_)

        return self

    def fit_predict(
        self,
        X: Sequence[Sequence[float]],
        y: Any = None,
        sample_weight: Optional[Sequence[int]] = None,
        color: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        return self.fit(X=X, y=y, sample_weight=sample_weight, color=color).labels_

    def fit_transform(
        self,
        X: Sequence[Sequence[float]],
        y: Any = None,
        sample_weight: Optional[Sequence[int]] = None,
        color: Optional[Sequence[int]] = None,
    ) -> Any:

        self.fit(X=X, y=y, sample_weight=sample_weight, color=color)

        return self._transform(X)

    def predict(
        self,
        X: Sequence[Sequence[float]],
        y: Any = None,
        sample_weight: Optional[Sequence[int]] = None,
        color: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        check_is_fitted(self)

        _X = self._check_X(X)
        _sample_weight = self._check_sample_weight_constraints(_X, sample_weight)
        _color = self._check_color(_X, color)

        _, labels = self._run_fair_clustering(
            _X, _sample_weight, _color, update_centers=False
        )

        return labels

    def score(
        self,
        X: Sequence[Sequence[float]],
        y: Any = None,
        sample_weight: Optional[Sequence[int]] = None,
        color: Optional[Sequence[int]] = None,
    ) -> float:
        check_is_fitted(self)

        _X = self._check_X(X)
        _sample_weight = self._check_sample_weight_constraints(_X, sample_weight)
        _color = self._check_color(_X, color)

        cost, _ = self._run_fair_clustering(
            _X, _sample_weight, _color, update_centers=False
        )

        return -cost

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "labels_")
