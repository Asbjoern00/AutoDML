from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

class CovariateExpander:
    def __init__(self, spline_degree=3, monomial_degree=2):
        self.spline_degree = spline_degree
        self.poly = PolynomialFeatures(degree=monomial_degree, include_bias=False, interaction_only=True)
        self.continuous_indices = None
        self.passthrough_indices = None
        self.preprocessor = None
        self.purge_cols = None

    @staticmethod
    def _is_numeric_binary(col):
        unique = np.unique(col[~np.isnan(col)])
        return len(unique) <= 2 and np.all(np.isclose(unique, unique.astype(int)))

    def fit(self, x):
        x = np.asarray(x)

        # ---- Step 1: Polynomial expansion ----
        x_poly = self.poly.fit_transform(x)
        self.purge_cols = np.isclose(np.std(x_poly, axis=0), 0)  # Remove columns without variability
        x_poly = x_poly[:, ~self.purge_cols]

        # ---- Step 2: Identify column types ----
        continuous = []
        passthrough = []

        for i in range(x_poly.shape[1]):
            if self._is_numeric_binary(x_poly[:, i]):
                passthrough.append(i)
            else:
                continuous.append(i)

        self.continuous_indices = continuous
        self.passthrough_indices = passthrough

        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "spline",
                    Pipeline(
                        [
                            ("spline", SplineTransformer(n_knots=3, degree=self.spline_degree, include_bias=False)),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    continuous,
                ),
                ("passthrough", Pipeline([("scaler", StandardScaler())]), passthrough),
            ]
        )

        self.preprocessor.fit(x_poly)
        return self

    def transform(self, x):
        x = np.asarray(x)
        x_poly = self.poly.transform(x)
        x_poly = x_poly[:, ~self.purge_cols]
        x_splined = self.preprocessor.transform(x_poly)
        x_out = np.concatenate([x_splined, np.ones((x_poly.shape[0], 1))], axis=1)

        return x_out
