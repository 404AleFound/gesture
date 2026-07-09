from typing import Any


class Classifier:
    name: str = "Classifier"
    param_grid: dict[str, list[Any]] = {}

    def build(self, seed: int = 0) -> Any:
        """Return a fresh sklearn estimator for this classifier."""
        raise NotImplementedError

    def predict(self, estimator, X):
        return estimator.predict(X)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class DecisionTree(Classifier):
    name = "DecisionTree"
    def build(self, seed: int = 0):
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(min_samples_leaf=5, random_state=seed)


class RandomForest(Classifier):
    name = "RandomForest"
    def build(self, seed: int = 0):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=6, random_state=seed, n_jobs=-1,
        )


class NaiveBayes(Classifier):
    name = "NaiveBayes"
    def build(self, seed: int = 0):
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()


class MLP(Classifier):
    name = "MLP"
    def build(self, seed: int = 0):
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=2000,
            n_iter_no_change=50,
            tol=1e-5,
            random_state=seed,
        )


class KNN(Classifier):
    name = "kNN"
    def build(self, seed: int = 0):
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=5)


# Registry the runner iterates over. Order = display order in reports.
ALL_MODELS: list[Classifier] = [
    DecisionTree(),
    RandomForest(),
    NaiveBayes(),
    MLP(),
    KNN(),
]
