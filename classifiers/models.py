from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


def get_classifiers(random_state=42):
    return {
        "NB": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "DT": DecisionTreeClassifier(random_state=random_state),
        "LDA": LinearDiscriminantAnalysis()
    }