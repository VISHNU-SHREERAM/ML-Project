from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def run_randomised_search_svc(feat, tgt, param_grid):
    base_clf = SVC()
    rand_clf = RandomizedSearchCV(base_clf, param_distributions = param_grid)

    rand_clf.fit(feat, tgt)
    return rand_clf

def run_grid_search_svc(feat, tgt, param_grid):
    base_clf = SVC()
    grid_clf = GridSearchCV(base_clf, param_grid = param_grid, verbose = 3)

    grid_clf.fit(feat, tgt)
    return grid_clf