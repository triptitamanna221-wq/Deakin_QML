from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def get_classical_baselines():
    return {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=300, class_weight="balanced"),
        "xgb": XGBClassifier(scale_pos_weight=10, eval_metric="logloss"),
        "kernel_svm": SVC(kernel="rbf", probability=True, class_weight="balanced"),
    }
