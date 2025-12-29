import xgboost as xgb

class XGBoostClassifier:
    def __init__(self, params=None):
        """
        Inicjalizacja klasyfikatora XGBoost.

        Args:
            params (dict): Hiperparametry modelu. Jeśli None, używa domyślnych.
        """
        if params is None:
            self.params = {
                'objective': 'binary:logistic',  # Cel: klasyfikacja binarna
                'eval_metric': 'logloss',       # Metoda oceny
                'n_estimators': 100,            # Liczba drzew
                'max_depth': 6,                 # Maksymalna głębokość drzewa
                'learning_rate': 0.1,           # Tempo uczenia
                'use_label_encoder': False      # Uniknięcie ostrzeżeń w nowszych wersjach
            }
        else:
            self.params = params

        self.model = xgb.XGBClassifier(**self.params)

    def fit(self, X_train, y_train):
        """
        Trenuje model na danych.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Zwraca przewidywane klasy (0 lub 1).
        """
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """
        Zwraca prawdopodobieństwo przynależności do klas.
        """
        return self.model.predict_proba(X_test)[:, 1]