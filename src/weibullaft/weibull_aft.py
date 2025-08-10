from lifelines import WeibullAFTFitter
import numpy as np
import pandas as pd

class WeibullAFTWrapper:
    """
    Sklearn-like wrapper around lifelines.WeibullAFTFitter.
    Expects y as a structured array with fields ('event', 'time') or a 2D array [time, event].
    Provides predict_survival_function(X, times) similar to sksurv estimators.
    """
    def __init__(self, penalizer=0.0, l1_ratio=0.0):
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.model = WeibullAFTFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        self.feature_names_ = None
        self.is_fitted_ = False

    def _y_to_series(self, y):
        # Accept structured array with fields ('event','time') in any order, or 2D array
        if hasattr(y, 'dtype') and y.dtype.names is not None:
            names = y.dtype.names
            if 'time' in names and 'event' in names:
                time = np.asarray(y['time'], dtype=float)
                event = np.asarray(y['event'], dtype=bool).astype(int)
            else:
                # Try common alternatives
                time_field = next((n for n in names if n.lower() in ('time','duration','t')), None)
                event_field = next((n for n in names if n.lower() in ('event','status','e','delta')), None)
                if time_field is None or event_field is None:
                    raise ValueError("Structured y must contain time and event fields.")
                time = np.asarray(y[time_field], dtype=float)
                event = np.asarray(y[event_field], dtype=bool).astype(int)
        else:
            y = np.asarray(y)
            if y.ndim != 2 or y.shape[1] != 2:
                raise ValueError("y must be structured with ('event','time') or a 2D array of shape [N,2] as [time, event].")
            time = np.asarray(y[:, 0], dtype=float)
            event = np.asarray(y[:, 1], dtype=bool).astype(int)
        return time, event

    def _Xy_to_dataframe(self, X, y):
        time, event = self._y_to_series(y)
        X = np.asarray(X)
        if self.feature_names_ is None:
            self.feature_names_ = [f"x{i}" for i in range(X.shape[1])]
        df_X = pd.DataFrame(X, columns=self.feature_names_)
        df = df_X.copy()
        df['time'] = time
        df['event'] = event
        return df

    def fit(self, X, y):
        df = self._Xy_to_dataframe(X, y)
        self.model.fit(df, duration_col='time', event_col='event')
        self.is_fitted_ = True
        return self

    def predict_survival_function(self, X, times=None):
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict_survival_function().")
        X = np.asarray(X)
        df_X = pd.DataFrame(X, columns=self.feature_names_)
        if times is None:
            # Use a reasonable default timeline based on training durations
            durations = self.model.durations.values
            t_min, t_max = np.percentile(durations, 1), np.percentile(durations, 99)
            t_min = max(t_min, 1e-6)
            times = np.linspace(t_min, t_max, 100)
        times = np.asarray(times, dtype=float)
        # lifelines returns a dataframe per row; we align outputs into a list of callables like sksurv does
        surv_dfs = self.model.predict_survival_function(df_X, times=times)
        # Return list of (times, survival_values) pairs, or a list of callables is sometimes expected; here return list of arrays
        return [np.asarray(surv_dfs.iloc[:, i]) for i in range(surv_dfs.shape[1])], times

    def predict_median(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict_median().")
        X = np.asarray(X)
        df_X = pd.DataFrame(X, columns=self.feature_names_)
        med = self.model.predict_median(df_X)
        return np.asarray(med, dtype=float)

    def score(self, X, y):
        # Optional: return negative partial log-likelihood-like metric; here we use concordance index from lifelines
        from lifelines.utils import concordance_index
        time, event = self._y_to_series(y)
        # Use median as a risk score surrogate (lower median time = higher risk)
        risk = -self.predict_median(X)
        return concordance_index(time, risk, event_observed=event)
