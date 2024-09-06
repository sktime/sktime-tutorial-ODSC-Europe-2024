# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
Implementation of ResidualBoostingForecaster.

A forecaster that fits base forecasters to the residuals of the previous
forecasters.
"""

__author__ = ["felipeangelimvieira"]
__all__ = ["SimpleEnsembleForecaster"]

from typing import Optional

import pandas as pd
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing


class SimpleEnsembleForecaster(BaseForecaster):
    """Simple ensemble forecaster.
    
    This forecaster fits a list of forecasters to the same training data and
    aggregates their predictions using a simple aggregation function.

    Parameters
    ----------
    forecasters : list of sktime forecasters
        List of forecasters to fit and aggregate.
    agg : str, optional (default="mean")
        Aggregation function to use. Must be one of "mean" or "median".
    """

    _tags = {
        # Model metadata
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "pd.Series",
        "scitype:y": "univariate",
        # Packaging info
        "authors": ["felipeangelimvieira"],
        "maintainers": ["felipeangelimvieira"],
        "python_version": None,
        "python_dependencies": None,
    }

    # todo: add any hyper-parameters and components to constructor
    def __init__(
        self, forecasters, agg = "mean"
    ):
        # estimators should precede parameters
        #  if estimators have default values, set None and initialize below

        self.forecasters = forecasters
        self.agg = agg

        super().__init__()

        # Handle default values, being careful to not overwrite the hyper-parameters
        # as they were passed!

        # Parameter checking logic
        if agg not in ["mean", "median"]:
            raise ValueError(f"agg must be 'mean' or 'median', got {agg}")

        # if tags of estimator depend on component tags, set them
        for forecaster in self.forecasters:
            if forecaster.get_tag("requires-fh-in-fit"):
                self.set_tags({"requires-fh-in-fit": True})

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit. 
        Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of pd.Series
        fh : ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
        X : sktime time series object, optional (default=None)
            guaranteed to be pd.DataFrame, or None

        Returns
        -------
        self : reference to self
        """

        self.forecasters_ = []
        for forecaster in self.forecasters:
            forecaster = forecaster.clone()
            forecaster.fit(y, X, fh)
            self.forecasters_.append(forecaster)

    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
        X : sktime time series object, optional (default=None)
            guaranteed to be pd.DataFrame, or None

        Returns
        -------
        y_pred : sktime time series object
            should be of the same type as seen in _fit, as in "y_inner_mtype" tag
            Point predictions
        """
        y_preds = []
        for forecaster in self.forecasters_:
            y_preds.append(
                forecaster.predict(fh, X)
            )

        aggregated = pd.concat(y_preds, axis=1).agg(self.agg, axis=1)
        
        # Must keep the name of the original series
        aggregated.name = y_preds[0].name
        return aggregated
    
    def _update(self, y, X=None, update_params=True):
        """Update fitted time series forecaster.

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of pd.Series
        X : sktime time series object, optional (default=None)
            guaranteed to be pd.DataFrame, or None
        update_params : bool, optional (default=True)
            Flag whether to update hyper-parameters.

        Returns
        -------
        self : reference to self
        """
        
        for i in range(len(self.forecasters_)):
            self.forecasters_[i].update(y, X, update_params)
        

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """

        return [
            {
                "forecasters": [ExponentialSmoothing()],
                "agg": "mean",
            },
            {
                "forecasters": [ExponentialSmoothing(), ExponentialSmoothing()],
                "agg": "median",
            },
        ]
