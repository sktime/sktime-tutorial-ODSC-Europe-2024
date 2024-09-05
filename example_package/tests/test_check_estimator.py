import pytest
from ensemble_forecaster import SimpleEnsembleForecaster
from sktime.utils.estimator_checks import (check_estimator,
                                           parametrize_with_checks)


@parametrize_with_checks(SimpleEnsembleForecaster)
def test_sktime_api_compliance(obj, test_name):
    check_estimator(obj, tests_to_run=test_name, raise_exceptions=True)
