from fastapi import APIRouter
from .classification import router as classification_router
from .regression import router as regression_router
from .time_series import router as time_series_router
from .anomaly_detection import router as anomaly_detection_router

router = APIRouter()

# Make sure the classification router is correctly registered with the prefix
router.include_router(classification_router, prefix="/numerical-classifier", tags=["numerical-classifier"])

# Make sure the regression router is correctly registered with the prefix
router.include_router(regression_router, prefix="/target-prediction", tags=["target-prediction"])

# Add the time series router with the time-series prefix
router.include_router(time_series_router, prefix="/time-series", tags=["time-series"])

# Add the anomaly detection router with the anomaly-detection prefix
router.include_router(anomaly_detection_router, prefix="/anomaly-detection", tags=["anomaly-detection"])