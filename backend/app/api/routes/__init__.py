from fastapi import APIRouter
from .classification import router as classification_router
from .regression import router as regression_router
from .time_series import router as time_series_router
from .anomaly_detection import router as anomaly_detection_router
from .image_classification import router as image_classification_router
from .auth import router as auth_router

router = APIRouter()

# Add the auth router
router.include_router(auth_router, prefix="/auth", tags=["auth"])

# Make sure the classification router is correctly registered with the prefix
router.include_router(classification_router, prefix="/numerical-classifier", tags=["numerical-classifier"])

# Make sure the regression router is correctly registered with the prefix
router.include_router(regression_router, prefix="/target-prediction", tags=["target-prediction"])

# Add the time series router with the time-series prefix
router.include_router(time_series_router, prefix="/time-series", tags=["time-series"])

# Add the anomaly detection router with the anomaly-detection prefix
router.include_router(anomaly_detection_router, prefix="/anomaly-detection", tags=["anomaly-detection"])

# Add the anomaly detection router with the image-classification prefix
router.include_router(image_classification_router, prefix="/image-classification", tags=["image-classification"])
