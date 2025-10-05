from typing import Any, Dict, Annotated
from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import uvicorn
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")  # keeps "/" serving a template if needed [optional]

# -------- Model load + feature order discovery --------
model = None
feature_columns = None

def discover_feature_columns(trained_model):
    # Prefer scikit-learn feature_names_in_; fall back to common alternatives; else None
    if hasattr(trained_model, "feature_names_in_"):
        return list(getattr(trained_model, "feature_names_in_"))
    if hasattr(trained_model, "feature_name_"):
        try:
            return list(getattr(trained_model, "feature_name_"))
        except Exception:
            pass
    if hasattr(trained_model, "feature_names"):
        try:
            return list(getattr(trained_model, "feature_names"))
        except Exception:
            pass
    return None

try:
    model = joblib.load("exoplanet_model.pkl")
    logger.info("✅ Model loaded successfully")
    feature_columns = discover_feature_columns(model)
    if feature_columns is not None:
        logger.info(f"✅ Model expects {len(feature_columns)} features (from model metadata)")
    else:
        logger.info("ℹ️ Model did not expose feature names; using request column order")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    logger.error(traceback.format_exc())

# -------- Optional: serve the HTML form at "/" --------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})  # current TemplateResponse usage [web:2]

# -------- Helpers --------
def require_and_cast(body: Dict[str, Any], key: str, caster, default=None):
    if key not in body:
        if default is not None:
            return default
        raise ValueError(f"Missing required field: {key}")
    try:
        return caster(body[key])
    except Exception:
        raise ValueError(f"Invalid value for {key}: {body[key]}")

# Dict body annotated with Body + OpenAPI examples so Swagger UI shows a POST body with ready samples
PredictBody = Annotated[
    Dict[str, Any],
    Body(
        openapi_examples={
            "candidate_like": {
                "summary": "Likely CANDIDATE",
                "value": {
                    "koi_pdisposition": "CANDIDATE",
                    "koi_score": 0.95,
                    "koi_fpflag_nt": 0,
                    "koi_fpflag_ss": 0,
                    "koi_fpflag_co": 0,
                    "koi_fpflag_ec": 0,
                    "koi_period": 13.825,
                    "koi_time0bk": 131.512,
                    "koi_impact": 0.146,
                    "koi_duration": 2.675,
                    "koi_depth": 489,
                    "koi_prad": 1.88,
                    "koi_teq": 793,
                    "koi_insol": 36.41,
                    "koi_model_snr": 26.3,
                    "koi_tce_plnt_num": 1,
                    "koi_tce_delivname": "q1_q16_tce",
                    "koi_steff": 5780,
                    "koi_slogg": 4.438,
                    "koi_srad": 0.927,
                    "ra": 291.93423,
                    "dec": 48.14109,
                    "koi_kepmag": 14.75
                }
            },
            "false_positive_like": {
                "summary": "Likely FALSE POSITIVE",
                "value": {
                    "koi_pdisposition": "FALSE POSITIVE",
                    "koi_score": 0.05,
                    "koi_fpflag_nt": 1,
                    "koi_fpflag_ss": 1,
                    "koi_fpflag_co": 1,
                    "koi_fpflag_ec": 1,
                    "koi_period": 1.2,
                    "koi_time0bk": 100.0,
                    "koi_impact": 0.9,
                    "koi_duration": 0.5,
                    "koi_depth": 50,
                    "koi_prad": 0.5,
                    "koi_teq": 1500,
                    "koi_insol": 200.0,
                    "koi_model_snr": 2.5,
                    "koi_tce_plnt_num": 1,
                    "koi_tce_delivname": "q1_q12_tce",
                    "koi_steff": 6200,
                    "koi_slogg": 3.8,
                    "koi_srad": 2.0,
                    "ra": 10.0,
                    "dec": -5.0,
                    "koi_kepmag": 18.5
                }
            },
            "confirmed_like": {
                "summary": "Possibly CONFIRMED (depends on trained classes)",
                "value": {
                    "koi_pdisposition": "CANDIDATE",
                    "koi_score": 0.99,
                    "koi_fpflag_nt": 0,
                    "koi_fpflag_ss": 0,
                    "koi_fpflag_co": 0,
                    "koi_fpflag_ec": 0,
                    "koi_period": 10.0,
                    "koi_time0bk": 120.0,
                    "koi_impact": 0.1,
                    "koi_duration": 4.0,
                    "koi_depth": 1500,
                    "koi_prad": 2.0,
                    "koi_teq": 600,
                    "koi_insol": 10.0,
                    "koi_model_snr": 80.0,
                    "koi_tce_plnt_num": 1,
                    "koi_tce_delivname": "q1_q17_dr25_tce",
                    "koi_steff": 5600,
                    "koi_slogg": 4.4,
                    "koi_srad": 1.0,
                    "ra": 250.0,
                    "dec": 30.0,
                    "koi_kepmag": 12.0
                }
            }
        }
    )
]

# -------- Predict endpoint: returns ONLY koi_disposition --------
@app.post("/predict")
async def predict(payload: PredictBody):
    try:
        if model is None:
            return JSONResponse(status_code=500, content={"error": "Model not loaded"})

        body = payload  # dict provided by Body annotation; shows in Swagger with examples
        # Encode categoricals
        pdisposition_mapping = {'CANDIDATE': 1, 'FALSE POSITIVE': 0}
        delivname_mapping = {
            'q1_q12_tce': 0,
            'q1_q16_tce': 1,
            'q1_q17_dr24_tce': 2,
            'q1_q17_dr25_tce': 3
        }
        koi_pdisposition = int(pdisposition_mapping.get(require_and_cast(body, "koi_pdisposition", str), 0))
        koi_tce_delivname = int(delivname_mapping.get(require_and_cast(body, "koi_tce_delivname", str), 1))

        # Numeric features
        data = pd.DataFrame([{
            'koi_pdisposition': koi_pdisposition,
            'koi_score': float(require_and_cast(body, "koi_score", float)),
            'koi_fpflag_nt': int(require_and_cast(body, "koi_fpflag_nt", int)),
            'koi_fpflag_ss': int(require_and_cast(body, "koi_fpflag_ss", int)),
            'koi_fpflag_co': int(require_and_cast(body, "koi_fpflag_co", int)),
            'koi_fpflag_ec': int(require_and_cast(body, "koi_fpflag_ec", int)),
            'koi_period': float(require_and_cast(body, "koi_period", float)),
            'koi_time0bk': float(require_and_cast(body, "koi_time0bk", float)),
            'koi_impact': float(require_and_cast(body, "koi_impact", float)),
            'koi_duration': float(require_and_cast(body, "koi_duration", float)),
            'koi_depth': float(require_and_cast(body, "koi_depth", float)),
            'koi_prad': float(require_and_cast(body, "koi_prad", float)),
            'koi_teq': float(require_and_cast(body, "koi_teq", float)),
            'koi_insol': float(require_and_cast(body, "koi_insol", float)),
            'koi_model_snr': float(require_and_cast(body, "koi_model_snr", float)),
            'koi_tce_plnt_num': int(require_and_cast(body, "koi_tce_plnt_num", int)),
            'koi_tce_delivname': koi_tce_delivname,
            'koi_steff': float(require_and_cast(body, "koi_steff", float)),
            'koi_slogg': float(require_and_cast(body, "koi_slogg", float)),
            'koi_srad': float(require_and_cast(body, "koi_srad", float)),
            'ra': float(require_and_cast(body, "ra", float)),
            'dec': float(require_and_cast(body, "dec", float)),
            'koi_kepmag': float(require_and_cast(body, "koi_kepmag", float))
        }])

        # Align to model's expected order if available
        if feature_columns is not None:
            data = data.reindex(columns=feature_columns, fill_value=0.0)

        # Predict discrete label (no probabilities)
        pred_raw = model.predict(data)[0]  # scikit-learn predict returns class labels [web:65][web:73]

        # Map numeric/encoded label to final disposition text
        prediction_mapping = {0: 'FALSE POSITIVE', 1: 'CANDIDATE', 2: 'CONFIRMED'}
        koi_disposition = prediction_mapping.get(pred_raw, str(pred_raw))

        return JSONResponse(content={"koi_disposition": koi_disposition})
    except ValueError as ve:
        return JSONResponse(status_code=400, content={"error": str(ve)})
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
