import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.api.inference import analyze_session
from src.api.policy import load_artifacts


logger = logging.getLogger(__name__)

app = FastAPI(
    title="RHYVA",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://rhyva.onrender.com",
    ],

    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class Measurement(BaseModel):
    second: float
    heartRate: Optional[float] = None
    respiratoryRate: Optional[float] = None


class SessionRequest(BaseModel):
    sport: str = Field(
        min_length=1,
        max_length=20,
    )

    initialRestConfirmed: bool = False

    initialRestSeconds: int = Field(
        default=0,
        ge=0,
        le=10000,
    )

    samples: List[Measurement] = Field(
        min_length=2,
        max_length=10000,
    )


@app.get("/health")
def health():
    metadata, _, _, _ = load_artifacts()

    return {
        "status": "ok",
        "model": metadata["baseline_tag"],
    }


@app.post("/api/analyze")
def analyze(request: SessionRequest):
    try:
        return analyze_session(
            seconds=[
                s.second
                for s in request.samples
            ],

            heart_rate=[
                s.heartRate
                for s in request.samples
            ],

            respiratory_rate=[
                s.respiratoryRate
                for s in request.samples
            ],

            sport=request.sport.strip().upper(),

            initial_rest_confirmed=(
                request.initialRestConfirmed
            ),

            initial_rest_seconds=(
                request.initialRestSeconds
            ),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception(
            "Erreur pendant l'analyse RHYVA"
        )

        raise HTTPException(
            status_code=500,
            detail="Analyse indisponible.",
        ) from error
