"""FastAPI Recommendation Serving Web Application."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from src.router import HybridRouter
from app.schemas import RecommendRequest, RecommendResponse, HealthResponse

router = HybridRouter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager loading router resources on startup."""
    try:
        router.load_resources()
    except Exception:
        pass
    yield


app = FastAPI(
    title="Two-Tower Hybrid Recommender API",
    description="Serving endpoint with cold-start dynamic strategy routing.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def health_check() -> HealthResponse:
    """Lightweight healthcheck endpoint for container verification."""
    return HealthResponse(status="ok")


@app.post("/api/recommend", response_model=RecommendResponse, status_code=status.HTTP_200_OK)
def recommend(request: RecommendRequest) -> RecommendResponse:
    """Generate dynamic hybrid recommendations for requesting user."""
    res = router.recommend(user_id=request.user_id, history=request.history, top_k=10)
    return RecommendResponse(
        user_id=res["user_id"],
        strategy_used=res["strategy_used"],
        recommendations=res["recommendations"]
    )

