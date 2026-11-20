"""Request and response shapes for the recommendation service."""

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    history: list[int] = Field(default_factory=list, max_length=500)
    k: int = Field(default=5, ge=1, le=50)


class Recommendation(BaseModel):
    rank: int
    movie_id: int
    title: str
    score: float


class RecommendResponse(BaseModel):
    recommendations: list[Recommendation]
    strategy: str
    model_version: str
    ignored_ids: list[int]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_version: str
