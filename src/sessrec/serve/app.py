"""HTTP service: five next films, or the popular list when the history is empty."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from sessrec.serve.inference import Recommender
from sessrec.serve.schemas import HealthResponse, RecommendRequest, RecommendResponse


def create_app(recommender: Recommender | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.recommender = recommender if recommender is not None else Recommender.load()
        app.state.recommender.warmup()
        yield

    app = FastAPI(title="Next-Movie Recommender: Rating-aware Self-Attention", lifespan=lifespan)

    def current(request: Request) -> Recommender:
        try:
            return request.app.state.recommender
        except AttributeError:
            loaded = recommender if recommender is not None else Recommender.load()
            request.app.state.recommender = loaded
            loaded.warmup()
            return loaded

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> dict:
        return {"status": "ok", "model_version": current(request).model_version}

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(body: RecommendRequest, request: Request) -> dict:
        return current(request).recommend(body.history, body.k)

    return app


app = create_app()
