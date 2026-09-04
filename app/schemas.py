"""Pydantic schemas for FastAPI recommendation requests and responses."""

from typing import List
from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="ID of requesting user")
    history: List[int] = Field(default_factory=list, description="List of item IDs previously interacted with")


class RecommendResponse(BaseModel):
    user_id: int = Field(..., description="ID of requesting user")
    strategy_used: str = Field(..., description="Strategy selected by dynamic router")
    recommendations: List[int] = Field(..., description="List of recommended item IDs")


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
