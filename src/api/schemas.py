from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., description="Вопрос пользователя")
    top_k: int = Field(default=5, ge=1, le=20, description="Количество возвращаемых чанков")


class SourceInfo(BaseModel):
    title: str
    url: str
    relevance: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceInfo]
