"""
Pydantic 스키마 정의
- 뉴스 응답
- 논문 응답
- Agent State
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime


# ========== 뉴스 관련 ==========

class Article(BaseModel):
    """뉴스 기사 모델"""
    title: str = Field(description="기사 제목")
    source: str = Field(description="출처")
    date: str = Field(description="발행일")
    summary: str = Field(description="핵심 내용 3줄 요약")
    url: Optional[str] = Field(default="", description="기사 URL")


class NewsResponse(BaseModel):
    """뉴스 Agent 응답"""
    articles: List[Article] = Field(description="수집된 뉴스 리스트")


# ========== 논문 관련 ==========

class Paper(BaseModel):
    """논문 모델"""
    title: str = Field(description="논문 제목")
    authors: List[str] = Field(description="저자 리스트")
    date: str = Field(description="발행일 (YYYY-MM-DD)")
    abstract: str = Field(description="초록 (500자 제한)")
    url: str = Field(description="PDF URL")


class PaperResponse(BaseModel):
    """논문 Agent 응답"""
    papers: List[Paper] = Field(description="검색된 논문 리스트")


# ========== Host Agent 관련 ==========

class AnalyzeRequest(BaseModel):
    """트렌드 분석 요청"""
    topic: str = Field(description="분석할 주제")
    max_news: int = Field(default=5, description="최대 뉴스 개수")
    max_papers: int = Field(default=5, description="최대 논문 개수")


class AnalyzeResponse(BaseModel):
    """트렌드 분석 응답"""
    report: str = Field(description="생성된 보고서")
    session_id: str = Field(description="세션 ID")


class RAGQueryRequest(BaseModel):
    """RAG 질의 요청"""
    query: str = Field(description="질문")
    session_id: str = Field(description="세션 ID")


class RAGQueryResponse(BaseModel):
    """RAG 질의 응답"""
    answer: str = Field(description="답변 (출처 포함)")


# ========== LangGraph State ==========

class AgentState(BaseModel):
    """LangGraph State"""
    topic: str = Field(description="분석 주제")
    news: List[Article] = Field(default_factory=list, description="수집된 뉴스")
    papers: List[Paper] = Field(default_factory=list, description="수집된 논문")
    report: str = Field(default="", description="생성된 보고서")
    session_id: str = Field(default="", description="세션 ID")
    
    class Config:
        arbitrary_types_allowed = True