"""
뉴스 Agent (Port 10020)
- 기술 뉴스 자동 수집
- DuckDuckGo 검색
- Structured Output
"""

import os
from fastapi import FastAPI, HTTPException
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from models.schemas import NewsResponse, Article
from tools.search_tool import news_search_tool

# 환경 변수 로드
load_dotenv()

# FastAPI 앱
app = FastAPI(title="News Agent", description="기술 뉴스 수집 Agent")

# LLM 설정
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AOAI_ENDPOINT"),
    api_key=os.getenv("AOAI_API_KEY"),
    deployment_name=os.getenv("AOAI_DEPLOY_GPT4O"),
    api_version=os.getenv("AOAI_API_VERSION", "2024-08-01-preview"),
    temperature=0
)

# Structured Output
structured_llm = llm.with_structured_output(NewsResponse)

# 프롬프트
prompt = ChatPromptTemplate.from_template(
    """당신은 기술 뉴스 수집 전문가입니다.
다음 검색 결과를 분석하여 NewsResponse 형식으로 변환하세요.

검색 기준:
- 최근 1개월 이내 발행
- 신뢰할 수 있는 출처 우선
- 기술적 깊이가 있는 기사 선호

각 기사의 핵심 내용을 3줄로 요약하세요.

검색 결과:
{results}

주제: {topic}
"""
)


@app.get("/")
async def root():
    """헬스 체크"""
    return {"status": "ok", "agent": "news"}


@app.post("/search", response_model=NewsResponse)
async def search_news(topic: str, max_results: int = 5):
    """
    뉴스 검색 엔드포인트
    
    Args:
        topic: 검색 주제
        max_results: 최대 결과 개수
        
    Returns:
        NewsResponse
    """
    try:
        print(f"[뉴스 Agent] 검색 시작: {topic}")
        
        # DuckDuckGo 검색
        results = news_search_tool.search_news(topic)
        
        if not results or results == "[]":
            print("[뉴스 Agent] 검색 결과 없음")
            return NewsResponse(articles=[])
        
        # LLM으로 정리
        chain = prompt | structured_llm
        response = chain.invoke({
            "results": results,
            "topic": topic
        })
        
        print(f"[뉴스 Agent] {len(response.articles)}개 뉴스 수집 완료")
        return response
    
    except Exception as e:
        print(f"[뉴스 Agent] 오류 발생: {e}")
        # 빈 응답 반환
        return NewsResponse(articles=[])


@app.get("/.well-known/agent-card.json")
async def agent_card():
    """
    Agent Card - A2A 표준 프로토콜
    Agent의 능력과 스펙을 공개
    """
    return {
        "version": "1.0.0",
        "name": "뉴스 수집 Agent",
        "description": "기술 뉴스 검색 및 수집 전문 Agent. DuckDuckGo 기반 최신 뉴스 검색.",
        "url": "http://localhost:10020",
        "capabilities": [
            {
                "name": "search_news",
                "description": "기술 관련 최신 뉴스 검색",
                "endpoint": "/search",
                "method": "POST",
                "parameters": {
                    "topic": {
                        "type": "string",
                        "description": "검색할 기술 주제",
                        "required": True,
                        "example": "RAG"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "최대 검색 결과 개수",
                        "required": False,
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "response_schema": {
                    "type": "NewsResponse",
                    "description": "뉴스 기사 리스트",
                    "properties": {
                        "articles": {
                            "type": "array",
                            "items": {
                                "type": "Article",
                                "properties": {
                                    "title": "string",
                                    "source": "string",
                                    "date": "string",
                                    "summary": "string",
                                    "url": "string"
                                }
                            }
                        }
                    }
                }
            }
        ],
        "tags": ["news", "search", "tech", "trending"],
        "author": "Tech Trend Scout Team",
        "contact": "http://localhost:10020",
        "created_at": "2025-01-13",
        "updated_at": "2025-01-13"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10020)