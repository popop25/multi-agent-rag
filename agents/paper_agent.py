"""
논문 Agent (Port 10021)
- arXiv 논문 검색
- 메타데이터 수집
"""

import os
from fastapi import FastAPI
from dotenv import load_dotenv

from models.schemas import PaperResponse, Paper
from tools.arxiv_tool import arxiv_search_tool

# 환경 변수 로드
load_dotenv()

# FastAPI 앱
app = FastAPI(title="Paper Agent", description="arXiv 논문 검색 Agent")


@app.get("/")
async def root():
    """헬스 체크"""
    return {"status": "ok", "agent": "paper"}


@app.post("/search", response_model=PaperResponse)
async def search_papers(topic: str, max_results: int = 5):
    """
    논문 검색 엔드포인트
    
    Args:
        topic: 검색 주제
        max_results: 최대 결과 개수
        
    Returns:
        PaperResponse
    """
    try:
        print(f"[논문 Agent] 검색 시작: {topic}")
        
        # arXiv 검색
        papers_data = arxiv_search_tool.search_papers(topic)
        
        if not papers_data:
            print("[논문 Agent] 검색 결과 없음")
            return PaperResponse(papers=[])
        
        # Paper 객체로 변환
        papers = []
        for data in papers_data[:max_results]:
            papers.append(Paper(
                title=data["title"],
                authors=data["authors"],
                date=data["date"],
                abstract=data["abstract"],
                url=data["url"]
            ))
        
        print(f"[논문 Agent] {len(papers)}개 논문 수집 완료")
        return PaperResponse(papers=papers)
    
    except Exception as e:
        print(f"[논문 Agent] 오류 발생: {e}")
        return PaperResponse(papers=[])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10021)