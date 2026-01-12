"""
뉴스 검색 도구 (Mock - 회사 환경용)
"""

import json
from typing import List, Dict


class NewsSearchTool:
    """기술 뉴스 검색 도구 (Mock)"""
    
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        print("[뉴스 Tool] Mock 모드로 초기화")
    
    def search_news(self, topic: str) -> str:
        """Mock 뉴스 데이터"""
        print(f"[뉴스 Tool] Mock 데이터 생성: {topic}")
        
        mock_results = [
            {
                "title": f"{topic.upper()} 기술 동향: AI 분야 혁신 가속화",
                "snippet": f"{topic}는 최근 AI 분야에서 가장 주목받는 기술입니다. 대규모 언어모델의 한계를 보완하는 핵심 방법론으로, 기업들의 도입이 급증하고 있습니다.",
                "link": "https://techcrunch.com/example1",
                "date": "2025-01-10"
            },
            {
                "title": f"글로벌 기업들의 {topic.upper()} 도입 현황",
                "snippet": f"구글, 마이크로소프트 등 글로벌 테크 기업들이 {topic} 기술을 자사 AI 서비스에 적극 통합하고 있습니다. 내부 문서 기반 Q&A 시스템 구축 사례가 증가하는 추세입니다.",
                "link": "https://venturebeat.com/example2",
                "date": "2025-01-09"
            },
            {
                "title": f"{topic.upper()} 성능 개선 연구 동향",
                "snippet": f"검색 품질 향상과 응답 정확도 개선을 위한 {topic} 연구가 활발합니다. Self-RAG, Adaptive-RAG 등 차세대 기술들이 주목받고 있습니다.",
                "link": "https://arxiv.org/example3",
                "date": "2025-01-08"
            },
            {
                "title": f"벡터 데이터베이스 시장 급성장, {topic.upper()}가 견인",
                "snippet": f"{topic} 도입 증가로 Pinecone, Weaviate, Chroma 등 벡터DB 기업들의 성장이 가속화되고 있습니다.",
                "link": "https://techradar.com/example4",
                "date": "2025-01-07"
            },
            {
                "title": f"{topic.upper()} 실무 적용 시 주요 고려사항",
                "snippet": f"청킹 전략, 임베딩 모델 선택, 검색 품질 모니터링 등 {topic} 실무 적용 시 반드시 고려해야 할 핵심 요소들을 전문가들이 제시했습니다.",
                "link": "https://medium.com/example5",
                "date": "2025-01-06"
            }
        ]
        
        return json.dumps(mock_results[:self.max_results])
    
    def format_results(self, results: str) -> List[Dict]:
        """검색 결과 포맷팅"""
        try:
            parsed = json.loads(results)
            formatted = []
            for item in parsed[:self.max_results]:
                formatted.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", ""),
                    "date": item.get("date", "")
                })
            return formatted
        except Exception as e:
            print(f"결과 포맷팅 실패: {e}")
            return []


news_search_tool = NewsSearchTool()