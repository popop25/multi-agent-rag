"""
DuckDuckGo 검색 도구
"""

from langchain_community.tools import DuckDuckGoSearchResults
from typing import List, Dict


class NewsSearchTool:
    """기술 뉴스 검색 도구"""
    
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.search = DuckDuckGoSearchResults(
            num_results=max_results,
            backend="news"
        )
    
    def search_news(self, topic: str) -> str:
        """
        뉴스 검색
        
        Args:
            topic: 검색 주제
            
        Returns:
            검색 결과 문자열
        """
        try:
            query = f"{topic} tech news recent"
            results = self.search.run(query)
            return results
        except Exception as e:
            print(f"뉴스 검색 실패: {e}")
            return "[]"
    
    def format_results(self, results: str) -> List[Dict]:
        """
        검색 결과 포맷팅
        
        Args:
            results: 검색 결과 문자열
            
        Returns:
            포맷팅된 결과 리스트
        """
        try:
            import json
            # DuckDuckGo 결과는 JSON 문자열로 반환됨
            parsed = json.loads(results) if isinstance(results, str) else results
            
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


# 싱글톤 인스턴스
news_search_tool = NewsSearchTool()