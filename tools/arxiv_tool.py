"""
arXiv 논문 검색 도구
"""

import arxiv
from typing import List, Dict
from datetime import datetime


class ArxivSearchTool:
    """arXiv 논문 검색 도구"""
    
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
    
    def search_papers(self, topic: str) -> List[Dict]:
        """
        논문 검색
        
        Args:
            topic: 검색 주제
            
        Returns:
            논문 정보 리스트
        """
        try:
            # arXiv 검색
            search = arxiv.Search(
                query=topic,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for result in search.results():
                papers.append({
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "date": result.published.strftime("%Y-%m-%d"),
                    "abstract": result.summary[:500],  # 500자 제한
                    "url": result.pdf_url
                })
            
            return papers
        
        except Exception as e:
            print(f"논문 검색 실패: {e}")
            return []
    
    def format_for_display(self, papers: List[Dict]) -> str:
        """
        논문 정보를 보기 좋게 포맷팅
        
        Args:
            papers: 논문 정보 리스트
            
        Returns:
            포맷팅된 문자열
        """
        if not papers:
            return "검색된 논문이 없습니다."
        
        formatted = []
        for i, paper in enumerate(papers, 1):
            authors = ", ".join(paper["authors"][:3])  # 처음 3명만
            if len(paper["authors"]) > 3:
                authors += " et al."
            
            formatted.append(
                f"{i}. {paper['title']}\n"
                f"   저자: {authors}\n"
                f"   날짜: {paper['date']}\n"
                f"   URL: {paper['url']}"
            )
        
        return "\n\n".join(formatted)


# 싱글톤 인스턴스
arxiv_search_tool = ArxivSearchTool()