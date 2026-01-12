"""
arXiv 논문 검색 도구 (Mock - 회사 환경용)
"""

from typing import List, Dict
from datetime import datetime, timedelta


class ArxivSearchTool:
    """arXiv 논문 검색 도구 (Mock)"""
    
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        print("[논문 Tool] Mock 모드로 초기화")
    
    def search_papers(self, topic: str) -> List[Dict]:
        """Mock 논문 데이터"""
        print(f"[논문 Tool] Mock 데이터 생성: {topic}")
        
        base_date = datetime.now()
        
        mock_papers = [
            {
                "title": f"Self-{topic.upper()}: Self-Reflective Retrieval-Augmented Generation",
                "authors": ["John Smith", "Jane Doe", "Robert Johnson"],
                "date": (base_date - timedelta(days=30)).strftime("%Y-%m-%d"),
                "abstract": f"We introduce Self-{topic.upper()}, a novel framework that learns to retrieve, generate, and critique through self-reflection. Our approach enables the model to adaptively decide when and what to retrieve across diverse queries, improving factual accuracy and reducing hallucination.",
                "url": "https://arxiv.org/abs/2401.00001"
            },
            {
                "title": f"Adaptive-{topic.upper()}: Learning Dynamic Retrieval Strategies",
                "authors": ["Alice Chen", "Bob Lee", "Carol Martinez"],
                "date": (base_date - timedelta(days=45)).strftime("%Y-%m-%d"),
                "abstract": f"This paper presents Adaptive-{topic.upper()}, which dynamically selects retrieval strategies based on query complexity. We demonstrate significant improvements in both efficiency and accuracy across multiple benchmarks.",
                "url": "https://arxiv.org/abs/2312.99999"
            },
            {
                "title": f"Corrective {topic.upper()}: Self-Correcting Retrieval Augmentation",
                "authors": ["David Kim", "Emma Wilson"],
                "date": (base_date - timedelta(days=60)).strftime("%Y-%m-%d"),
                "abstract": f"We propose Corrective {topic.upper()} (CRAG), which evaluates retrieved documents and triggers web search when relevance is insufficient. This approach significantly reduces factual errors in generation.",
                "url": "https://arxiv.org/abs/2312.88888"
            },
            {
                "title": f"Multi-Modal {topic.upper()}: Integrating Vision and Language",
                "authors": ["Frank Zhang", "Grace Park", "Henry Brown"],
                "date": (base_date - timedelta(days=75)).strftime("%Y-%m-%d"),
                "abstract": f"We extend {topic.upper()} to multi-modal settings, enabling retrieval and generation across text, images, and tables. Our experiments show superior performance in visual question answering tasks.",
                "url": "https://arxiv.org/abs/2311.77777"
            },
            {
                "title": f"Efficient {topic.upper()} for Long-Context Applications",
                "authors": ["Iris Liu", "Jack Cooper"],
                "date": (base_date - timedelta(days=90)).strftime("%Y-%m-%d"),
                "abstract": f"This work addresses computational challenges in {topic.upper()} for long documents. We propose chunking strategies and caching mechanisms that reduce latency by 60% while maintaining accuracy.",
                "url": "https://arxiv.org/abs/2311.66666"
            }
        ]
        
        return mock_papers[:self.max_results]
    
    def format_for_display(self, papers: List[Dict]) -> str:
        """논문 정보를 보기 좋게 포맷팅"""
        if not papers:
            return "검색된 논문이 없습니다."
        
        formatted = []
        for i, paper in enumerate(papers, 1):
            authors = ", ".join(paper["authors"][:3])
            if len(paper["authors"]) > 3:
                authors += " et al."
            
            formatted.append(
                f"{i}. {paper['title']}\n"
                f"   저자: {authors}\n"
                f"   날짜: {paper['date']}\n"
                f"   URL: {paper['url']}"
            )
        
        return "\n\n".join(formatted)


arxiv_search_tool = ArxivSearchTool()