"""
도구 테스트
"""

print("=== 뉴스 검색 테스트 ===")
try:
    from tools.search_tool import news_search_tool
    result = news_search_tool.search_news("AI")
    print(f"결과 길이: {len(result)}")
    print("뉴스 검색: OK")
except Exception as e:
    print(f"뉴스 검색 실패: {e}")

print("\n=== 논문 검색 테스트 ===")
try:
    from tools.arxiv_tool import arxiv_search_tool
    papers = arxiv_search_tool.search_papers("machine learning")
    print(f"검색된 논문: {len(papers)}개")
    if papers:
        print(f"첫 번째 논문: {papers[0]['title']}")
    print("논문 검색: OK")
except Exception as e:
    print(f"논문 검색 실패: {e}")