# Tech Trend Scout

기술 트렌드 자동 분석 Agent 서비스

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
# Terminal 1: 뉴스 Agent
uvicorn agents.news_agent:app --port 10020

# Terminal 2: 논문 Agent
uvicorn agents.paper_agent:app --port 10021

# Terminal 3: Host Agent
uvicorn agents.host_agent:app --port 10023

# Terminal 4: Streamlit UI
streamlit run app.py
```

## 사용법

1. 주제 입력 (예: "RAG")
2. "분석 시작" 버튼 클릭
3. 보고서 확인
4. 추가 질문 입력
