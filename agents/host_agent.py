"""
Host Agent (Port 10023)
- 전체 조율
- LangGraph 워크플로우
- RAG 검색
- 보고서 생성
"""

import os
import uuid
from pathlib import Path
from typing import TypedDict, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import httpx

from models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    Article,
    Paper
)

# 환경 변수 로드
load_dotenv()


# ========== Agent 자동 발견 ==========

async def discover_agents():
    """
    Agent 자동 발견
    각 Agent의 AgentCard를 조회하여 정보 수집
    """
    agent_urls = [
        "http://localhost:10020",
        "http://localhost:10021"
    ]
    
    discovered = []
    
    print("\n[Host] ===== Agent 자동 발견 시작 =====")
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        for url in agent_urls:
            try:
                response = await client.get(f"{url}/.well-known/agent-card.json")
                response.raise_for_status()
                
                card = response.json()
                discovered.append(card)
                
                print(f"[Host] ✅ Agent 발견: {card['name']}")
                print(f"       URL: {card['url']}")
                print(f"       능력: {[cap['name'] for cap in card['capabilities']]}")
                
            except httpx.ConnectError:
                print(f"[Host] ⚠️  Agent 연결 불가: {url}")
            except Exception as e:
                print(f"[Host] ❌ Agent 발견 실패 ({url}): {e}")
    
    print(f"[Host] ===== 총 {len(discovered)}개 Agent 발견 완료 =====\n")
    
    return discovered


# ========== Lifespan 이벤트 ==========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 라이프사이클 관리"""
    # Startup
    print("\n[Host] Host Agent 시작 중...")
    agents = await discover_agents()
    
    if len(agents) < 2:
        print("[Host] ⚠️  경고: 일부 Agent가 실행되지 않았습니다.")
        print("[Host] 뉴스 Agent(10020), 논문 Agent(10021)를 먼저 실행하세요.")
    
    yield
    
    # Shutdown
    print("\n[Host] Host Agent 종료 중...")


# FastAPI 앱
app = FastAPI(
    title="Host Agent",
    description="트렌드 분석 조율 Agent",
    lifespan=lifespan
)

# LLM 설정
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AOAI_ENDPOINT"),
    api_key=os.getenv("AOAI_API_KEY"),
    deployment_name=os.getenv("AOAI_DEPLOY_GPT4O"),
    api_version=os.getenv("AOAI_API_VERSION", "2024-08-01-preview"),
    temperature=0.7
)

# 임베딩 모델
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AOAI_ENDPOINT"),
    api_key=os.getenv("AOAI_API_KEY"),
    azure_deployment=os.getenv("AOAI_DEPLOY_EMBED_3_LARGE"),
    chunk_size=1
)


# ========== State 정의 ==========

class AgentState(TypedDict):
    """LangGraph State"""
    topic: str
    news: List[dict]
    papers: List[dict]
    report: str
    session_id: str


# ========== vectorstore 관리 ==========

def save_vectorstore(session_id: str, vectorstore):
    """vectorstore 저장"""
    try:
        Path("./vector_db").mkdir(exist_ok=True)
        vectorstore.save_local(f"./vector_db/{session_id}")
        print(f"[Host] vectorstore 저장 완료: {session_id}")
    except Exception as e:
        print(f"[Host] vectorstore 저장 실패: {e}")


def load_vectorstore(session_id: str):
    """vectorstore 로드"""
    try:
        vectorstore = FAISS.load_local(
            f"./vector_db/{session_id}",
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"[Host] vectorstore 로드 완료: {session_id}")
        return vectorstore
    except Exception as e:
        print(f"[Host] vectorstore 로드 실패: {e}")
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")


# ========== Node 함수들 ==========

async def fetch_news(state: AgentState) -> dict:
    """뉴스 Agent 호출"""
    try:
        print(f"[Host] 뉴스 Agent 호출: {state['topic']}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:10020/search",
                params={"topic": state["topic"]}
            )
            response.raise_for_status()
        
        news_data = response.json()
        articles = news_data.get("articles", [])
        
        print(f"[Host] 뉴스 {len(articles)}개 수집")
        return {"news": articles}
    
    except httpx.TimeoutException:
        print("[Host] 뉴스 Agent 타임아웃")
        return {"news": []}
    except Exception as e:
        print(f"[Host] 뉴스 Agent 호출 실패: {e}")
        return {"news": []}


async def fetch_papers(state: AgentState) -> dict:
    """논문 Agent 호출"""
    try:
        print(f"[Host] 논문 Agent 호출: {state['topic']}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:10021/search",
                params={"topic": state["topic"]}
            )
            response.raise_for_status()
        
        paper_data = response.json()
        papers = paper_data.get("papers", [])
        
        print(f"[Host] 논문 {len(papers)}개 수집")
        return {"papers": papers}
    
    except httpx.TimeoutException:
        print("[Host] 논문 Agent 타임아웃")
        return {"papers": []}
    except Exception as e:
        print(f"[Host] 논문 Agent 호출 실패: {e}")
        return {"papers": []}


async def create_vectorstore(state: AgentState) -> dict:
    """RAG를 위한 vectorstore 생성"""
    try:
        print("[Host] vectorstore 생성 시작")
        
        # 문서 변환
        docs = []
        
        # 뉴스 → Document
        for article in state["news"]:
            docs.append(Document(
                page_content=f"{article['title']}\n\n{article['summary']}",
                metadata={
                    "source": "news",
                    "title": article["title"],
                    "date": article.get("date", "")
                }
            ))
        
        # 논문 → Document
        for paper in state["papers"]:
            docs.append(Document(
                page_content=f"{paper['title']}\n\n{paper['abstract']}",
                metadata={
                    "source": "paper",
                    "title": paper["title"],
                    "authors": ", ".join(paper["authors"]),
                    "url": paper["url"]
                }
            ))
        
        if not docs:
            print("[Host] 문서가 없어 vectorstore 생성 스킵")
            return {}
        
        # 청킹
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        split_docs = splitter.split_documents(docs)
        
        # 벡터스토어 생성
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        
        # 저장
        save_vectorstore(state["session_id"], vectorstore)
        
        print(f"[Host] vectorstore 생성 완료: {len(split_docs)}개 청크")
        return {}
    
    except Exception as e:
        print(f"[Host] vectorstore 생성 실패: {e}")
        return {}


async def generate_report(state: AgentState) -> dict:
    """보고서 생성"""
    try:
        print("[Host] 보고서 생성 시작")
        
        # 경고 메시지 생성
        warnings = []
        if not state["news"]:
            warnings.append("⚠️ 뉴스 검색이 지연되어 뉴스 정보가 포함되지 않았습니다.")
        if not state["papers"]:
            warnings.append("⚠️ 논문 검색이 지연되어 논문 정보가 포함되지 않았습니다.")
        
        if not state["news"] and not state["papers"]:
            return {"report": "검색 결과가 없습니다. 다른 주제로 시도해주세요."}
        
        # 프롬프트
        prompt = ChatPromptTemplate.from_template(
            """당신은 기술 트렌드 분석 전문가입니다.
다음 정보를 바탕으로 기술 트렌드 보고서를 작성하세요.

보고서 구성:
1. 📊 제목: "{topic} 트렌드 보고서"
2. 🔥 핵심 트렌드 요약 (3-5줄)
3. 🏷️ 주요 키워드 (5개, 해시태그 형식)
4. 📰 뉴스 하이라이트 (있는 경우만)
5. 📄 논문 하이라이트 (있는 경우만)
6. 💡 종합 분석

뉴스:
{news}

논문:
{papers}

주제: {topic}
"""
        )
        
        chain = prompt | llm | StrOutputParser()
        report = chain.invoke({
            "topic": state["topic"],
            "news": state["news"] if state["news"] else "검색 결과 없음",
            "papers": state["papers"] if state["papers"] else "검색 결과 없음"
        })
        
        # 경고 메시지 추가
        if warnings:
            report = "\n".join(warnings) + "\n\n" + report
        
        print("[Host] 보고서 생성 완료")
        return {"report": report}
    
    except Exception as e:
        print(f"[Host] 보고서 생성 실패: {e}")
        return {"report": f"보고서 생성 중 오류가 발생했습니다: {e}"}


# ========== LangGraph 구성 ==========

workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("fetch_news", fetch_news)
workflow.add_node("fetch_papers", fetch_papers)
workflow.add_node("create_vectorstore", create_vectorstore)
workflow.add_node("generate_report", generate_report)

# 엣지 연결
workflow.set_entry_point("fetch_news")
workflow.add_edge("fetch_news", "fetch_papers")
workflow.add_edge("fetch_papers", "create_vectorstore")
workflow.add_edge("create_vectorstore", "generate_report")
workflow.add_edge("generate_report", END)

# 메모리 추가
memory = MemorySaver()
graph_app = workflow.compile(checkpointer=memory)


# ========== API 엔드포인트 ==========

@app.get("/")
async def root():
    """헬스 체크"""
    return {"status": "ok", "agent": "host"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_trend(request: AnalyzeRequest):
    """
    트렌드 분석 실행
    
    Args:
        request: 분석 요청 (topic, max_news, max_papers)
        
    Returns:
        AnalyzeResponse (report, session_id)
    """
    try:
        print(f"[Host] 분석 요청: {request.topic}")
        
        # 세션 ID 생성
        session_id = str(uuid.uuid4())
        
        # 세션별 실행
        config = {"configurable": {"thread_id": session_id}}
        
        # 초기 상태
        initial_state = {
            "topic": request.topic,
            "news": [],
            "papers": [],
            "report": "",
            "session_id": session_id
        }
        
        # 워크플로우 실행
        result = await graph_app.ainvoke(initial_state, config)
        
        return AnalyzeResponse(
            report=result["report"],
            session_id=session_id
        )
    
    except Exception as e:
        print(f"[Host] 분석 실패: {e}")
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {e}")


@app.post("/rag_query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """
    RAG 기반 추가 질문
    
    Args:
        request: RAG 질의 (query, session_id)
        
    Returns:
        RAGQueryResponse (answer)
    """
    try:
        print(f"[Host] RAG 질의: {request.query}")
        
        # vectorstore 로드
        vectorstore = load_vectorstore(request.session_id)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # 문서 포맷팅
        def format_docs(docs):
            formatted = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata['source']
                title = doc.metadata['title']
                content = doc.page_content
                formatted.append(
                    f"[문서 {i} - {source.upper()}]\n"
                    f"제목: {title}\n"
                    f"내용: {content}"
                )
            return "\n\n".join(formatted)
        
        # RAG 프롬프트
        template = """다음 문서를 참고하여 질문에 답변하세요.
**반드시 어떤 문서를 참고했는지 [문서 N] 형식으로 명시하세요.**
문서에 없는 내용은 "문서에서 찾을 수 없습니다"라고 답하세요.

참고 문서:
{context}

질문: {question}

답변 (출처 포함):
"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # RAG 체인
        chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        answer = chain.invoke(request.query)
        
        print("[Host] RAG 질의 완료")
        return RAGQueryResponse(answer=answer)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Host] RAG 질의 실패: {e}")
        raise HTTPException(status_code=500, detail=f"답변 생성 중 오류 발생: {e}")


@app.get("/.well-known/agent-card.json")
async def agent_card():
    """
    Agent Card - A2A 표준 프로토콜
    Agent의 능력과 스펙을 공개
    """
    return {
        "version": "1.0.0",
        "name": "Host Agent",
        "description": "트렌드 분석 조율 Agent. Multi-Agent 협업 및 RAG 기반 Q&A 제공.",
        "url": "http://localhost:10023",
        "capabilities": [
            {
                "name": "analyze_trend",
                "description": "기술 트렌드 자동 분석 (뉴스 + 논문 통합)",
                "endpoint": "/analyze",
                "method": "POST",
                "parameters": {
                    "topic": {
                        "type": "string",
                        "description": "분석할 기술 주제",
                        "required": True,
                        "example": "RAG"
                    },
                    "max_news": {
                        "type": "integer",
                        "description": "최대 뉴스 개수",
                        "required": False,
                        "default": 5
                    },
                    "max_papers": {
                        "type": "integer",
                        "description": "최대 논문 개수",
                        "required": False,
                        "default": 5
                    }
                },
                "response_schema": {
                    "type": "AnalyzeResponse",
                    "properties": {
                        "report": "string (markdown)",
                        "session_id": "string (uuid)"
                    }
                }
            },
            {
                "name": "rag_query",
                "description": "RAG 기반 추가 질문 (분석 결과 기반)",
                "endpoint": "/rag_query",
                "method": "POST",
                "parameters": {
                    "query": {
                        "type": "string",
                        "description": "질문 내용",
                        "required": True,
                        "example": "RAG 구현 시 주의사항은?"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "분석 세션 ID",
                        "required": True
                    }
                },
                "response_schema": {
                    "type": "RAGQueryResponse",
                    "properties": {
                        "answer": "string (출처 포함)"
                    }
                }
            }
        ],
        "dependencies": [
            {
                "name": "뉴스 Agent",
                "url": "http://localhost:10020",
                "type": "A2A"
            },
            {
                "name": "논문 Agent",
                "url": "http://localhost:10021",
                "type": "A2A"
            }
        ],
        "features": [
            "Multi-Agent A2A 협업",
            "LangGraph 워크플로우",
            "RAG (Retrieval-Augmented Generation)",
            "Vector Database (FAISS)",
            "세션 기반 Q&A"
        ],
        "tags": ["orchestration", "rag", "analysis", "multi-agent"],
        "author": "Tech Trend Scout Team",
        "contact": "http://localhost:10023",
        "created_at": "2025-01-13",
        "updated_at": "2025-01-13"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10023)