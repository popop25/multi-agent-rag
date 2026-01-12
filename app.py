"""
Tech Trend Scout - Streamlit UI
"""

import streamlit as st
import requests
from datetime import datetime

# 페이지 설정
st.set_page_config(
    page_title="Tech Trend Scout",
    page_icon="🔍",
    layout="wide"
)

# CSS 스타일
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 1rem;
}
.sub-header {
    text-align: center;
    color: #666;
    margin-bottom: 2rem;
}
.warning {
    background-color: #fff3cd;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ffc107;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "report" not in st.session_state:
    st.session_state.report = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ========== 헤더 ==========
st.markdown('<p class="main-header">🔍 Tech Trend Scout</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI 기반 기술 트렌드 자동 분석 서비스</p>', unsafe_allow_html=True)

# ========== 사이드바 ==========
with st.sidebar:
    st.title("⚙️ 설정")
    
    st.markdown("---")
    
    max_news = st.number_input(
        "뉴스 개수",
        min_value=1,
        max_value=10,
        value=5,
        help="수집할 최대 뉴스 개수"
    )
    
    max_papers = st.number_input(
        "논문 개수",
        min_value=1,
        max_value=10,
        value=5,
        help="수집할 최대 논문 개수"
    )
    
    st.markdown("---")
    
    # Agent 상태 확인
    st.subheader("🤖 Agent 상태")
    
    def check_agent(url, name):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return f"✅ {name}: 정상"
            else:
                return f"❌ {name}: 오류"
        except:
            return f"⚠️ {name}: 연결 불가"
    
    news_status = check_agent("http://localhost:10020", "뉴스 Agent")
    paper_status = check_agent("http://localhost:10021", "논문 Agent")
    host_status = check_agent("http://localhost:10023", "Host Agent")
    
    st.text(news_status)
    st.text(paper_status)
    st.text(host_status)
    
    st.markdown("---")
    
    # 새로 시작
    if st.button("🔄 새로 시작", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.report = None
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    st.caption("💡 Tip: Agent가 연결 불가 상태라면\n각 Agent를 먼저 실행해주세요.")

# ========== 메인 영역 ==========

# 탭 구성
tab1, tab2 = st.tabs(["📊 트렌드 분석", "💬 추가 질문 (RAG)"])

with tab1:
    st.subheader("트렌드 분석")
    
    # 주제 입력
    col1, col2 = st.columns([4, 1])
    
    with col1:
        topic = st.text_input(
            "분석할 주제를 입력하세요",
            placeholder="예: RAG, LangChain, Kubernetes",
            label_visibility="collapsed"
        )
    
    with col2:
        analyze_button = st.button("🔍 분석 시작", type="primary", use_container_width=True)
    
    # 분석 실행
    if analyze_button:
        if not topic:
            st.warning("주제를 입력해주세요.")
        else:
            try:
                with st.spinner("🔄 트렌드 분석 중... (15-20초 소요)"):
                    # Host Agent 호출
                    response = requests.post(
                        "http://localhost:10023/analyze",
                        json={
                            "topic": topic,
                            "max_news": max_news,
                            "max_papers": max_papers
                        },
                        timeout=60
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    st.session_state.session_id = result["session_id"]
                    st.session_state.report = result["report"]
                    st.session_state.chat_history = []  # 채팅 히스토리 초기화
                
                st.success("✅ 분석 완료!")
                st.rerun()
            
            except requests.exceptions.Timeout:
                st.error("⏱️ 요청 시간 초과. Agent가 실행 중인지 확인해주세요.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Host Agent에 연결할 수 없습니다. Agent를 먼저 실행해주세요.")
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ HTTP 오류: {e}")
            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")
    
    # 보고서 출력
    if st.session_state.report:
        st.markdown("---")
        st.markdown(st.session_state.report)
        
        # 다운로드 버튼
        st.download_button(
            label="📥 보고서 다운로드",
            data=st.session_state.report,
            file_name=f"trend_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

with tab2:
    st.subheader("추가 질문 (RAG)")
    
    if not st.session_state.session_id:
        st.info("💡 먼저 '트렌드 분석' 탭에서 분석을 실행해주세요.")
    else:
        st.caption(f"세션 ID: {st.session_state.session_id[:8]}...")
        
        # 채팅 히스토리 표시
        for chat in st.session_state.chat_history:
            with st.chat_message(chat["role"]):
                st.write(chat["content"])
        
        # 질문 입력
        question = st.chat_input("궁금한 점을 질문하세요")
        
        if question:
            # 사용자 메시지 추가
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })
            
            with st.chat_message("user"):
                st.write(question)
            
            # RAG 질의
            try:
                with st.spinner("답변 생성 중..."):
                    response = requests.post(
                        "http://localhost:10023/rag_query",
                        json={
                            "query": question,
                            "session_id": st.session_state.session_id
                        },
                        timeout=30
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    answer = result["answer"]
                    
                    # 어시스턴트 메시지 추가
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })
                    
                    with st.chat_message("assistant"):
                        st.write(answer)
            
            except requests.exceptions.Timeout:
                st.error("⏱️ 요청 시간 초과")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    st.error("❌ 세션을 찾을 수 없습니다. 다시 분석을 실행해주세요.")
                else:
                    st.error(f"❌ HTTP 오류: {e}")
            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")

# ========== 푸터 ==========
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p>🚀 Tech Trend Scout | AI Bootcamp 최종 과제</p>
        <p style='font-size: 0.9rem;'>Multi-Agent A2A 기반 기술 트렌드 분석 서비스</p>
    </div>
    """,
    unsafe_allow_html=True
)