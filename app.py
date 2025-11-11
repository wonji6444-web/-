import streamlit as st
import pandas as pd
import io
import time
from google import genai
from google.genai.errors import APIError

# --- 상수 설정 ---
MODEL_OPTIONS = [
    'gemini-2.0-flash',
    'gemini-2.0-pro',
    'gemini-1.5-flash',
    'gemini-1.5-pro',
]

SYSTEM_PROMPT = """당신은 따뜻하고 공감 능력이 뛰어난 전문 심리 상담가입니다.

1. 사용자는 자신이 겪고 있는 심리적인 고충에 대해서 털어놓습니다.
2. 사용자의 심리적인 문제요인을 정리하고, 이에 대해 전문적인 지식을 활용하여, 심리상담가가 해줄 수 있는 답변을 제공하세요.
3. 마지막에는 이 상담은 AI가 답변을 제공한 것으로, 상담 이후에도 심리적 불편이 해소되지 않을 경우, 전문가를 찾아 치료받을 것을 권장하세요. 이후 더 필요한 사항에 대해서 물어보세요.

답변은 희망과 안정감을 줄 수 있도록, 사용자를 자극하거나 판단하지 않도록 각별히 주의해야 합니다."""

# --- Streamlit 상태 초기화 ---
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = f"session_{time.strftime('%Y%m%d%H%M%S')}"
if 'csv_log' not in st.session_state:
    st.session_state['csv_log'] = []

# --- 기능 함수 정의 ---
def initialize_client(api_key):
    """Gemini 클라이언트 초기화"""
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"클라이언트 초기화 오류: {e}")
        return None

def reset_conversation():
    """대화 초기화"""
    st.session_state['history'] = []
    st.session_state['session_id'] = f"session_{time.strftime('%Y%m%d%H%M%S')}"
    st.session_state['csv_log'] = []
    st.rerun()

def call_gemini_api(client, model_name, prompt, max_retries=5):
    """Gemini API 호출 (429/503 재시도 포함, 최근 6턴 히스토리 유지)"""
    # 최근 6턴 히스토리 추출 (12개 메시지: user + model 쌍)
    recent_history = st.session_state['history'][-12:] if len(st.session_state['history']) > 0 else []
    
    # 히스토리 컨텍스트 구성
    history_context = ""
    for msg in recent_history:
        if msg['role'] == 'user':
            history_context += f"사용자: {msg['text']}\n"
        elif msg['role'] == 'model':
            history_context += f"상담가: {msg['text']}\n"
    
    # 전체 프롬프트 구성 (시스템 프롬프트 + 히스토리 + 현재 메시지)
    if history_context:
        full_prompt = f"{SYSTEM_PROMPT}\n\n이전 대화:\n{history_context}\n사용자: {prompt}"
    else:
        full_prompt = f"{SYSTEM_PROMPT}\n\n사용자: {prompt}"
    
    for attempt in range(max_retries):
        try:
            # 세션 생성 및 메시지 전송
            chat = client.chats.create(model=model_name)
            response = chat.send_message(full_prompt)
            return response.text
            
        except APIError as e:
            error_msg = str(e)
            # 429 (Rate Limit) 또는 503 (Service Unavailable) 오류 재시도
            if ('429' in error_msg or '503' in error_msg or 'UNAVAILABLE' in error_msg) and attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 10)  # 최대 10초 대기 (지수 백오프)
                st.warning(f"서버 일시적 오류 발생. {wait_time}초 후 재시도 중... ({(attempt + 1)}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                st.error(f"API 오류: {error_msg}")
                return f"죄송합니다. API 오류가 발생했습니다: {error_msg}"
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 5)
                st.warning(f"오류 발생. {wait_time}초 후 재시도 중... ({(attempt + 1)}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                st.error(f"예기치 않은 오류: {e}")
                return f"죄송합니다. 오류가 발생했습니다: {str(e)}"
    
    return "죄송합니다. 여러 번 시도했지만 응답을 받을 수 없었습니다. 잠시 후 다시 시도해 주세요."

# --- UI 정의 ---
st.set_page_config(page_title="심리상담 AI 챗봇", layout="centered")

st.title("🌱 심리상담 AI 챗봇")
st.caption("따뜻하고 전문적인 심리 상담을 제공합니다.")

# 1. API 키 설정
api_key = st.secrets.get('GEMINI_API_KEY')
if not api_key:
    st.info("🔑 Streamlit Secrets에 'GEMINI_API_KEY'가 설정되어 있지 않습니다. 아래에 임시 키를 입력해 주세요.")
    api_key = st.text_input("Gemini API Key", type="password")
    if not api_key:
        st.stop()

# 클라이언트 초기화
client = initialize_client(api_key)
if not client:
    st.stop()

# 2. 사이드바 설정
with st.sidebar:
    st.header("설정 및 도구")
    
    # 모델 선택
    selected_model = st.selectbox(
        "사용 모델 선택",
        MODEL_OPTIONS,
        index=0  # gemini-2.0-flash 기본 선택
    )
    
    # 대화 정보
    st.subheader("대화 정보")
    st.markdown(f"**모델:** `{selected_model}`")
    st.markdown(f"**세션 ID:** `{st.session_state['session_id']}`")
    st.markdown(f"**대화 턴 수:** `{len(st.session_state['history']) // 2}`")
    
    # 로그 다운로드
    if st.button("💾 로그 다운로드 (CSV)"):
        if st.session_state['csv_log']:
            df = pd.DataFrame(st.session_state['csv_log'])
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            st.download_button(
                label="CSV 파일 다운로드",
                data=csv_buffer.getvalue(),
                file_name=f"counseling_log_{st.session_state['session_id']}.csv",
                mime="text/csv"
            )
        else:
            st.info("다운로드할 로그가 없습니다.")
    
    # 대화 초기화
    if st.button("🔄 대화 초기화", type="primary"):
        reset_conversation()
    
    st.markdown("---")
    st.warning("⚠️ 본 챗봇은 AI 상담이며, 심각한 심리적 불편은 반드시 전문가와 상담해야 합니다.")

# 3. 대화 히스토리 표시
for message in st.session_state['history']:
    if 'role' in message and 'text' in message:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "model" else "🙂"):
            st.markdown(message["text"])

# 4. 사용자 입력 처리
if user_prompt := st.chat_input("당신의 고민을 편안하게 털어놓아주세요..."):
    # 사용자 메시지 표시
    with st.chat_message("user", avatar="🙂"):
        st.markdown(user_prompt)
    
    # AI 응답 생성
    with st.spinner("전문적인 상담 답변을 생각하는 중입니다..."):
        model_response = call_gemini_api(client, selected_model, user_prompt)
    
    # AI 응답 표시
    with st.chat_message("model", avatar="🤖"):
        st.markdown(model_response)
    
    # 히스토리 저장
    st.session_state['history'].append({"role": "user", "text": user_prompt})
    st.session_state['history'].append({"role": "model", "text": model_response})
    
    # CSV 로그 기록
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    st.session_state['csv_log'].append({
        'session_id': st.session_state['session_id'],
        'model': selected_model,
        'timestamp': timestamp,
        'role': 'user',
        'message': user_prompt
    })
    st.session_state['csv_log'].append({
        'session_id': st.session_state['session_id'],
        'model': selected_model,
        'timestamp': timestamp,
        'role': 'model',
        'message': model_response
    })
    
    # UI 업데이트
    st.rerun()
