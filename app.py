import streamlit as st
import pandas as pd
import io
import time
from google import genai
from google.genai import types
from google.genai.errors import APIError

# --- 상수 설정 ---
# 사용 가능한 모델 목록
MODEL_OPTIONS = [
    'gemini-2.5-flash',
    'gemini-2.5-pro',
    'gemini-2.0-pro',
    'gemini-2.0-flash',
]

# 시스템 프롬프트 (심리 상담가 페르소나 및 안전 지침 포함)
SYSTEM_PROMPT = """
당신은 따뜻하고 공감 능력이 뛰어난 전문 심리 상담가입니다. 사용자는 자신이 겪고 있는 심리적인 고충에 대해서 털어놓습니다.

1. 사용자가 털어놓은 심리적인 문제 요인을 경청하고, 공감하며, 핵심 내용을 전문적인 지식을 활용하여 명확하게 정리해 주세요.
2. 정리된 내용을 바탕으로, 심리상담가로서 해줄 수 있는 구체적이고 전문적인 조언을 부드러운 어투로 제공하세요. 답변은 희망과 안정감을 줄 수 있도록, 사용자를 자극하거나 판단하지 않도록 각별히 주의해야 합니다.
3. 답변의 마지막에는 이 상담은 인공지능이 제공한 것으로, 상담 이후에도 심리적 불편이 해소되지 않을 경우, 반드시 전문 심리상담가나 정신과 전문가를 찾아 적절한 치료를 받을 것을 정중히 권장하세요.
4. 마지막으로, 사용자에게 다음 대화를 이어갈 수 있도록 "더 필요한 사항이나 나누고 싶은 이야기가 있으신가요?"와 같은 질문을 덧붙여 주세요.
"""

# --- Streamlit 상태 초기화 ---
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = f"session_{time.strftime('%Y%m%d%H%M%S')}"
if 'csv_log' not in st.session_state:
    st.session_state['csv_log'] = []


# --- 기능 함수 정의 ---

def initialize_chat_client(api_key):
    """API 키로 Gemini 클라이언트를 초기화합니다."""
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Gemini 클라이언트 초기화 오류: {e}")
        return None

def reset_conversation():
    """대화 기록 및 세션 ID를 초기화합니다."""
    st.session_state['history'] = []
    st.session_state['session_id'] = f"session_{time.strftime('%Y%m%d%H%M%S')}"
    st.session_state['csv_log'] = []
    st.rerun()

def call_gemini_with_retry(client, model_name, prompt, max_retries=3):
    """Gemini API를 호출하고 429 오류 시 재시도합니다."""
    # 히스토리 중 최근 6턴만 유지하여 API에 전달 (429 오류 방지 및 비용 절감)
    recent_history = st.session_state['history'][-12:]  # 6턴 = 12개의 메시지 파트 (user, model)
    
    # 대화 히스토리를 메시지 리스트로 구성 (현재 사용자 메시지 포함)
    contents = []
    for msg in recent_history:
        if 'role' in msg and 'text' in msg:
            contents.append(
                types.Content(
                    role=msg['role'],
                    parts=[types.Part(text=msg['text'])]
                )
            )
    
    # 현재 사용자 메시지 추가
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=prompt)]
        )
    )
    
    for attempt in range(max_retries):
        try:
            # generate_content를 사용하여 대화 생성 (히스토리 포함)
            # system_instruction을 직접 전달
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                system_instruction=SYSTEM_PROMPT
            )
            return response.text
        except APIError as e:
            if '429' in str(e) and attempt < max_retries - 1:
                st.warning(f"API 호출 제한(429) 발생. {attempt + 1}회차 재시도 중...")
                time.sleep(2 ** attempt)  # 지수 백오프
            else:
                st.error(f"Gemini API 호출 오류: {e}")
                return "죄송합니다. 현재 상담 서버에 오류가 발생하여 응답을 드릴 수 없습니다."
        except Exception as e:
            st.error(f"예기치 않은 오류 발생: {e}")
            import traceback
            st.error(f"상세 오류: {traceback.format_exc()}")
            return "죄송합니다. 처리 중 예기치 않은 오류가 발생했습니다."
    return "API 호출에 최종 실패했습니다. 나중에 다시 시도해 주세요."


# --- UI 정의 ---

st.set_page_config(page_title="심리상담 AI 챗봇", layout="centered")

st.title("🌱 심리상담 AI 챗봇")
st.caption("따뜻하고 전문적인 심리 상담을 제공합니다.")


# 1. API 키 설정 (Streamlit Secrets 또는 임시 UI)
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.info("Streamlit Secrets에 'GEMINI_API_KEY'가 설정되어 있지 않습니다. 아래에 임시 키를 입력해 주세요.", icon="🔑")
    api_key = st.text_input("Gemini API Key", type="password")
    if not api_key:
        st.stop()

# 클라이언트 초기화
client = initialize_chat_client(api_key)
if not client:
    st.stop()


# 2. 사이드바 및 설정
with st.sidebar:
    st.header("설정 및 도구")
    
    # 모델 선택
    selected_model = st.selectbox(
        "사용 모델 선택", 
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index('gemini-2.5-flash')
    )

    # 대화 정보
    st.subheader("대화 정보")
    st.markdown(f"**모델:** `{selected_model}`")
    st.markdown(f"**세션 ID:** `{st.session_state['session_id']}`")
    st.markdown(f"**대화 턴 수:** `{len(st.session_state['history']) // 2}`")
    
    # 로그 다운로드 옵션
    if st.button("💾 로그 다운로드 (CSV)"):
        df = pd.DataFrame(st.session_state['csv_log'])
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="CSV 파일 다운로드",
            data=csv_buffer.getvalue(),
            file_name=f"counseling_log_{st.session_state['session_id']}.csv",
            mime="text/csv"
        )

    # 대화 초기화
    if st.button("🔄 대화 초기화", type="primary"):
        reset_conversation()
        
    st.markdown("---")
    st.warning("⚠️ 본 챗봇은 AI 상담이며, 심각한 심리적 불편은 반드시 전문가와 상담해야 합니다.", icon="🚨")


# 3. 대화 표시 및 처리
for message in st.session_state['history']:
    # 메시지 표시 시에도 안전하게 'role'과 'text' 키를 사용
    if 'role' in message and 'text' in message:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "model" else "🙂"):
            st.markdown(message["text"])


# 4. 사용자 입력 처리
if user_prompt := st.chat_input("당신의 고민을 편안하게 털어놓아주세요..."):
    
    # a. 사용자 메시지 표시 (히스토리에는 API 호출 후 추가)
    with st.chat_message("user", avatar="🙂"):
        st.markdown(user_prompt)

    # b. Gemini 호출
    with st.spinner("전문적인 상담 답변을 생각하는 중입니다..."):
        
        # 재시도 로직을 포함하여 API 호출 (히스토리 포함)
        model_response = call_gemini_with_retry(client, selected_model, user_prompt)

    # c. 모델 응답 표시 및 기록
    with st.chat_message("model", avatar="🤖"):
        st.markdown(model_response)
    
    # d. 히스토리에 사용자 메시지와 모델 응답 추가
    st.session_state['history'].append({"role": "user", "text": user_prompt})
    st.session_state['history'].append({"role": "model", "text": model_response})
    
    # e. CSV 로그 기록
    st.session_state['csv_log'].append({
        'session_id': st.session_state['session_id'],
        'model': selected_model,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'role': 'user',
        'message': user_prompt
    })
    st.session_state['csv_log'].append({
        'session_id': st.session_state['session_id'],
        'model': selected_model,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'role': 'model',
        'message': model_response
    })

    # f. UI 업데이트를 위해 재실행
    st.rerun()