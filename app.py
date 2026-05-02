"""
app.py — Company Policy AI Chat App
Streamlit UI powered by LangChain + Google GenAI + Chroma RAG
"""

import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

# ── Page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="ACME Policy Assistant",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;1,9..144,300&display=swap');

  /* Global */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* Hide default Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }

  /* Main background */
  .stApp {
    background: #0f1117;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #1e2535;
  }
  [data-testid="stSidebar"] * { color: #c8d0e0 !important; }

  /* Top brand bar */
  .brand-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 0 28px 0;
    border-bottom: 1px solid #1e2535;
    margin-bottom: 24px;
  }
  .brand-icon {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #4f7cff 0%, #8b5cf6 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
  }
  .brand-name {
    font-family: 'Fraunces', serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #e8eaf0 !important;
    line-height: 1.2;
  }
  .brand-sub {
    font-size: 0.7rem;
    color: #5a6580 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  /* Section headers in sidebar */
  .sidebar-section {
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #3d4a63 !important;
    font-weight: 600;
    margin: 20px 0 8px 0;
  }

  /* Status pill */
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
  }
  .status-ready {
    background: rgba(34, 197, 94, 0.12);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.2);
  }
  .status-error {
    background: rgba(239, 68, 68, 0.12);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.2);
  }
  .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: currentColor;
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  /* Main content area */
  .main-header {
    text-align: center;
    padding: 40px 0 32px;
  }
  .main-title {
    font-family: 'Fraunces', serif;
    font-size: 2.4rem;
    font-weight: 300;
    color: #e8eaf0;
    letter-spacing: -0.02em;
    line-height: 1.2;
  }
  .main-title span {
    background: linear-gradient(135deg, #4f7cff 0%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 600;
    font-style: italic;
  }
  .main-subtitle {
    color: #5a6580;
    font-size: 0.95rem;
    margin-top: 8px;
  }

  /* Chat container */
  .chat-container {
    max-width: 760px;
    margin: 0 auto;
  }

  /* Message bubbles */
  .msg-row {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    animation: fadeSlideIn 0.3s ease;
  }
  .msg-row.user { flex-direction: row-reverse; }
  @keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
    margin-top: 2px;
  }
  .avatar.ai {
    background: linear-gradient(135deg, #4f7cff22, #8b5cf622);
    border: 1px solid #4f7cff44;
  }
  .avatar.user {
    background: linear-gradient(135deg, #22c55e22, #16a34a22);
    border: 1px solid #22c55e44;
  }

  .bubble {
    max-width: 78%;
    padding: 14px 18px;
    border-radius: 16px;
    font-size: 0.9rem;
    line-height: 1.65;
  }
  .bubble.ai {
    background: #161b27;
    border: 1px solid #1e2535;
    color: #c8d0e0;
    border-top-left-radius: 4px;
  }
  .bubble.user {
    background: linear-gradient(135deg, #4f7cff15, #8b5cf615);
    border: 1px solid #4f7cff33;
    color: #d4d8e8;
    border-top-right-radius: 4px;
  }
  .bubble strong { color: #e8eaf0; }
  .bubble code {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    background: #0f1117;
    padding: 2px 6px;
    border-radius: 4px;
    color: #8b9fcf;
  }

  /* Source docs expander */
  .source-tag {
    display: inline-block;
    background: #0f1117;
    border: 1px solid #1e2535;
    color: #5a6580;
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    padding: 2px 8px;
    border-radius: 4px;
    margin: 2px 2px 0 0;
  }

  /* Input area */
  .stTextInput > div > div > input,
  .stChatInput textarea {
    background: #161b27 !important;
    border: 1px solid #1e2535 !important;
    border-radius: 12px !important;
    color: #c8d0e0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
  }
  .stChatInput textarea:focus {
    border-color: #4f7cff88 !important;
    box-shadow: 0 0 0 3px rgba(79, 124, 255, 0.1) !important;
  }

  /* Buttons */
  .stButton > button {
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
  }

  /* Suggested questions */
  .suggestion-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin: 24px auto 32px;
    max-width: 760px;
  }
  .suggestion-card {
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 12px 16px;
    cursor: pointer;
    transition: all 0.2s;
    color: #8b9fcf;
    font-size: 0.82rem;
    line-height: 1.4;
  }
  .suggestion-card:hover {
    border-color: #4f7cff55;
    color: #c8d0e0;
    background: #1a2038;
  }
  .suggestion-icon { color: #4f7cff; font-size: 1rem; margin-bottom: 4px; }

  /* Divider */
  hr { border-color: #1e2535 !important; }

  /* Metric cards in sidebar */
  .metric-card {
    background: #0f1117;
    border: 1px solid #1e2535;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
  }
  .metric-label {
    font-size: 0.67rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #3d4a63;
  }
  .metric-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.1rem;
    color: #c8d0e0;
    margin-top: 2px;
  }

  /* File uploader */
  [data-testid="stFileUploader"] {
    background: #0f1117 !important;
    border: 1px dashed #1e2535 !important;
    border-radius: 10px !important;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: #0f1117; }
  ::-webkit-scrollbar-thumb { background: #1e2535; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "messages": [],          # list of {"role": ..., "content": ..., "sources": ...}
        "engine": None,
        "engine_ready": False,
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "doc_count": 0,
        "error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # Brand
    st.markdown("""
    <div class="brand-bar">
      <div class="brand-icon">📋</div>
      <div>
        <div class="brand-name">ACME Policy AI</div>
        <div class="brand-sub">HR Knowledge Base</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Status
    if st.session_state.engine_ready:
        st.markdown(
            '<div class="status-pill status-ready"><div class="dot"></div>Assistant Ready</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill status-error"><div class="dot"></div>Not Initialized</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sidebar-section">Configuration</div>', unsafe_allow_html=True)

    api_key_input = st.text_input(
        "OpenAI API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="AIza...",
        help="Get your key from Google AI Studio (platform.openai.com)",
    )
    if api_key_input:
        st.session_state.api_key = api_key_input

    col1, col2 = st.columns(2)
    with col1:
        init_btn = st.button("⚡ Initialize", use_container_width=True, type="primary")
    with col2:
        rebuild_btn = st.button("🔄 Rebuild DB", use_container_width=True)

    if init_btn or rebuild_btn:
        if not st.session_state.api_key:
            st.error("Please enter your Google API Key.")
        else:
            with st.spinner("Setting up RAG engine…"):
                try:
                    from rag_engine import PolicyRAGEngine
                    force = rebuild_btn
                    engine = PolicyRAGEngine(
                        api_key=st.session_state.api_key,
                        force_rebuild=force,
                    )
                    st.session_state.engine = engine
                    st.session_state.engine_ready = True
                    st.session_state.doc_count = engine.get_doc_count()
                    st.session_state.error = None
                    st.success("Ready!")
                except Exception as e:
                    st.session_state.error = str(e)
                    st.session_state.engine_ready = False
                    st.error(f"Error: {e}")

    # Stats
    if st.session_state.engine_ready:
        st.markdown('<div class="sidebar-section">Knowledge Base</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Vector Chunks</div>
          <div class="metric-value">{st.session_state.doc_count}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Embedding Model</div>
          <div class="metric-value" style="font-size:0.78rem">text-embedding-3-small</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Chat Model</div>
          <div class="metric-value" style="font-size:0.78rem">gpt-4o-mini</div>
        </div>
        """, unsafe_allow_html=True)

    # Upload docs
    st.markdown('<div class="sidebar-section">Upload Policy Docs</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        docs_dir = Path(__file__).parent / "docs"
        docs_dir.mkdir(exist_ok=True)
        saved = []
        for f in uploaded:
            dest = docs_dir / f.name
            dest.write_bytes(f.getbuffer())
            saved.append(f.name)
        st.success(f"Saved {len(saved)} file(s). Click **Rebuild DB** to re-index.")

    # Clear chat
    st.markdown('<div class="sidebar-section">Actions</div>', unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="color:#3d4a63;font-size:0.7rem">Powered by LangChain · Google Gemini · ChromaDB</div>',
        unsafe_allow_html=True,
    )


# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
  <div class="main-title">Your <span>Policy Expert</span>,<br>always available</div>
  <div class="main-subtitle">Ask anything about ACME's policies — leave, benefits, conduct & more</div>
</div>
""", unsafe_allow_html=True)

# Suggested questions (show only when chat is empty)
SUGGESTIONS = [
    ("🏖️", "How many PTO days do I get as a new employee?"),
    ("🤱", "What is the parental leave policy?"),
    ("💰", "When is the annual bonus paid out?"),
    ("🏠", "Can I work from home? What's the WFH policy?"),
    ("💊", "What does ACME's health insurance cover?"),
    ("🔒", "What are the password and security requirements?"),
]

if not st.session_state.messages:
    cols = st.columns(2)
    for i, (icon, q) in enumerate(SUGGESTIONS):
        with cols[i % 2]:
            if st.button(f"{icon} {q}", key=f"sug_{i}", use_container_width=True):
                st.session_state._prefill = q
                st.rerun()

# Handle prefill from suggestion click
prefill = st.session_state.pop("_prefill", None)

# ── Render chat history ───────────────────────────────────────────────────────

for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    sources = msg.get("sources", [])

    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="📋"):
            st.markdown(content)
            if sources:
                with st.expander("📚 Source sections used", expanded=False):
                    for doc in sources:
                        src = doc.metadata.get("source", "policy")
                        snippet = doc.page_content[:200].replace("\n", " ") + "…"
                        st.markdown(f"**`{Path(src).name}`** — {snippet}")


# ── Chat input ────────────────────────────────────────────────────────────────

user_input = st.chat_input(
    placeholder="Ask about leave, benefits, conduct, security policies…",
    disabled=not st.session_state.engine_ready,
) or prefill

if user_input:
    if not st.session_state.engine_ready:
        st.warning("⚠️ Please initialize the assistant first using the sidebar.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Build LangChain message history
    lc_history = []
    for m in st.session_state.messages[:-1]:
        if m["role"] == "user":
            lc_history.append(HumanMessage(content=m["content"]))
        else:
            lc_history.append(AIMessage(content=m["content"]))

    # Stream the response
    with st.chat_message("assistant", avatar="📋"):
        with st.spinner("Searching policies…"):
            try:
                result = st.session_state.engine.query(
                    question=user_input,
                    chat_history=lc_history,
                )
                answer = result["answer"]
                sources = result["source_docs"]
            except Exception as e:
                answer = f"⚠️ Error: {e}\n\nPlease check your API key and try again."
                sources = []

        st.markdown(answer)

        if sources:
            with st.expander("📚 Source sections used", expanded=False):
                for doc in sources:
                    src = doc.metadata.get("source", "policy")
                    snippet = doc.page_content[:200].replace("\n", " ") + "…"
                    st.markdown(f"**`{Path(src).name}`** — {snippet}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
