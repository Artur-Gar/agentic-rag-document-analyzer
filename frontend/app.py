from html import escape
from uuid import uuid4

import streamlit as st

from frontend.client import BackendClient


client = BackendClient()


def _ensure_state() -> None:
    """Initialize Streamlit session state used by the frontend."""
    state = st.session_state
    state.setdefault("session_id", str(uuid4().hex))
    state.setdefault("question_input", "")
    state.setdefault("request_status", "")
    state.setdefault("answer", "")
    state.setdefault("verification_report", "")
    state.setdefault("uploader_nonce", 0)


def _uploader_key() -> str:
    """Return the current file-uploader widget key."""
    return f"uploaded_files_{st.session_state.uploader_nonce}"


def _status_html(message: str, success: bool = False) -> str:
    """Render a status pill with an optional green success tick."""
    if success:
        return (
            "<div class='status-chip success'>"
            "<span class='tick'>&#10003;</span>"
            f"<span>{escape(message)}</span>"
            "</div>"
        )
    return f"<div class='status-chip'>{escape(message)}</div>"


def _render_documents(documents: list[tuple[str, bytes]]) -> str:
    """Render the active documents list with green ticks."""
    if not documents:
        return "<div class='status-chip'>No documents uploaded yet.</div>"

    rows = []
    for name, _ in documents:
        rows.append(
            "<div class='uploaded-file-row'>"
            f"<span class='uploaded-file-name'>{escape(name)}</span>"
            "<span class='tick'>&#10003;</span>"
            "</div>"
        )
    return "<div class='uploaded-file-list'>" + "".join(rows) + "</div>"


def _render_output_card(title: str, body: str, empty_message: str) -> str:
    """Render a readable output panel with stronger text contrast."""
    content = escape(body).replace("\n", "<br>") if body else empty_message
    state_class = "output-card filled" if body else "output-card empty"
    return (
        f"<div class='{state_class}'>"
        f"<div class='output-title'>{escape(title)}</div>"
        f"<div class='output-body'>{content}</div>"
        "</div>"
    )


def _collect_uploaded_files() -> list[tuple[str, bytes]]:
    """Collect browser-uploaded files into name and bytes pairs."""
    uploaded_files = st.session_state.get(_uploader_key()) or []
    return [(uploaded_file.name, uploaded_file.getvalue()) for uploaded_file in uploaded_files]


def _reset_session() -> None:
    """Reset the current frontend session and clear UI state."""
    st.session_state.session_id = str(uuid4().hex)
    st.session_state.question_input = ""
    st.session_state.request_status = "Session reset. Upload files and ask a new question."
    st.session_state.answer = ""
    st.session_state.verification_report = ""
    st.session_state.uploader_nonce += 1


def _submit_question() -> None:
    """Send the current question and active documents to the backend."""
    question_text = st.session_state.question_input.strip()
    documents = _collect_uploaded_files()

    if not question_text:
        st.session_state.request_status = "Enter a question before submitting."
        return
    if not documents:
        st.session_state.request_status = "Upload at least one document before submitting."
        return

    with st.spinner("Generating answer..."):
        try:
            response = client.ask(
                question=question_text,
                documents=documents,
                session_id=st.session_state.session_id,
            )
            st.session_state.answer = response["draft_answer"]
            st.session_state.verification_report = response["verification_report"]
            st.session_state.session_id = response["session_id"]
            st.session_state.request_status = "Answer generated successfully."
        except Exception as exc:
            st.session_state.answer = f"Error: {exc}"
            st.session_state.verification_report = ""
            st.session_state.request_status = "The request failed."


def _backend_status_text() -> str:
    """Return a short frontend-visible backend connectivity message."""
    return "Backend status: connected" if client.health() else "Backend status: unavailable"


def render_app() -> None:
    """Render the Streamlit frontend application."""
    st.set_page_config(page_title="Agentic RAG", layout="wide")
    _ensure_state()

    st.markdown(
        """
        <style>
        div.block-container {
            max-width: 1380px;
            padding-top: 2.4rem;
            padding-bottom: 2rem;
        }
        [data-testid="stHeaderActionElements"] {
            display: none;
        }
        .hero-copy {
            max-width: 780px;
            margin-bottom: 1.25rem;
        }
        .status-chip {
            margin: 8px 0 12px;
            padding: 12px 14px;
            border: 1px solid #3f3f46;
            border-radius: 12px;
            color: #e4e4e7;
            background: rgba(39, 39, 42, 0.75);
        }
        .status-chip.success {
            border-color: rgba(34, 197, 94, 0.55);
            background: rgba(20, 83, 45, 0.28);
            color: #dcfce7;
        }
        .tick {
            color: #4ade80;
            font-weight: 700;
            margin-left: 10px;
        }
        .uploaded-file-list {
            margin-top: 0.75rem;
            border: 1px solid #3f3f46;
            border-radius: 12px;
            overflow: hidden;
        }
        .uploaded-file-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 14px;
            border-bottom: 1px solid #3f3f46;
            background: rgba(39, 39, 42, 0.75);
        }
        .uploaded-file-row:last-child {
            border-bottom: none;
        }
        .uploaded-file-name {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            color: #f4f4f5;
        }
        .output-card {
            border: 1px solid #3f3f46;
            border-radius: 16px;
            background: #1c1d23;
            padding: 16px 18px;
            height: 360px;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .output-card.filled {
            background: #20212a;
            border-color: #4b5563;
        }
        .output-card.empty {
            background: #18181f;
            border-style: dashed;
        }
        .output-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: #f4f4f5;
            margin-bottom: 0.85rem;
        }
        .output-body {
            font-size: 1rem;
            line-height: 1.72;
            color: #f5f7fb;
            white-space: normal;
            word-break: break-word;
            overflow-y: auto;
            padding-right: 6px;
            flex: 1;
        }
        .output-card.empty .output-body {
            color: #a1a1aa;
        }
        div[data-testid="stTextArea"] textarea {
            min-height: 180px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Agentic RAG")
    st.markdown(
        "<div class='hero-copy'>Upload a report, ask a focused question, and inspect both the generated answer and its verification report in one view.</div>",
        unsafe_allow_html=True,
    )

    if client.health():
        st.success(_backend_status_text())
    else:
        st.warning(_backend_status_text())

    input_col, output_col = st.columns([0.9, 1.1], gap="large")

    with input_col:
        st.subheader("Inputs")
        st.file_uploader(
            "Documents",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            key=_uploader_key(),
        )
        st.markdown(_render_documents(_collect_uploaded_files()), unsafe_allow_html=True)

        st.text_area(
            "Question",
            key="question_input",
            height=150,
            placeholder="Ask a focused question about the uploaded document.",
        )

        action_col, reset_col = st.columns(2, gap="small")
        with action_col:
            st.button("Submit", on_click=_submit_question, type="primary", use_container_width=True)
        with reset_col:
            st.button("Reset Session", on_click=_reset_session, use_container_width=True)

        if st.session_state.request_status:
            st.info(st.session_state.request_status)

    with output_col:
        st.subheader("Outputs")
        st.markdown(
            _render_output_card(
                "Answer",
                st.session_state.answer,
                "The generated answer will appear here after submission.",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            _render_output_card(
                "Verification Report",
                st.session_state.verification_report,
                "The verification report will appear here after submission.",
            ),
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    render_app()
