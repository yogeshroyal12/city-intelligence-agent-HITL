
import streamlit as st
import uuid

from langgraph.types import Command
from create_agent import agent


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="City Intelligence AI",
    page_icon="🌍",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 2rem;
    max-width: 950px;
}

[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 15px;
}

.header-title {
    font-size: 34px;
    font-weight: 700;
    color: #FFFFFF;
}

.header-subtitle {
    color: #A0A0A0;
    font-size: 15px;
}

.tool-card {
    background-color: #202938;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #374151;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "pending_interrupts" not in st.session_state:
    st.session_state.pending_interrupts = []

if "pending_count" not in st.session_state:
    st.session_state.pending_count = 0


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_agent_state(result):
    """Extract agent state and pending interrupts."""

    state = result.value

    interrupts = list(result.interrupts)

    return state, interrupts


def display_result(result):
    """Handle completed or interrupted agent execution."""

    state, interrupts = get_agent_state(result)

    if interrupts:

        st.session_state.pending_interrupts = interrupts

        st.session_state.pending_count = sum(
            len(interrupt.value.get("action_requests", []))
            for interrupt in interrupts
        )

        st.rerun()

    else:

        final_message = state["messages"][-1].content

        if isinstance(final_message, str):
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_message
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": str(final_message)
            })


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="header-title">🌍 City Intelligence AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="header-subtitle">'
    'Your intelligent assistant for real-time weather '
    'and the latest city news.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🌍 City Agent")

    st.caption("Powered by Groq + LangChain")

    st.divider()

    st.subheader("Available Tools")

    st.markdown("""
    🌦️ **Weather Information**

    Get current weather using OpenWeatherMap.

    📰 **Latest News**

    Search city-related news using Tavily.

    🤖 **AI Agent**

    Intelligent tool selection using Groq.
    """)

    st.divider()

    st.subheader("Human Approval")

    st.caption(
        "The agent requests your permission "
        "before executing external tools."
    )

    if st.button("🗑️ Clear Conversation", use_container_width=True):

        st.session_state.messages = []
        st.session_state.pending_interrupts = []
        st.session_state.pending_count = 0
        st.session_state.thread_id = str(uuid.uuid4())

        st.rerun()


# ==========================================
# CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==========================================
# HUMAN-IN-THE-LOOP APPROVAL
# ==========================================

if st.session_state.pending_interrupts:

    st.warning("⚠️ Agent requires your approval before tool execution.")

    action_requests = []

    for interrupt in st.session_state.pending_interrupts:

        action_requests.extend(
            interrupt.value.get("action_requests", [])
        )

    for index, action in enumerate(action_requests):

        with st.container(border=True):

            st.markdown(
                f"### 🔧 Tool Request: {action['name']}"
            )

            st.markdown("**Tool Arguments:**")

            st.json(action.get("args", {}))

    col1, col2 = st.columns(2)

    with col1:

        approve = st.button(
            "✅ Approve",
            type="primary",
            use_container_width=True
        )

    with col2:

        reject = st.button(
            "❌ Reject",
            use_container_width=True
        )

    if approve or reject:

        decisions = []

        for _ in action_requests:

            if approve:

                decisions.append({
                    "type": "approve"
                })

            else:

                decisions.append({
                    "type": "reject",
                    "message": (
                        "The user rejected this tool execution. "
                        "Do not retry this tool call."
                    )
                })

        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id
            }
        }

        try:

            with st.spinner("Agent is processing your decision..."):

                result = agent.invoke(
                    Command(
                        resume={
                            "decisions": decisions
                        }
                    ),
                    config=config,
                    version="v2"
                )

                st.session_state.pending_interrupts = []
                st.session_state.pending_count = 0

                display_result(result)

        except Exception as e:

            st.error(f"Agent error: {str(e)}")

        st.rerun()


# ==========================================
# CHAT INPUT
# ==========================================

if not st.session_state.pending_interrupts:

    user_input = st.chat_input(
        "Ask about weather or latest city news..."
    )

    if user_input:

        # Store user message

        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # Display user message

        with st.chat_message("user"):
            st.markdown(user_input)

        # Invoke agent

        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id
            }
        }

        try:

            with st.chat_message("assistant"):

                with st.spinner("🤖 City Agent is thinking..."):

                    result = agent.invoke(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": user_input
                                }
                            ]
                        },
                        config=config,
                        version="v2"
                    )

                    state, interrupts = get_agent_state(result)

                    if interrupts:

                        st.session_state.pending_interrupts = interrupts

                        st.session_state.pending_count = sum(
                            len(i.value.get("action_requests", []))
                            for i in interrupts
                        )

                        st.info(
                            "Your approval is required "
                            "before executing the requested tool."
                        )

                    else:

                        final_message = state["messages"][-1].content

                        if not isinstance(final_message, str):
                            final_message = str(final_message)

                        st.markdown(final_message)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": final_message
                        })

        except Exception as e:

            st.error(f"Something went wrong: {str(e)}")

        st.rerun()


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "City Intelligence AI | Groq + LangChain + Tavily + OpenWeatherMap"
)


import streamlit as st

st.set_page_config(
    page_title="City Intelligence Agent",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 City Intelligence Agent")

st.success("Streamlit is working successfully!")

st.write("Welcome to your AI-powered city assistant.")

city = st.text_input("Enter a city name")

if st.button("Search"):
    if city:
        st.write(f"You entered: {city}")
    else:
        st.warning("Please enter a city name.")

