import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(__file__))
from agent import run_agent
from tools import get_and_clear_trace
import uuid

# Configure Streamlit page
st.set_page_config(page_title="TeleConnect Retention Agent", page_icon="📞", layout="centered")

st.title("📞 TeleConnect Retention Agent")
st.markdown("Welcome! I am the AI retention assistant. Ask me to look up customers, predict their churn risk, find retention offers, or log interactions.")

# Initialize chat history

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("tools"):
            tools_str = " ➔ ".join(message["tools"])
            st.markdown(f"<p style='font-size:0.8em; color:gray;'><i>Tools used: {tools_str}</i></p>", unsafe_allow_html=True)

# React to user input
if prompt := st.chat_input("E.g., Look up CUST-001 and tell me their churn risk."):
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Analyzing and invoking tools..."):
            try:
                # Clear any lingering trace state
                get_and_clear_trace()
                
                # Call the google-adk agent with the persistent session ID
                response_text = run_agent(prompt, session_id=st.session_state.session_id)
                
                # Fetch the tools executed during this turn
                trace = get_and_clear_trace()
                tools_called = [t["tool"] for t in trace]
                
                st.markdown(response_text)
                
                if tools_called:
                    tools_str = " ➔ ".join(tools_called)
                    st.markdown(f"<p style='font-size:0.8em; color:gray;'><i>Tools used: {tools_str}</i></p>", unsafe_allow_html=True)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "tools": tools_called
                })
                
            except Exception as e:
                error_msg = f"⚠️ An error occurred while running the agent: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
