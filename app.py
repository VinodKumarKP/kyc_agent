import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any

import numpy as np
import plotly.express as px
import streamlit as st
import pandas as pd

from modules.mcp_client import MCPBedrockClient
# from apply_theme import apply_theme

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp_client = MCPBedrockClient(region_name='us-east-1')


class ChatState(Enum):
    INITIAL = "initial"
    COLLECTING_NAME = "collecting_name"
    COLLECTING_ACCOUNT = "collecting_account"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    SERVICE_MENU = "service_menu"
    PROCESSING_REQUEST = "processing_request"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    COMPLETED = 'completed'


@dataclass
class ChatSession:
    session_id: str
    state: ChatState = ChatState.INITIAL.name
    customer_name: str = None
    last_four_digits: str = None
    conversation_history: List[Dict[str, str]] = None
    customer_info: Dict[str, Any] = None
    sentiment_history: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []


class BedRockAgentManager:

    def __init__(self):
        self.mcp_client = mcp_client

    async def invoke_agent(
            self,
            prompt: str,
            user_id: str,
            session_id: str,
            agent_config: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Invoke AWS Bedrock agent with streaming response.

        Args:
            prompt: User input prompt
            user_id: Unique user identifier
            session_id: Current session identifier
            agent_name: Name of the agent to invoke
            agent_type: Type of the agent (e.g., 'bedrock', 'mcp')
            agent_config: Optional configuration for the agent

        Returns:
            Optional[str]: Full response from the agent, or None on error
        """
        try:

            self.mcp_client.add_servers(agent_config.get('servers', []))
            self.mcp_client.set_system_prompt(agent_config.get('system_prompt'))
            self.mcp_client.set_progress_callback(logger.info)
            return await self.mcp_client.process_mcp_response(prompt, user_id)

        except Exception as e:
            error_msg = f"Error invoking Bedrock agent: {str(e)}"
            st.error(error_msg)
            logger.error(error_msg)
            # st.session_state.response_queue.put((user_id, error_msg, True))
            return None


class KYCChatbot:

    def __init__(self):
        self.bedrock_agent_manager = BedRockAgentManager()
        self.direct_mcp_client = MCPBedrockClient(region_name='us-east-1')
        self.direct_mcp_initialized = False

    async def _initialize_direct_mcp(self):
        """Initialize the direct MCP client if not already initialized"""
        if not self.direct_mcp_initialized:
            servers = [{
                'name': 'customer_info',
                'command': 'python3.13',
                'args': ['mcp_servers/customer_info.py'],
                'description': 'Customer information and banking tools'
            }]

            self.direct_mcp_client.add_servers(servers)
            self.direct_mcp_client.set_system_prompt("Direct MCP tool execution")
            self.direct_mcp_client.set_progress_callback(lambda x: None)  # Silent callback

            success = await self.direct_mcp_client.initialize_mcp_sessions()
            if success:
                self.direct_mcp_initialized = True
                logger.info("Direct MCP client initialized successfully")
            else:
                logger.error("Failed to initialize direct MCP client")
                return False
        return True

    async def request_credit_limit_increase(self, customer_id: str, requested_amount: float, reason: str = "") -> Dict[
        str, Any]:
        """
        Directly request credit limit increase using MCP client
        """
        try:
            # Initialize MCP client if needed
            if not await self._initialize_direct_mcp():
                return {"error": "Failed to initialize MCP client"}

            # Execute the tool directly
            tool_name = "customer_info-request_credit_limit_increase"
            arguments = {
                "customer_id": customer_id,
                "requested_amount": requested_amount,
                "reason": reason
            }

            result = await self.direct_mcp_client.execute_mcp_tool(tool_name, arguments)

            # Parse the result if it's JSON
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"result": result}

        except Exception as e:
            logger.error(f"Error requesting credit limit increase: {str(e)}")
            return {"error": f"Failed to process request: {str(e)}"}

    async def get_account_balance_direct(self, customer_id: str) -> Dict[str, Any]:
        """
        Directly get account balance using MCP client
        """
        try:
            # Initialize MCP client if needed
            if not await self._initialize_direct_mcp():
                return {"error": "Failed to initialize MCP client"}

            # Execute the tool directly
            tool_name = "customer_info-get_account_balance"
            arguments = {"customer_id": customer_id}

            result = await self.direct_mcp_client.execute_mcp_tool(tool_name, arguments)

            # Parse the result if it's JSON
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"result": result}

        except Exception as e:
            logger.error(f"Error getting account balance: {str(e)}")
            return {"error": f"Failed to get balance: {str(e)}"}

    async def process_message(self, session: ChatSession, user_input: str) -> str:
        """Process user message and return response"""
        try:
            response = await self._handle_state(session, user_input)
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            response = "I apologize, but I encountered an error. Please try again."

        return response

    async def _handle_state(self, session: ChatSession, user_input: str) -> str:
        """Handle message based on current session state"""

        if session.state == ChatState.INITIAL.name:
            return await self._handle_initial(session)

        elif session.state == ChatState.COLLECTING_NAME.name:
            return await self._handle_name_collection(session, user_input)

        elif session.state == ChatState.VERIFIED.name:
            return await self._handle_verified_state(session)

        elif session.state == ChatState.SERVICE_MENU.name:
            return await self._handle_service_request(session, user_input)

        elif session.state == ChatState.SENTIMENT_ANALYSIS.name:
            return await self._handle_sentiment_analysis(session)

        elif session.state == ChatState.COMPLETED.name:
            return await self._handle_summary_request(session)

        else:
            return "I'm not sure how to help with that. Let's start over."

    async def _handle_initial(self, session: ChatSession) -> str:
        """Handle initial state"""
        session.state = ChatState.COLLECTING_NAME.name
        return """Welcome to Customer Service! 👋
    I'm here to help you with your account. To get started, I'll need to verify your identity.
    Please provide your full name as it appears on your account."""

    async def _handle_name_collection(self, session: ChatSession, user_input: str) -> str:
        """Handle name collection"""

        agent_config = {
            "servers": [{
                'name': 'customer_info',
                'command': 'python3.13',
                'args': ['mcp_servers/customer_info.py'],
                'description': 'Security scanning and analysis tools'
            }
            ],
            "system_prompt": """
            You are a KYC verification agent. Your task is to verify user identity based on provided information.
            Also, based on the kyc status, provide the kyc status such as verified or not verified, risk level, and other details.
            Provide kyc risk level as low, medium, or high. kyc score should be between 0 to 100. verification should be percentage between 0 to 100.
            There should be key 'documents' with list of the documents and its metadata and its verification status in kyc dictionary.
            If the customer is high risk customer, provide recommendation to escalate the case to supervisor.
            If the customer is low risk customer, provide recommendation such as waive late fee, reset autopay, or update credit limit.
            Recommendations should be separate key inside kyc dictionary.
            All the keys should be consistent with the customer_info.py tool.
            Always respond with valid JSON format, do not add any additional text.
            if the customer exists, just provide the customer details and verification flag as follow. Put account_details as a dictionary and kyc as a dictionary
            {   
                response
            }
            if customer does not exists, just provide the response as follow
            {
                verified: false
            }
            """
        }
        response = await self.bedrock_agent_manager.invoke_agent(prompt=user_input,
                                                                 user_id="user_id",
                                                                 session_id=session.session_id,
                                                                 agent_config=agent_config)
        print(response)
        response = json.loads(response)
        if len(response.get('account_details', {})) > 0:
            session.customer_info = response
            session.customer_name = session.customer_info.get('full_name', session.customer_name)
            session.state = ChatState.VERIFIED.name
            return await self._handle_verified_state(session)
        else:
            session.state = ChatState.INITIAL.name
            return "I'm sorry, but I couldn't verify your identity with the information provided. Please start over and ensure your name and account digits are correct."

    async def _handle_verified_state(self, session: ChatSession) -> str:
        """Handle verified state - show service menu"""
        session.state = ChatState.SERVICE_MENU.name

        customer_name = session.customer_info.get('full_name', session.customer_name)

        return f"""Great! I've verified your identity, {customer_name}. ✅

    How can I help you today? You can ask me about:

    💳 **Account Balance** - Check your current balance and available credit
    📊 **Transaction History** - View your recent transactions  
    🎯 **Credit Score** - Check your current credit score and details
    📈 **Credit Limit Increase** - Request an increase to your credit limit
    📋 **Account Summary** - Get a complete overview of your account

    Just tell me what you'd like to know about, or ask in your own words!"""

    async def _handle_service_request(self, session, user_input):
        session.state = ChatState.PROCESSING_REQUEST.name
        agent_config = {
            "servers": [{
                'name': 'customer_info',
                'command': 'python3.13',
                'args': ['mcp_servers/customer_info.py'],
                'description': 'Security scanning and analysis tools'
            }
            ],
            "system_prompt": f"""
            You are a KYC verification agent. Your task to provide information using the provided customer id {session.customer_info.get("account_details", {}).get("customer_id")}
            Use the available tools to provide the information. If user says thank you or goodbye, just say goodbye and end the conversation.
            """
        }
        response = await self.bedrock_agent_manager.invoke_agent(prompt=user_input,
                                                                 user_id="user_id",
                                                                 session_id=session.session_id,
                                                                 agent_config=agent_config)
        if 'goodbye' in response.lower():
            session.state = ChatState.COMPLETED.name
        else:
            session.state = ChatState.SERVICE_MENU.name
        return response

    async def _handle_summary_request(self, session):
        session.state = ChatState.PROCESSING_REQUEST.name
        agent_config = {
            "servers": [{
                'name': 'customer_info',
                'command': 'python3.13',
                'args': ['mcp_servers/customer_info.py'],
                'description': 'Security scanning and analysis tools'
            }
            ],
            "system_prompt": """
            You are a KYC verification agent. Your task is to provide summary of the conversation history.
            """
        }
        user_input = f"""
        Please provide a summary of the below conversation history.
        {session.conversation_history}
        """

        response = await self.bedrock_agent_manager.invoke_agent(prompt=user_input,
                                                                 user_id="user_id",
                                                                 session_id=session.session_id,
                                                                 agent_config=agent_config)
        session.state = ChatState.SENTIMENT_ANALYSIS.name
        return response

    async def _handle_sentiment_analysis(self, session):
        session.state = ChatState.PROCESSING_REQUEST.name
        agent_config = {
            "servers": [{
                'name': 'customer_info',
                'command': 'python3.13',
                'args': ['mcp_servers/customer_info.py'],
                'description': 'Security scanning and analysis tools'
            }
            ],
            "system_prompt": """
            You are a KYC verification agent. Your task is analyze the sentiment and provide the sentiment score.
            """
        }
        user_input = f"""
        Please provide a summary of the below conversation history. Analyze the sentiment of the conversation and provide the sentiment score.
        {session.conversation_history}
        Response should be in JSON format only with list of dictionary with keys 'timestamp' and 'sentiment_score'.
        Don't provide any additional text or explanation.
        """

        response = await self.bedrock_agent_manager.invoke_agent(prompt=user_input,
                                                                 user_id="user_id",
                                                                 session_id=session.session_id,
                                                                 agent_config=agent_config)
        session.state = ChatState.INITIAL.name
        return response


def get_or_create_event_loop():
    """Get existing event loop or create a new one if needed"""
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            # If the loop is closed, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    except RuntimeError:
        # No event loop in current thread, create new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


async def run_async_task(coroutine):
    """Helper function to run async tasks safely"""
    try:
        return await coroutine
    except Exception as e:
        logger.error(f"Error in async task: {str(e)}")
        return {"error": f"Failed to execute task: {str(e)}"}


def display_sidebar():
    st.sidebar.markdown("### 👤 Customer Profile")
    # render_analytics_dashboard()
    # st.sidebar.write(st.session_state.chat_session.state)
    # st.sidebar.write(st.session_state.chat_session.customer_info)

    if st.session_state.chat_session.customer_info is not None:
        customer_info = st.session_state.chat_session.customer_info
        st.sidebar.markdown(f"""
        <div class="sidebar-section">
            <h4>{customer_info.get("account_details", {}).get('full_name')}</h4>
            <p><strong>ID:</strong> {customer_info.get("account_details", {}).get('customer_id')}</p>
            <p><strong>Account:</strong> {customer_info.get("account_details", {}).get('account_number')}</p>
            <p><strong>Status:</strong> {customer_info.get("account_details", {}).get('account_status')}</p>
            <p><strong>Credit Limit:</strong> {customer_info.get("account_details", {}).get('credit_limit')}</p>
            <p><strong>Credit Score:</strong> {customer_info.get("account_details", {}).get('credit_score')}</p>
            <p><strong>Credit Score:</strong> {customer_info.get("account_details", {}).get('phone')}</p>
            <p><strong>Credit Score:</strong> {customer_info.get("account_details", {}).get('email')}</p>
        </div>
        """, unsafe_allow_html=True)

        status_class = f"risk-{customer_info.get("kyc", {}).get('risk_level', 'low')}"
        st.sidebar.markdown(f"""
                <div class="metric-card {status_class}">
                    <h5>🛡️ KYC Status</h5>
                    <p><strong>Risk Level:</strong> {customer_info.get("kyc", {}).get('risk_level', 'low').title()}</p>
                    <p><strong>Risk Score:</strong> {customer_info.get("kyc", {}).get('kyc_score', 'low')}/100</p>
                    <p><strong>Verification:</strong> {customer_info.get("kyc", {}).get('verification_percentage', '100%')}</p>
                </div>
                """, unsafe_allow_html=True)

        # Documents
        html = ''
        for document in customer_info.get("kyc", {}).get('documents', []):
            verified = document.get("verified", False)
            verified_class = "verification-badge verified" if verified else "verification-badge failed"
            status =  "Verified" if verified else "Not Verified"
            html = f"""{html}<div class="document-item">
                    <div class="document-info">
                        <div class="document-icon">🆔</div>
                        <div class="document-details">
                            <div class="document-name">{document.get("type")}</div>
                            <div class="document-meta">{document.get("meta_data")}</div>
                        </div>
                    </div>
                    <div class="{verified_class}">{status}</div></div>"""

        st.sidebar.markdown(f"""<div class="document-list">
                    <h5 style="margin-bottom: 16px; color: #1e293b; font-size: 16px;">Verification Documents</h5>
                    {html}</div>""", unsafe_allow_html=True)

        st.sidebar.divider()

        # Quick Actions Section
        st.sidebar.write("**Quick Actions:**")
        col1, col2 = st.sidebar.columns(2)

        # Account Balance Button
        with col1:
            if st.sidebar.button("💳 Check Balance", use_container_width=True):
                customer_id = st.session_state.chat_session.customer_info.get("account_details", {}).get('customer_id')
                if customer_id:
                    with st.spinner("Getting account balance..."):
                        # Use the shared event loop
                        loop = st.session_state.event_loop
                        try:
                            balance_info = loop.run_until_complete(
                                run_async_task(st.session_state.chatbot.get_account_balance_direct(customer_id))
                            )

                            if 'error' not in balance_info:
                                st.sidebar.success("Balance Retrieved!")
                                request_summary = f"""Balance information retrieved:
    - Current Limit: ${balance_info.get('credit_limit'):,.2f}
    - Current Balance: ${balance_info.get('current_balance'):,.2f}
    - Available Credit: ${balance_info.get('available_credit'):,.2f}
    - Utilization Percentage: {balance_info.get('utilization_percentage', 0):.2f}%
    """

                                st.session_state.chat_session.conversation_history.append({
                                    "role": "assistant",
                                    "content": request_summary,
                                    "timestamp": datetime.now().isoformat()
                                })
                            else:
                                st.sidebar.error(f"Error: {balance_info['error']}")
                        except Exception as e:
                            st.sidebar.error(f"Error: {str(e)}")

            if st.button("🚫 Waive Fee", use_container_width=True):
                st.sidebar.success("Waived late fee for customer.")

            if st.sidebar.button("📈 Request Credit Limit Increase", use_container_width=True):
                st.session_state.show_credit_form = True

        with col2:
            # Credit Limit Increase Button and Form

            if st.button("🚨 Escalate to Supervisor", use_container_width=True):
                st.sidebar.success("Escalated to Supervisor")

            if st.button("🔄 Reset AutoPay", use_container_width=True):
                st.sidebar.success("Auto pay successfully reset for customer.")

        # Credit Limit Increase Form
        if getattr(st.session_state, 'show_credit_form', False):
            st.sidebar.write("**Credit Limit Increase Request:**")

            with st.sidebar.form("credit_limit_form"):
                current_limit = st.session_state.chat_session.customer_info.get("account_details", {}).get('credit_limit', 0)
                st.write(f"Current Limit: ${current_limit:,.2f}")

                requested_amount = st.number_input(
                    "Requested New Limit ($)",
                    min_value=float(current_limit + 100),
                    max_value=float(current_limit * 3),
                    value=float(current_limit * 1.5),
                    step=100.0,
                    format="%.2f"
                )

                reason = st.text_area(
                    "Reason for Increase",
                    placeholder="e.g., Increased income, major purchase planned, etc.",
                    max_chars=200
                )

                col1, col2 = st.columns(2)
                with col1:
                    submit_request = st.form_submit_button("Submit Request", use_container_width=True)
                with col2:
                    cancel_request = st.form_submit_button("Cancel", use_container_width=True)

                if submit_request:
                    customer_id = st.session_state.chat_session.customer_info.get('customer_id')
                    if customer_id and requested_amount > current_limit:
                        with st.spinner("Processing credit limit request..."):
                            # Use the shared event loop
                            loop = st.session_state.event_loop
                            try:
                                result = loop.run_until_complete(
                                    run_async_task(st.session_state.chatbot.request_credit_limit_increase(
                                        customer_id, requested_amount, reason
                                    ))
                                )

                                if 'error' not in result:
                                    st.sidebar.success("Request Submitted Successfully!")
                                    st.sidebar.json(result)

                                    # Add to chat history
                                    request_summary = f"""Credit limit increase request submitted:
- Current Limit: ${current_limit:,.2f}
- Requested Limit: ${requested_amount:,.2f}
- Status: {result.get('status', 'Unknown')}
- Request ID: {result.get('request_id', 'N/A')}
- Processing Time: {result.get('estimated_processing_days', 'N/A')} days"""

                                    st.session_state.chat_session.conversation_history.append({
                                        "role": "assistant",
                                        "content": request_summary,
                                        "timestamp": datetime.now().isoformat()
                                    })
                                else:
                                    st.sidebar.error(f"Error: {result['error']}")
                                    st.session_state.chat_session.conversation_history.append({
                                        "role": "assistant",
                                        "content": f"Error: {result['error'],}",
                                        "timestamp": datetime.now().isoformat()
                                    })
                            except Exception as e:
                                st.sidebar.error(f"Error: {str(e)}")

                        st.session_state.show_credit_form = False
                        st.rerun()
                    else:
                        st.sidebar.error("Invalid request amount")

                if cancel_request:
                    st.session_state.show_credit_form = False
                    st.rerun()

        st.sidebar.divider()



def load_css():
    directory_name = os.path.dirname(__file__)
    css_file_path = os.path.join(directory_name, "style.css")
    with open(css_file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_call_controls():
    """Render call control buttons"""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("🎤 Start Call - Happy Customer", disabled=st.session_state.call_active, key="start_call_happy_customer"):
            st.session_state.call_active = True
            st.session_state.call_start_time = datetime.now()
            st.session_state.fake_conversation = [
                {
                    "message": "Hi, my name is John Doe, and my account number ends with 1234.",
                    "parsed": False
                },
                {
                    "message": "I need to increase my credit limit to 6000",
                    "parsed": False
                },
                {
                    "message": "Thank you for your help!",
                    "parsed": False
                }
            ]
            st.rerun()
    with col2:
        if st.button("🎤 Start Call - Sad Customer", disabled=st.session_state.call_active,
                     key="start_call_frustrated_customer"):
            st.session_state.call_active = True
            st.session_state.call_start_time = datetime.now()
            st.session_state.fake_conversation = [
                {
                    "message": "Hi, my name is Jane Smith, and my account number ends with 5678.",
                    "parsed": False
                },
                {
                    "message": "I need to request a credit limit increase to 8000",
                    "parsed": False
                },
                {
                    "message": "I am very frustrated by the service. Good bye!",
                    "parsed": False
                }
            ]

    with col3:
        if st.button("⏹️ End Call", disabled=not st.session_state.call_active, key="end_call"):
            st.rerun()

    with col4:
        if st.button("🔄 Clear Transcript", key="clear_transcript"):
            st.session_state.chat_session.conversation_history = []
            st.session_state.chat_session.state = ChatState.COLLECTING_NAME.name
            st.session_state.chat_session.customer_info = None
            welcome_response = """Welcome to Customer Service! 👋
    I'm here to help you with your account. To get started, I'll need to verify your identity.
    Please provide your full name as it appears on your account."""
            st.session_state.chat_session.conversation_history.append({
                "role": "assistant",
                "content": welcome_response,
                "timestamp": datetime.now().isoformat()
            })
            # Reset chatbot instance
            st.rerun()

    if st.session_state.call_active:
        st.markdown('<div class="live-indicator"></div><strong>LIVE CALL</strong>', unsafe_allow_html=True)
    else:
        st.markdown("📴 **Call Inactive**")

    with col5:
        # Call timer
        if st.session_state.call_active and st.session_state.call_start_time:
            elapsed = datetime.now() - st.session_state.call_start_time
            minutes = int(elapsed.total_seconds() // 60)
            seconds = int(elapsed.total_seconds() % 60)
            st.markdown(f"**⏱️ {minutes:02d}:{seconds:02d}**")

def render_analytics_dashboard():
    """Render analytics dashboard"""
    if st.sidebar.checkbox("📊 Show Analytics Dashboard"):
        st.markdown("### 📊 Call Analytics Dashboard")

        # Mock analytics data
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Today's Calls", "127", "↑ 8%")

        with col2:
            st.metric("Avg Resolution Time", "4:32", "↓ 12%")

        with col3:
            st.metric("Customer Satisfaction", "4.6/5", "↑ 0.2")

        with col4:
            st.metric("High Risk Alerts", "3", "↓ 2")

        # Call volume chart
        hours = list(range(9, 18))
        call_volume = np.random.poisson(15, len(hours))

        fig = px.bar(
            x=hours,
            y=call_volume,
            title="Call Volume by Hour",
            labels={'x': 'Hour', 'y': 'Number of Calls'}
        )
        st.plotly_chart(fig, use_container_width=True)


def main():
    # apply_theme()
    st.set_page_config(
        page_title="LYC Call Intelligence Dashboard",
        page_icon="📞",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    load_css()

    if "fake_conversation" not in st.session_state:
        st.session_state.fake_conversation = []


    if "chat_session" not in st.session_state:
        st.session_state.chat_session = ChatSession(session_id=str(uuid.uuid4()))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chatbot" not in st.session_state:
        st.session_state.chatbot = KYCChatbot()

    if "call_active" not in st.session_state:
        st.session_state.call_active = False

    if "call_start_time" not in st.session_state:
        st.session_state.call_start_time = None

    if "transcript_history" not in st.session_state:
        st.session_state.transcript_history = []

    if "auto_queries" not in st.session_state:
        st.session_state.auto_queries = []

    if "initialized" not in st.session_state:
        st.session_state.initialized = False

        # Initialize event loop once and reuse it
        if "event_loop" not in st.session_state:
            try:
                # Try to get existing loop first
                st.session_state.event_loop = asyncio.get_event_loop()
            except RuntimeError:
                # If no loop exists, create a new one
                st.session_state.event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(st.session_state.event_loop)

    st.markdown("""
        <div class="main-header">
            <h1>📞 LYC Call Intelligence Dashboard</h1>
            <p>Real-time conversation analysis & proactive customer assistance</p>
        </div>
        """, unsafe_allow_html=True)

    # Chat interface
    st.subheader("💬 Chat with Customer Service")
    display_sidebar()
    render_call_controls()
    # render_quick_actions()

    # Display chat messages
    col1, col2 = st.columns([4, 1])

    with col1:
        chat_container = st.container(height=500, key="chat_container")
        render_conversation_history(chat_container)

        if not st.session_state.initialized:
            with st.spinner("Initializing customer service..."):
                try:
                    # Create event loop for async operation
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    welcome_response = st.session_state.event_loop.run_until_complete(
                        st.session_state.chatbot.process_message(
                            st.session_state.chat_session,
                            ""
                        )
                    )

                    loop.close()

                    st.session_state.chat_session.conversation_history.append({
                        "role": "assistant",
                        "content": welcome_response,
                        "timestamp": datetime.now().isoformat()
                    })
                    st.session_state.initialized = True
                    st.rerun()

                except Exception as e:
                    st.error(f"Error initializing chatbot: {str(e)}")
                    logger.error(f"Initialization error: {str(e)}")

        user_input = st.chat_input("Type your message here...")

        if user_input:
            parse_input(user_input, chat_container)
            st.rerun()


        if st.session_state.call_active:
            for index, message in enumerate(st.session_state.fake_conversation):
                if not message["parsed"]:
                    with chat_container:
                        bot_response = parse_input(message["message"], chat_container)
                        message["parsed"] = True
                        with st.chat_message("assistant"):
                            st.write(bot_response)
                        if index == 0:
                            display_sidebar()
                            render_side_col(col2)
            st.session_state.call_active = False
            st.rerun()

        render_side_col(col2)

    # st.rerun()

def render_insights_panel():
    """Render proactive insights panel"""


    if st.session_state.chat_session.customer_info is None:
        st.markdown("### 💡 Proactive Insights")
        st.info("Load a customer to see proactive insights.")
        return

    if st.session_state.chat_session.customer_info.get("kyc", {}).get("risk_level") == "high":
        st.markdown("""
                    <div class="compliance-alert">
                        <h4>⚠️ High Risk Customer Alert</h4>
                        <p>This customer requires immediate attention and enhanced due diligence.</p>
                    </div>
                    """, unsafe_allow_html=True)

    # Sentiment tracking
    if st.session_state.chat_session.sentiment_history:
        st.markdown("#### Customer Sentiment")

        # Create sentiment chart
        sentiment_df = pd.DataFrame(st.session_state.chat_session.sentiment_history)
        if not sentiment_df.empty:
            fig = px.line(
                sentiment_df,
                x='timestamp',
                y='sentiment_score',
                title="Real-time Sentiment Analysis",
                range_y=[0, 1]
            )
            fig.update_layout(height=200)
            st.plotly_chart(fig, use_container_width=True)

            # Current sentiment
            current_sentiment = sentiment_df['sentiment_score'].iloc[-1]
            if current_sentiment < 0.4:
                st.error("😰 Customer appears frustrated")
            elif current_sentiment < 0.6:
                st.warning("😐 Customer sentiment is neutral")
            else:
                st.success("😊 Customer appears satisfied")

def render_side_col(col2):
    with col2:
        render_insights_panel()

        customer_info = st.session_state.chat_session.customer_info
        if customer_info is not None:
            status_class = f"risk-{customer_info.get("kyc", {}).get('risk_level', 'low')}"
            recommendation = ''
            for i, rec in enumerate(customer_info.get('kyc', {}).get('recommendations'), 1):
                recommendation = f"{recommendation}{i}. {rec}<br>"
            st.markdown(f"""
                            <div class="metric-card {status_class}">
                            <h5>🛡️#### 🎯 Recommended Actions</h5>
                            <p>{recommendation}</p>
                            </div>""",
                        unsafe_allow_html=True)

def parse_input(user_input, chat_container):
    with chat_container:
        with st.chat_message("user"):
            st.write(user_input)
    st.session_state.chat_session.conversation_history.append(
        {"role": "user",
         "content": user_input,
         "timestamp": datetime.now().isoformat()
         }
    )

    with st.spinner("Processing your request..."):
        try:

            bot_response = st.session_state.event_loop.run_until_complete(
                st.session_state.chatbot.process_message(
                    st.session_state.chat_session,
                    user_input
                )
            )

            # Add bot response to chat
            st.session_state.chat_session.conversation_history.append({
                "role": "assistant",
                "content": bot_response,
                "timestamp": datetime.now().isoformat()
            })

            if st.session_state.chat_session.state == ChatState.COMPLETED.name:
                bot_response = st.session_state.event_loop.run_until_complete(
                    st.session_state.chatbot.process_message(
                        st.session_state.chat_session,
                        user_input
                    )
                )

                # Add bot response to chat
                st.session_state.chat_session.conversation_history.append({
                    "role": "assistant",
                    "content": bot_response,
                    "timestamp": datetime.now().isoformat()
                })

                bot_response = st.session_state.event_loop.run_until_complete(
                    st.session_state.chatbot.process_message(
                        st.session_state.chat_session,
                        user_input
                    )
                )

                st.session_state.chat_session.sentiment_history = json.loads(bot_response)
                print(bot_response)

            # # Update conversation history in session
            # st.session_state.chat_session.conversation_history.append({
            #     "user": user_input,
            #     "assistant": bot_response
            # })
            # display_sidebar()
            return bot_response


        except Exception as e:
            st.error(f"Error processing your message: {str(e)}")
            logger.error(f"Message processing error: {str(e)}")


def render_conversation_history(chat_container):
    with chat_container:
        for message in st.session_state.chat_session.conversation_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message["content"])

if __name__ == "__main__":
    main()
