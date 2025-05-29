import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Any

import streamlit as st

from modules.mcp_client import MCPBedrockClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatState(Enum):
    INITIAL = "initial"
    COLLECTING_NAME = "collecting_name"
    COLLECTING_ACCOUNT = "collecting_account"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    SERVICE_MENU = "service_menu"
    PROCESSING_REQUEST = "processing_request"


@dataclass
class ChatSession:
    session_id: str
    state: ChatState = ChatState.INITIAL
    customer_name: str = None
    last_four_digits: str = None
    conversation_history: List[Dict[str, str]] = None
    customer_info: Dict[str, Any] = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []


class BedRockAgentManager:

    def __init__(self):
        self.mcp_client = MCPBedrockClient(region_name='us-east-1')

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
        # Initialize a separate MCP client for direct interactions
        self.direct_mcp_client = MCPBedrockClient(region_name='us-east-1')
        self.direct_mcp_initialized = False

    async def _initialize_direct_mcp(self):
        """Initialize the direct MCP client if not already initialized"""
        if not self.direct_mcp_initialized:
            servers = [{
                'name': 'customer_info',
                'command': 'python3.11',
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

        if session.state == ChatState.INITIAL:
            return await self._handle_initial(session)

        elif session.state == ChatState.COLLECTING_NAME:
            return await self._handle_name_collection(session, user_input)

        elif session.state == ChatState.VERIFIED:
            return await self._handle_verified_state(session)

        elif session.state == ChatState.SERVICE_MENU:
            return await self._handle_service_request(session, user_input)

        else:
            return "I'm not sure how to help with that. Let's start over."

    async def _handle_initial(self, session: ChatSession) -> str:
        """Handle initial state"""
        session.state = ChatState.COLLECTING_NAME
        return """Welcome to Customer Service! 👋
    I'm here to help you with your account. To get started, I'll need to verify your identity.
    Please provide your full name as it appears on your account."""

    async def _handle_name_collection(self, session: ChatSession, user_input: str) -> str:
        """Handle name collection"""

        agent_config = {
            "servers": [{
                'name': 'customer_info',
                'command': 'python3.11',
                'args': ['mcp_servers/customer_info.py'],
                'description': 'Security scanning and analysis tools'
            }
            ],
            "system_prompt": """
            You are a KYC verification agent. Your task is to verify user identity based on provided information.
            Always respond with valid JSON format.
            if the customer exists, just provide the customer details and verification flag as follow
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
        response = json.loads(response)
        if response.get('verified'):
            session.customer_info = response
            session.customer_name = session.customer_info.get('full_name', session.customer_name)
            session.state = ChatState.VERIFIED
            return await self._handle_verified_state(session)
        else:
            session.state = ChatState.INITIAL
            return "I'm sorry, but I couldn't verify your identity with the information provided. Please start over and ensure your name and account digits are correct."

    async def _handle_verified_state(self, session: ChatSession) -> str:
        """Handle verified state - show service menu"""
        session.state = ChatState.SERVICE_MENU

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
        session.state = ChatState.PROCESSING_REQUEST
        agent_config = {
            "servers": [{
                'name': 'customer_info',
                'command': 'python3.11',
                'args': ['mcp_servers/customer_info.py'],
                'description': 'Security scanning and analysis tools'
            }
            ],
            "system_prompt": f"""
            You are a KYC verification agent. Your task to provide information using the provided customer id {session.customer_info["customer_id"]}
            Use the available tools to provide the information
            """
        }
        response = await self.bedrock_agent_manager.invoke_agent(prompt=user_input,
                                                                 user_id="user_id",
                                                                 session_id=session.session_id,
                                                                 agent_config=agent_config)
        session.state = ChatState.SERVICE_MENU
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
    st.sidebar.title("🏦 Customer Portal")
    st.sidebar.write("**Current State:**")
    st.sidebar.write(st.session_state.chat_session.state.value)

    if st.session_state.chat_session.customer_info:
        st.sidebar.write("**Customer Information:**")
        customer_info_display = st.session_state.chat_session.customer_info.copy()

        # Format customer info nicely for display
        if 'full_name' in customer_info_display:
            st.sidebar.write(f"**Name:** {customer_info_display['full_name']}")
        if 'customer_id' in customer_info_display:
            st.sidebar.write(f"**Customer ID:** {customer_info_display['customer_id']}")
        if 'account_number' in customer_info_display:
            st.sidebar.write(f"**Account:** {customer_info_display['account_number']}")
        if 'account_status' in customer_info_display:
            st.sidebar.write(f"**Status:** {customer_info_display['account_status'].title()}")

        st.sidebar.divider()

        # Quick Actions Section
        st.sidebar.write("**Quick Actions:**")

        # Account Balance Button
        if st.sidebar.button("💳 Check Balance", use_container_width=True):
            customer_id = st.session_state.chat_session.customer_info.get('customer_id')
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
                            st.sidebar.json(balance_info)
                        else:
                            st.sidebar.error(f"Error: {balance_info['error']}")
                    except Exception as e:
                        st.sidebar.error(f"Error: {str(e)}")

        # Credit Limit Increase Button and Form
        if st.sidebar.button("📈 Request Credit Limit Increase", use_container_width=True):
            st.session_state.show_credit_form = True

        # Credit Limit Increase Form
        if getattr(st.session_state, 'show_credit_form', False):
            st.sidebar.write("**Credit Limit Increase Request:**")

            with st.sidebar.form("credit_limit_form"):
                current_limit = st.session_state.chat_session.customer_info.get('credit_limit', 0)
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

                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": request_summary
                                    })
                                else:
                                    st.sidebar.error(f"Error: {result['error']}")
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": f"Error: {result['error']}"
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


def main():
    # Configure page layout for full width
    st.set_page_config(
        page_title="Customer Service Portal",
        page_icon="🏦",
        layout="wide",  # This makes the app use full width
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling and full width usage
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
            background: linear-gradient(90deg, #1f4e79 0%, #2d5aa0 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
            width: 100%;
        }
        .main-header h1 {
            margin: 0;
            font-size: 2.5rem;
        }
        .main-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.2rem;
            opacity: 0.9;
        }
        .chat-column {
            height: 600px;
            overflow-y: auto;
            padding: 1rem;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            background-color: #fafafa;
        }
        .info-column {
            height: 600px;
            overflow-y: auto;
            padding: 1rem;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            background-color: #f8f9fa;
        }
            /* Override Streamlit's default container width */
            .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
                padding-left: 1rem;
                padding-right: 1rem;
                max-width: none;
            }
        </style>
        """, unsafe_allow_html=True)

    # Initialize session state variables
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = ChatSession(session_id=str(uuid.uuid4()))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chatbot" not in st.session_state:
        st.session_state.chatbot = KYCChatbot()

    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    # Initialize event loop once and store it in session state
    if "event_loop" not in st.session_state:
        st.session_state.event_loop = get_or_create_event_loop()

    # Full width header
    st.markdown("""
        <div class="main-header">
            <h1>🏦 Customer Service Portal</h1>
            <p>Secure KYC Verification & Account Services</p>
        </div>
        """, unsafe_allow_html=True)

    # Create two columns for the main content
    col1, col2 = st.columns([2, 1])  # Chat takes 2/3, info panel takes 1/3

    with col1:
        st.subheader("💬 Chat with Customer Service")

        # Chat container with custom styling
        chat_container = st.container()
        with chat_container:
            # Create a div for chat messages with scrolling
            st.markdown('<div class="chat-column">', unsafe_allow_html=True)

            # Display chat messages
            for message in st.session_state.messages:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(message["content"])

            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("📊 Session Information")

        # Information panel with custom styling
        st.markdown('<div class="info-column">', unsafe_allow_html=True)

        # Display session state
        st.write("**Current State:**")
        st.info(st.session_state.chat_session.state.value.replace('_', ' ').title())

        # Display customer information if available
        if st.session_state.chat_session.customer_info:
            st.write("**Customer Information:**")
            customer_info = st.session_state.chat_session.customer_info.copy()

            # Format customer info nicely
            if 'full_name' in customer_info:
                st.write(f"**Name:** {customer_info['full_name']}")
            if 'customer_id' in customer_info:
                st.write(f"**Customer ID:** {customer_info['customer_id']}")
            if 'account_type' in customer_info:
                st.write(f"**Account Type:** {customer_info['account_type']}")
            if 'verified' in customer_info:
                verification_status = "✅ Verified" if customer_info['verified'] else "❌ Not Verified"
                st.write(f"**Status:** {verification_status}")
        else:
            st.write("*No customer information available*")
            st.write("Please complete the verification process to see your account details.")

        # Display conversation count
        st.write(f"**Messages:** {len(st.session_state.messages)}")

        # Display session ID for debugging
        st.write(f"**Session ID:** `{st.session_state.chat_session.session_id[:8]}...`")

        st.markdown('</div>', unsafe_allow_html=True)

    # Initialize chatbot if not done
    if not st.session_state.initialized:
        with st.spinner("Initializing customer service..."):
            try:
                welcome_response = st.session_state.event_loop.run_until_complete(
                    run_async_task(st.session_state.chatbot.process_message(
                        st.session_state.chat_session,
                        ""
                    ))
                )

                if isinstance(welcome_response, dict) and 'error' in welcome_response:
                    st.error(f"Error initializing chatbot: {welcome_response['error']}")
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": welcome_response
                    })
                    st.session_state.initialized = True
                    st.rerun()

            except Exception as e:
                st.error(f"Error initializing chatbot: {str(e)}")
                logger.error(f"Initialization error: {str(e)}")

    # Chat input at the bottom, spanning full width
    user_input = st.chat_input("Type your message here...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Processing your request..."):
            try:
                bot_response = st.session_state.event_loop.run_until_complete(
                    run_async_task(st.session_state.chatbot.process_message(
                        st.session_state.chat_session,
                        user_input
                    ))
                )

                if isinstance(bot_response, dict) and 'error' in bot_response:
                    st.error(f"Error: {bot_response['error']}")
                else:
                    # Add bot response to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": bot_response
                    })

                st.rerun()
            except Exception as e:
                st.error(f"Error processing your message: {str(e)}")
                logger.error(f"Message processing error: {str(e)}")

    # Display sidebar for additional debugging info
    display_sidebar()


if __name__ == "__main__":
    main()