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


def display_sidebar():
    st.sidebar.title("🏦 Customer Portal")
    st.sidebar.write(st.session_state.chat_session.state)
    st.sidebar.write(st.session_state.chat_session.customer_info)

def main():
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = ChatSession(session_id=str(uuid.uuid4()))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chatbot" not in st.session_state:
        st.session_state.chatbot = KYCChatbot()

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
            <h1>🏦 Customer Service Portal</h1>
            <p>Secure KYC Verification & Account Services</p>
        </div>
        """, unsafe_allow_html=True)

    # Chat interface
    st.subheader("💬 Chat with Customer Service")
    display_sidebar()

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message["content"])

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

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": welcome_response
                })
                st.session_state.initialized = True
                st.rerun()

            except Exception as e:
                st.error(f"Error initializing chatbot: {str(e)}")
                logger.error(f"Initialization error: {str(e)}")

    user_input = st.chat_input("Type your message here...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Processing your request..."):
            try:

                bot_response = st.session_state.event_loop.run_until_complete(
                    st.session_state.chatbot.process_message(
                        st.session_state.chat_session,
                        user_input
                    )
                )

                # Add bot response to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": bot_response
                })

                # # Update conversation history in session
                # st.session_state.chat_session.conversation_history.append({
                #     "user": user_input,
                #     "assistant": bot_response
                # })
                display_sidebar()
                st.rerun()
            except Exception as e:
                st.error(f"Error processing your message: {str(e)}")
                logger.error(f"Message processing error: {str(e)}")

    # st.rerun()

if __name__ == "__main__":
    main()
