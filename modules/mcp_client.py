# Configure logging
import asyncio
import json
import logging
import os
import subprocess
import time
from typing import Dict, Any, List, Optional

import boto3
from mcp import StdioServerParameters, stdio_client, ClientSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPServerConfig:
    """Configuration for a single MCP server"""

    def __init__(self, name: str, command: str, args: List[str], description: Optional[str] = None):
        self.name = name
        self.command = command
        self.args = args
        self.description = description or f"MCP Server: {name}"

        # Validate command exists
        if not os.path.exists(command):
            raise ValueError(f"Command {command} does not exist for server {name}")

        # Validate script files exist
        for script in args:
            if not os.path.exists(script):
                raise ValueError(f"Server script {script} does not exist for server {name}")


class MCPServerSession:
    """Manages a single MCP server session"""

    def __init__(self, config: MCPServerConfig):
        self.initialization_error = None
        self.config = config
        self.stdio_context = None
        self.session_context = None
        self.mcp_session = None
        self.read = None
        self.write = None
        self.tools = {}
        self.initialized = False

    async def initialize(self):
        """Initialize this server session"""
        try:
            server_params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args
            )

            self.stdio_context = stdio_client(server_params)
            self.read, self.write = await self.stdio_context.__aenter__()

            self.session_context = ClientSession(self.read, self.write)
            self.mcp_session = await self.session_context.__aenter__()

            try:
                await asyncio.wait_for(self.mcp_session.initialize(), timeout=15.0)
            except asyncio.TimeoutError:
                self.initialization_error = "MCP session initialization timed out"
                logger.error(f"Server '{self.config.name}' initialization timed out")
                return False
            except Exception as e:
                self.initialization_error = f"MCP session initialization failed: {str(e)}"
                logger.error(f"Server '{self.config.name}' initialization failed: {e}")
                return False

            # Load tools from this server
            tools_response = await self.mcp_session.list_tools()
            for tool in tools_response.tools:
                # Prefix tool name with server name to avoid conflicts
                tool_key = f"{self.config.name}.{tool.name}"
                self.tools[tool_key] = {
                    'name': tool.name,  # Original tool name for server calls
                    'server_name': self.config.name,
                    'description': f"[{self.config.name}] {tool.description}",
                    'schema': tool.inputSchema
                }

            self.initialized = True
            logger.info(f"Initialized MCP server '{self.config.name}' with {len(self.tools)} tools")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize MCP server '{self.config.name}': {e}")
            return False

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool on this server"""
        try:
            result = await self.mcp_session.call_tool(tool_name, arguments)

            if result.content:
                text_content = []
                for content in result.content:
                    if hasattr(content, 'text'):
                        text_content.append(content.text)
                return '\n'.join(text_content)

            return "Tool executed successfully"

        except Exception as e:
            logger.error(f"Error executing tool {tool_name} on server {self.config.name}: {e}")
            return f"Error: {str(e)}"

    async def cleanup(self):
        """Cleanup this server session"""
        try:
            if self.session_context:
                await self.session_context.__aexit__(None, None, None)
            if self.stdio_context:
                await self.stdio_context.__aexit__(None, None, None)
            self.initialized = False
        except Exception as e:
            logger.error(f"Error cleaning up server {self.config.name}: {e}")


class MCPBedrockClient:
    def __init__(self, region_name: str = 'us-east-2'):
        """Initialize Bedrock client with support for multiple MCP servers"""
        self.mcp_initialized = False
        self.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )
        # self.model_id = 'us.anthropic.claude-3-5-sonnet-20241022-v2:0'
        self.model_id = 'us.anthropic.claude-3-5-sonnet-20241022-v2:0' #Example model ID

        # Multiple server support
        self.server_configs: List[MCPServerConfig] = []
        self.server_sessions: Dict[str, MCPServerSession] = {}
        self.all_tools: Dict[str, Dict] = {}

        self.main_loop = None
        self.system_prompt = None
        self.progress_callback = None

    def add_server(self, name: str, command: str, args: List[str], description: Optional[str] = None):
        """Add an MCP server configuration"""
        try:
            config = MCPServerConfig(name, command, args, description)
            self.server_configs.append(config)
            logger.info(f"Added MCP server configuration: {name}")
        except ValueError as e:
            logger.error(f"Failed to add server {name}: {e}")
            raise

    def add_servers(self, servers: List[Dict[str, Any]]):
        """Add multiple MCP server configurations

        Args:
            servers: List of server configs, each with keys: name, command, args, description (optional)
        """
        for server_config in servers:
            self.add_server(
                name=server_config['name'],
                command=self.which(server_config['command']),
                args=server_config['args'],
                description=server_config.get('description')
            )

    def set_system_prompt(self, system_prompt: str):
        """Set the system prompt to be used for the MCP server"""
        if system_prompt is None:
            raise ValueError("System prompt not set. Please set the system prompt before initializing.")
        self.system_prompt = system_prompt

    def set_progress_callback(self, callback):
        """Set the progress callback to be used for the MCP server"""
        self.progress_callback = callback

    async def initialize_mcp_sessions(self):
        """Initialize all MCP server sessions"""
        try:
            if not self.server_configs:
                raise ValueError("No MCP servers configured. Please add servers before initializing.")

            self.progress_callback("Initializing MCP sessions...")

            # Store reference to current event loop
            self.main_loop = asyncio.get_event_loop()

            # Initialize all server sessions
            success_count = 0
            for config in self.server_configs:
                session = MCPServerSession(config)
                success = await session.initialize()

                if success:
                    self.server_sessions[config.name] = session
                    # Merge tools from this server
                    self.all_tools.update(session.tools)
                    success_count += 1
                    self.progress_callback(f"Initialized server: {config.name}")
                else:
                    logger.error(f"Failed to initialize server: {config.name}")

            if success_count == 0:
                raise Exception("No MCP servers could be initialized")

            self.mcp_initialized = True
            self.progress_callback(f"Successfully initialized {success_count}/{len(self.server_configs)} MCP servers")
            self.progress_callback(f"Total tools available: {len(self.all_tools)}")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize MCP sessions: {e}")
            return False

    async def cleanup_mcp_sessions(self):
        """Cleanup all MCP server sessions"""
        try:
            for session in self.server_sessions.values():
                await session.cleanup()
            self.server_sessions.clear()
            self.all_tools.clear()
            self.mcp_initialized = False
        except Exception as e:
            logger.error(f"Error cleaning up MCP sessions: {e}")

    async def execute_mcp_tool(self, tool_key: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool via MCP using the appropriate server session"""
        try:
            tool_key = tool_key.replace('-', '.')
            if tool_key not in self.all_tools:
                return f"Error: Tool {tool_key} not found"

            tool_info = self.all_tools[tool_key]
            server_name = tool_info['server_name']
            original_tool_name = tool_info['name']

            if server_name not in self.server_sessions:
                return f"Error: Server {server_name} not available"

            session = self.server_sessions[server_name]
            original_key = tool_info.get('original_key', tool_key)
            self.progress_callback(f"Executing tool: {original_key} on server: {server_name}")

            result = await session.execute_tool(original_tool_name, arguments)
            return result

        except Exception as e:
            logger.error(f"Error executing tool {tool_key}: {e}")
            return f"Error: {str(e)}"

    def get_bedrock_tools_config(self) -> Dict[str, Any]:
        """Convert all MCP tools to Bedrock format"""
        bedrock_tools = []

        for tool_key, tool_info in self.all_tools.items():
            # Clean tool name to match Bedrock requirements (alphanumeric, underscore, hyphen only)
            clean_name = tool_key.replace('.', '-')

            bedrock_tool = {
                "name": clean_name,
                "description": tool_info['description'],
                "input_schema": tool_info['schema']
            }
            bedrock_tools.append(bedrock_tool)

        return {
            "tools": bedrock_tools,
            "tool_choice": {"type": "auto"}
        }

    def get_server_summary(self) -> str:
        """Get a summary of all configured servers and their tools"""
        if not self.server_sessions:
            return "No MCP servers initialized"

        summary = []
        summary.append(f"Active MCP Servers ({len(self.server_sessions)}):")

        for server_name, session in self.server_sessions.items():
            server_tools = [tool_key for tool_key in self.all_tools.keys()
                            if self.all_tools[tool_key]['server_name'] == server_name]
            summary.append(f"  • {server_name}: {len(server_tools)} tools")
            for tool_key in server_tools[:3]:  # Show first 3 tools as examples
                original_key = self.all_tools[tool_key].get('original_key', tool_key)
                summary.append(f"    - {original_key}")
            if len(server_tools) > 3:
                summary.append(f"    - ... and {len(server_tools) - 3} more")

        summary.append(f"\nTotal tools available: {len(self.all_tools)}")
        return "\n".join(summary)

    async def query_bedrock_with_mcp(self, user_message: str) -> str:
        """Query Bedrock using all available MCP tools"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_message}]
                }
            ]

            # Add server summary to system prompt if available
            enhanced_system_prompt = self.system_prompt
            if self.server_sessions:
                server_info = self.get_server_summary()
                enhanced_system_prompt += f"\n\nAvailable MCP Tools:\n{server_info}"

            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 512,
                "system": enhanced_system_prompt,
                "messages": messages,
                **self.get_bedrock_tools_config()
            }

            self.progress_callback("Sending prompt to Bedrock for parsing and coming up with action plan...")
            logger.info(f"Sending request to Bedrock: {user_message}")
            logger.info(self.model_id)
            logger.info(body)

            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )

            response_body = json.loads(response['body'].read())
            self.progress_callback(f"Received response from Bedrock")
            return await self.process_response_with_mcp(response_body, messages)

        except Exception as e:
            logger.error(f"Error in Bedrock query: {e}")
            return f"Error: {str(e)}"

    async def process_response_with_mcp(self, response_body: Dict[str, Any],
                                        conversation_history: List[Dict]) -> str:
        """Process Bedrock response using all available MCP tools"""
        max_iterations = 10
        iteration_count = 0
        current_response = response_body

        while iteration_count < max_iterations:
            iteration_count += 1
            content = current_response.get('content', [])

            tool_calls = []
            text_response = ""

            for item in content:
                if item.get('type') == 'text':
                    text_response += item.get('text', '')
                elif item.get('type') == 'tool_use':
                    tool_calls.append(item)

            if not tool_calls:
                if not text_response.strip() and iteration_count > 1:
                    return "Task completed successfully using MCP tools."
                return text_response

            self.progress_callback(f"Iteration {iteration_count}: Executing {len(tool_calls)} MCP tools")

            conversation_history.append({
                "role": "assistant",
                "content": content
            })

            # Execute MCP tools using appropriate server sessions
            tool_results = []

            # Building tool call sequence for logging and debugging as multiline text
            tool_call_sequence = "<br>".join([f"{index+1}. {tool_call.get('name', '')} - {tool_call.get('input', {})}" for index, tool_call in enumerate(tool_calls)])
            self.progress_callback(f"Tool call sequence:<br>{tool_call_sequence}")

            for tool_call in tool_calls:
                tool_name = tool_call.get('name')
                tool_input = tool_call.get('input', {})
                tool_use_id = tool_call.get('id')

                # Execute via appropriate MCP server session
                mcp_result = await self.execute_mcp_tool(tool_name, tool_input)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": [{"type": "text", "text": mcp_result}]
                })

            conversation_history.append({
                "role": "user",
                "content": tool_results
            })

            # Continue conversation
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "system": self.system_prompt,
                "messages": conversation_history,
                **self.get_bedrock_tools_config()
            }

            try:
                self.progress_callback("Continuing conversation with Bedrock...")
                next_response = self.bedrock_client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body)
                )
                current_response = json.loads(next_response['body'].read())

            except Exception as e:
                logger.error(f"Error in iteration {iteration_count}: {e}")
                return f"MCP tools executed, but error in response: {str(e)}"

        return "Maximum iterations reached."

    async def _handle_mcp_request(self, prompt: str, user_id: str) -> str:
        """Handle MCP-enhanced requests"""
        try:
            # Initialize MCP sessions if not already done
            if not self.mcp_initialized:
                success = await self.initialize_mcp_sessions()
                if not success:
                    return "Error: Could not initialize MCP tools. Please check server configurations, syntax or import errors in the MCP server scripts."

            # Query with MCP support
            response = await self.query_bedrock_with_mcp(prompt)
            return response

        except Exception as e:
            error_msg = f"Error in MCP request: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def process_mcp_response(self, prompt, user_id):
        import asyncio
        try:
            # If we're already in the main async context, run directly
            return await self._handle_mcp_request(prompt, user_id)
        except Exception as e:
            logger.error(f"Error in process_mcp_response: {e}")
            return f"Error: {str(e)}"
        # finally:
        #     # Cleanup MCP sessions
        #     await self.cleanup_mcp_sessions()

    def __del__(self):
        """Close all MCP sessions"""
        if self.server_sessions:
            asyncio.run(self.cleanup_mcp_sessions())
            logger.info("All MCP sessions closed.")

    async def close(self):
        """Close all MCP sessions"""
        logger.info("Closing all MCP sessions...")
        await self.cleanup_mcp_sessions()

    def which(self, program):
        try:
            result = subprocess.run(['which', program], capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            raise RuntimeError(f"'{program}' is not found in the system path.")


# Example usage:
"""
# Initialize client
client = MCPBedrockClient()

# Add multiple servers
servers = [
    {
        'name': 'security_tools',
        'command': '/usr/bin/python3',
        'args': ['security_mcp_server.py'],
        'description': 'Security scanning and analysis tools'
    },
    {
        'name': 'data_tools', 
        'command': '/usr/bin/python3',
        'args': ['data_mcp_server.py'],
        'description': 'Data processing and analysis tools'
    },
    {
        'name': 'file_tools',
        'command': '/usr/bin/node',
        'args': ['file_mcp_server.js'],
        'description': 'File system operations'
    }
]

client.add_servers(servers)
client.set_system_prompt("You are a helpful assistant with access to multiple tool servers.")
client.set_progress_callback(lambda msg: print(f"Progress: {msg}"))

# Use the client
response = client.process_mcp_response("Analyze the security of my application", "user123")
print(response)
"""
