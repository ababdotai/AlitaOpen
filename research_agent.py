"""
This module implements the ResearchAgent class using LangGraph framework.
It creates an agentic retrieval system that can decompose queries, plan retrieval steps,
and use ReAct pattern to call MCP tools for information gathering from multiple sources.

The agent evaluates the sufficiency of retrieved information and formats results
for use by the ScriptGenerator.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from model_client import ModelClientFactory, ModelClient
from static_mcp.mcp_config_loader import load_mcp_servers_config
from utils import handle_error


class RetrievalState(TypedDict):
    """State definition for the retrieval graph"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    original_query: str
    decomposed_queries: List[str]
    retrieval_plan: List[str]
    retrieved_info: Dict[str, List[Dict[str, Any]]]
    is_sufficient: bool
    formatted_result: str
    iteration_count: int


class ResearchAgent:
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the ResearchAgent with configuration settings.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
                Expected keys:
                    - agent.primary_llm: The primary LLM model name to use.
                    - api.openai_api_key: API key for OpenAI (if using OpenAI models).
                    - api.anthropic_api_key: API key for Anthropic (if using Anthropic models).
                    - Optional: api.temperature: Temperature for API calls (default 0.7).
                    - Optional: api.max_tokens: Maximum tokens for the API response (default 4096).
        """
        try:
            # API configuration
            api_config: Dict[str, Any] = config.get("api", {})
            self.temperature: float = float(api_config.get("temperature", 0.7))
            self.max_tokens: int = int(api_config.get("max_tokens", 4096))
            self.max_iterations: int = 5  # Maximum retrieval iterations
            
            # Initialize model client using factory
            self.model_client: ModelClient = ModelClientFactory.create_client(config)
            
            # Load MCP servers configuration
            self.mcp_servers_config: Dict[str, Any] = load_mcp_servers_config()
            
            # Initialize the graph (will be created when needed)
            self.graph = None
            
            logging.info("ResearchAgent initialized with model: %s", 
                         self.model_client.get_model_name())
            logging.info("Loaded MCP servers: %s", list(self.mcp_servers_config.keys()))
            
        except Exception as e:
            handle_error(e)
    

    
    async def _create_graph(self) -> StateGraph:
        """
        Create and configure the LangGraph with MCP tools.
        
        Returns:
            StateGraph: Configured retrieval graph
        """
        try:
            # Initialize MCP client with servers configuration
            mcp_client = MultiServerMCPClient(self.mcp_servers_config.get("mcpServers", {}))
            mcp_tools = await mcp_client.get_tools()

            logging.info("Available MCP tools: %s", [tool.name for tool in mcp_tools])
            
            # Store tools for use in agent node
            self.mcp_tools = mcp_tools
            
            # Create graph builder
            graph_builder = StateGraph(RetrievalState)
            
            # Add nodes
            graph_builder.add_node("decompose_query", self._decompose_query_node)
            graph_builder.add_node("plan_retrieval", self._plan_retrieval_node)
            graph_builder.add_node("agent", self._agent_node)
            graph_builder.add_node("tools", self._tools_node)
            graph_builder.add_node("evaluate_sufficiency", self._evaluate_sufficiency_node)
            graph_builder.add_node("format_results", self._format_results_node)
            
            # Add edges
            graph_builder.add_edge(START, "decompose_query")
            graph_builder.add_edge("decompose_query", "plan_retrieval")
            graph_builder.add_edge("plan_retrieval", "agent")
            
            # Conditional edge from agent
            graph_builder.add_conditional_edges(
                "agent",
                tools_condition,
                {
                    "tools": "tools",
                    "__end__": "evaluate_sufficiency",
                }
            )
            
            graph_builder.add_edge("tools", "evaluate_sufficiency")
            
            # Conditional edge from evaluate_sufficiency
            graph_builder.add_conditional_edges(
                "evaluate_sufficiency",
                self._should_continue_retrieval,
                {
                    "continue": "agent",
                    "finish": "format_results"
                }
            )
            
            graph_builder.add_edge("format_results", END)
            
            # Compile the graph with proper checkpointer configuration
            graph = graph_builder.compile(checkpointer=MemorySaver())
            graph.name = "Retrieval Agent"
            
            return graph
            
        except Exception as e:
            logging.error("Failed to create retrieval graph: %s", str(e))
            raise
    
    def _decompose_query_node(self, state: RetrievalState) -> Dict[str, Any]:
        """
        Decompose the original query into sub-queries for targeted retrieval.
        
        Args:
            state: Current retrieval state
            
        Returns:
            Dict[str, Any]: Updated state with decomposed queries
        """
        try:
            original_query = state["original_query"]
            
            decomposition_prompt = f"""
            You are a query decomposition expert. Break down the following query into specific sub-queries 
            that can be answered by different information sources (web search, GitHub repositories, PyPI packages).
            
            Original query: {original_query}
            
            Provide 2-4 focused sub-queries that cover different aspects:
            1. General information and documentation
            2. Code examples and implementations
            3. Related libraries and packages
            4. Best practices and tutorials
            
            Return only the sub-queries, one per line, without numbering.
            """
            
            messages = [HumanMessage(content=decomposition_prompt)]
            response = self.model_client.create_completion(
                messages=[{"role": "user", "content": decomposition_prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Parse decomposed queries
            decomposed_queries = [q.strip() for q in response.split('\n') if q.strip()]
            
            logging.info("Decomposed query '%s' into %d sub-queries", 
                         original_query, len(decomposed_queries))
            
            return {
                "decomposed_queries": decomposed_queries,
                "messages": state["messages"] + [AIMessage(content=f"Decomposed into: {decomposed_queries}")]
            }
            
        except Exception as e:
            logging.error("Error in decompose_query_node: %s", str(e))
            return {"decomposed_queries": [state["original_query"]]}
    
    def _plan_retrieval_node(self, state: RetrievalState) -> Dict[str, Any]:
        """
        Plan the retrieval strategy based on decomposed queries.
        
        Args:
            state: Current retrieval state
            
        Returns:
            Dict[str, Any]: Updated state with retrieval plan
        """
        try:
            decomposed_queries = state["decomposed_queries"]
            
            planning_prompt = f"""
            You are a retrieval planning expert. Given these sub-queries, create a retrieval plan 
            that specifies which tools to use for each query:
            
            Sub-queries: {decomposed_queries}
            
            Available tools:
            - web_search_exa: For general web search and documentation
            - search_repositories: For GitHub code repositories
            - search_packages: For PyPI package information
            
            Create a step-by-step plan. For each step, specify:
            1. The tool to use
            2. The specific query to search for
            3. The expected type of information
            
            Return the plan as a simple list of steps, one per line.
            """
            
            response = self.model_client.create_completion(
                messages=[{"role": "user", "content": planning_prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Parse retrieval plan
            retrieval_plan = [step.strip() for step in response.split('\n') if step.strip()]
            
            logging.info("Created retrieval plan with %d steps", len(retrieval_plan))
            logging.info("\n" + "\n".join(retrieval_plan))
            
            return {
                "retrieval_plan": retrieval_plan,
                "retrieved_info": {"web_results": [], "github_repos": [], "pypi_packages": []},
                "iteration_count": 0,
                "messages": state["messages"] + [AIMessage(content=f"Retrieval plan: {retrieval_plan}")]
            }
            
        except Exception as e:
            logging.error("Error in plan_retrieval_node: %s", str(e))
            return {"retrieval_plan": ["Search for general information"]}
    
    def _agent_node(self, state: RetrievalState) -> Dict[str, Any]:
        """
        Agent node that calls the language model with tool binding to decide on next retrieval action.
        This node actually invokes the model and can generate tool calls based on the current state.
        
        Args:
            state: Current retrieval state
            
        Returns:
            Dict[str, Any]: Updated state with model response (potentially with tool calls)
        """
        try:
            original_query = state["original_query"]
            retrieval_plan = state["retrieval_plan"]
            retrieved_info = state["retrieved_info"]
            iteration_count = state["iteration_count"]
            
            # Create context-aware prompt
            context_prompt = f"""
            You are a retrieval agent. Your task is to gather comprehensive information for: {original_query}
            
            Retrieval plan: {retrieval_plan}
            Current iteration: {iteration_count + 1}
            
            Retrieved so far:
            - Web results: {len(retrieved_info.get('web_results', []))}
            - GitHub repos: {len(retrieved_info.get('github_repos', []))}
            - PyPI packages: {len(retrieved_info.get('pypi_packages', []))}
            
            Based on the plan and what you've retrieved, decide what to search for next.
            Use the available tools to gather more specific information.
            
            If you have sufficient information, respond with "SUFFICIENT" to end retrieval.
            """
            
            # Get the current messages and add the new prompt
            messages = state["messages"] + [HumanMessage(content=context_prompt)]
            
            # Check if we should continue or stop based on iteration count and retrieved info
            total_items = (
                len(retrieved_info.get("web_results", [])) +
                len(retrieved_info.get("github_repos", [])) +
                len(retrieved_info.get("pypi_packages", []))
            )
            
            if total_items >= 5 or iteration_count >= self.max_iterations:
                # Signal to end retrieval
                ai_message = AIMessage(content="SUFFICIENT - I have gathered enough information.")
            else:
                # Determine which tool to use based on current needs and available tools
                tool_calls = self._determine_next_tool_calls(original_query, retrieved_info, iteration_count)
                
                if tool_calls:
                    ai_message = AIMessage(
                        content="I need to gather more information using the available tools.",
                        tool_calls=tool_calls
                    )
                else:
                    # No suitable tools found, end retrieval
                    ai_message = AIMessage(content="No suitable tools available, ending retrieval.")
            
            return {
                "messages": messages + [ai_message],
                "iteration_count": iteration_count + 1
            }
            
        except Exception as e:
            logging.error("Error in agent_node: %s", str(e))
            return {
                "messages": state["messages"] + [AIMessage(content=f"Error: {str(e)}")],
                "iteration_count": state.get("iteration_count", 0) + 1
            }
    
    def _determine_next_tool_calls(self, query: str, retrieved_info: dict, iteration_count: int) -> list:
        """
        Determine which tool to call next based on current state and available tools.
        
        Args:
            query: Original search query
            retrieved_info: Currently retrieved information
            iteration_count: Current iteration number
            
        Returns:
            List of tool calls to make
        """
        if not hasattr(self, 'mcp_tools') or not self.mcp_tools:
            return []
        
        # Get available tool names
        available_tools = {tool.name for tool in self.mcp_tools}
        
        # Priority order for different types of searches
        tool_priorities = [
            "web_search_exa",  # Web search first
            "github_search",   # Then GitHub repositories
            "pypi_search",     # Then PyPI packages
        ]
        
        # Determine what type of information we still need
        web_results_count = len(retrieved_info.get("web_results", []))
        github_results_count = len(retrieved_info.get("github_repos", []))
        pypi_results_count = len(retrieved_info.get("pypi_packages", []))
        
        # Choose tool based on what we're missing and what's available
        for tool_name in tool_priorities:
            if tool_name in available_tools:
                # Check if we need this type of information
                if tool_name == "web_search_exa" and web_results_count < 3:
                    return [{
                        "id": f"call_{iteration_count}_{tool_name}",
                        "name": tool_name,
                        "args": {"query": query, "num_results": 5}
                    }]
                elif tool_name == "github_search" and github_results_count < 2:
                    return [{
                        "id": f"call_{iteration_count}_{tool_name}",
                        "name": tool_name,
                        "args": {"query": query, "limit": 3}
                    }]
                elif tool_name == "pypi_search" and pypi_results_count < 2:
                    return [{
                        "id": f"call_{iteration_count}_{tool_name}",
                        "name": tool_name,
                        "args": {"query": query, "limit": 3}
                    }]
        
        # If no specific tool is needed, try the first available tool
        if available_tools:
            first_tool = next(iter(available_tools))
            return [{
                "id": f"call_{iteration_count}_{first_tool}",
                "name": first_tool,
                "args": {"query": query}
            }]
        
        return []
    
    async def _tools_node(self, state: RetrievalState) -> Dict[str, Any]:
        """
        Execute tool calls and update retrieved_info with results.
        
        Args:
            state: Current retrieval state
            
        Returns:
            Dict[str, Any]: Updated state with tool results
        """
        try:
            # Get the last message which should contain tool calls
            messages = state["messages"]
            last_message = messages[-1] if messages else None
            
            if not last_message or not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                logging.warning("No tool calls found in last message")
                return {"messages": messages}
            
            # Execute tool calls using the default ToolNode
            tool_node = ToolNode(self.mcp_tools)
            tool_result = await tool_node.ainvoke(state)
            
            # Parse tool results and update retrieved_info
            retrieved_info = state["retrieved_info"].copy()
            
            # Get the tool result messages
            new_messages = tool_result.get("messages", [])
            tool_messages = [msg for msg in new_messages if hasattr(msg, 'content') and msg not in messages]
            
            for tool_message in tool_messages:
                try:
                    # Parse tool result content
                    if hasattr(tool_message, 'content') and tool_message.content:
                        content = tool_message.content
                        
                        # Try to parse as JSON if it looks like structured data
                        if content.strip().startswith('{') or content.strip().startswith('['):
                            try:
                                result_data = json.loads(content)
                                
                                # Determine result type based on tool name or content structure
                                if isinstance(result_data, list):
                                    # GitHub search results
                                    if any('github.com' in str(item) for item in result_data[:3] if isinstance(item, dict)):
                                        retrieved_info["github_repos"].extend(result_data)
                                        logging.info("Added %d GitHub repositories to retrieved_info", len(result_data))
                                    # PyPI search results
                                    elif any('pypi.org' in str(item) for item in result_data[:3] if isinstance(item, dict)):
                                        retrieved_info["pypi_packages"].extend(result_data)
                                        logging.info("Added %d PyPI packages to retrieved_info", len(result_data))
                                    # Web search results
                                    else:
                                        retrieved_info["web_results"].extend(result_data)
                                        logging.info("Added %d web results to retrieved_info", len(result_data))
                                elif isinstance(result_data, dict):
                                    # Single result - determine type and add to appropriate list
                                    if 'github.com' in str(result_data):
                                        retrieved_info["github_repos"].append(result_data)
                                        logging.info("Added 1 GitHub repository to retrieved_info")
                                    elif 'pypi.org' in str(result_data):
                                        retrieved_info["pypi_packages"].append(result_data)
                                        logging.info("Added 1 PyPI package to retrieved_info")
                                    else:
                                        retrieved_info["web_results"].append(result_data)
                                        logging.info("Added 1 web result to retrieved_info")
                            except json.JSONDecodeError:
                                # If not JSON, treat as text result
                                text_result = {"content": content, "source": "tool_call"}
                                retrieved_info["web_results"].append(text_result)
                                logging.info("Added 1 text result to retrieved_info")
                        else:
                            # Plain text result
                            text_result = {"content": content, "source": "tool_call"}
                            retrieved_info["web_results"].append(text_result)
                            logging.info("Added 1 text result to retrieved_info")
                            
                except Exception as parse_error:
                    logging.error("Error parsing tool result: %s", str(parse_error))
                    continue
            
            return {
                "messages": tool_result.get("messages", messages),
                "retrieved_info": retrieved_info
            }
            
        except Exception as e:
            logging.error("Error in tools_node: %s", str(e))
            return {
                "messages": state["messages"] + [AIMessage(content=f"Tool execution error: {str(e)}")]
            }
    
    def _evaluate_sufficiency_node(self, state: RetrievalState) -> Dict[str, Any]:
        """
        Evaluate whether the retrieved information is sufficient.
        
        Args:
            state: Current retrieval state
            
        Returns:
            Dict[str, Any]: Updated state with sufficiency evaluation
        """
        try:
            original_query = state["original_query"]
            retrieved_info = state["retrieved_info"]
            iteration_count = state["iteration_count"]
            
            # Count total retrieved items
            total_items = (
                len(retrieved_info.get("web_results", [])) +
                len(retrieved_info.get("github_repos", [])) +
                len(retrieved_info.get("pypi_packages", []))
            )
            
            evaluation_prompt = f"""
            Evaluate if the retrieved information is sufficient to answer: {original_query}
            
            Retrieved information summary:
            - Web results: {len(retrieved_info.get('web_results', []))} items
            - GitHub repositories: {len(retrieved_info.get('github_repos', []))} items  
            - PyPI packages: {len(retrieved_info.get('pypi_packages', []))} items
            - Total items: {total_items}
            - Iterations completed: {iteration_count}
            
            Consider:
            1. Do we have enough diverse information sources?
            2. Is the information relevant and comprehensive?
            3. Have we reached the maximum iteration limit ({self.max_iterations})?
            
            Respond with only "SUFFICIENT" or "INSUFFICIENT".
            """
            
            response = self.model_client.create_completion(
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.1,  # Low temperature for consistent evaluation
                max_tokens=50
            )
            
            is_sufficient = (
                "SUFFICIENT" in response.upper() or 
                total_items >= 10 or 
                iteration_count >= self.max_iterations
            )
            
            logging.info("Sufficiency evaluation: %s (total items: %d, iterations: %d)", 
                         "SUFFICIENT" if is_sufficient else "INSUFFICIENT", 
                         total_items, iteration_count)
            
            return {
                "is_sufficient": is_sufficient,
                "messages": state["messages"] + [AIMessage(content=f"Evaluation: {'SUFFICIENT' if is_sufficient else 'INSUFFICIENT'}")]
            }
            
        except Exception as e:
            logging.error("Error in evaluate_sufficiency_node: %s", str(e))
            return {"is_sufficient": True}  # Default to sufficient on error
    
    def _should_continue_retrieval(self, state: RetrievalState) -> str:
        """
        Determine whether to continue retrieval or finish.
        
        Args:
            state: Current retrieval state
            
        Returns:
            str: "continue" or "finish"
        """
        return "finish" if state.get("is_sufficient", True) else "continue"
    
    def _format_results_node(self, state: RetrievalState) -> Dict[str, Any]:
        """
        Format the retrieved information for use by ScriptGenerator.
        
        Args:
            state: Current retrieval state
            
        Returns:
            Dict[str, Any]: Updated state with formatted results
        """
        try:
            retrieved_info = state["retrieved_info"]
            
            # Format according to ScriptGenerator expectations
            formatted_result = {
                "web_results": retrieved_info.get("web_results", []),
                "github_repos": retrieved_info.get("github_repos", []),
                "pypi_packages": retrieved_info.get("pypi_packages", [])
            }
            
            logging.info("Formatted retrieval results: %d web, %d github, %d pypi", 
                         len(formatted_result["web_results"]),
                         len(formatted_result["github_repos"]),
                         len(formatted_result["pypi_packages"]))
            
            return {
                "formatted_result": json.dumps(formatted_result),
                "messages": state["messages"] + [AIMessage(content="Results formatted for ScriptGenerator")]
            }
            
        except Exception as e:
            logging.error("Error in format_results_node: %s", str(e))
            return {"formatted_result": json.dumps({"web_results": [], "github_repos": [], "pypi_packages": []})}
    
    async def retrieve(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Main retrieval method that orchestrates the entire retrieval process.
        
        Args:
            query (str): The search query to process
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Formatted retrieval results compatible with ScriptGenerator
        """
        try:
            logging.info("Starting retrieval for query: %s", query)
            
            # Create graph if not already created
            if self.graph is None:
                self.graph = await self._create_graph()
            
            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=f"Retrieve information for: {query}")],
                "original_query": query,
                "decomposed_queries": [],
                "retrieval_plan": [],
                "retrieved_info": {"web_results": [], "github_repos": [], "pypi_packages": []},
                "is_sufficient": False,
                "formatted_result": "",
                "iteration_count": 0
            }
            
            # Run the retrieval graph with proper configuration
            config = {"configurable": {"thread_id": "retrieval_thread"}}
            final_state = await self.graph.ainvoke(initial_state, config=config)
            
            # Parse and return formatted results
            formatted_result = final_state.get("formatted_result", "{}")
            result = json.loads(formatted_result) if formatted_result else {
                "web_results": [], "github_repos": [], "pypi_packages": []
            }
            
            logging.info("Retrieval completed successfully for query: %s", query)
            return result
            
        except Exception as e:
            logging.error("Error during retrieval for query '%s': %s", query, str(e))
            handle_error(e)
            # Return empty results on error
            return {"web_results": [], "github_repos": [], "pypi_packages": []}
    
    def search(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Synchronous wrapper for the retrieve method to maintain compatibility.
        
        Args:
            query (str): The search query to process
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Formatted retrieval results
        """
        try:
            # Check if there's already a running event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're already in an async context, we can't use run_until_complete
                # This is a limitation - the method should be called from sync context
                logging.warning("search() called from async context, returning empty results")
                return {"web_results": [], "github_repos": [], "pypi_packages": []}
            except RuntimeError:
                # No running loop, we can create one
                pass
            
            # Use asyncio.run() which properly handles loop creation and cleanup
            result = asyncio.run(self.retrieve(query))
            return result
        except Exception as e:
            logging.error("Error in synchronous search wrapper: %s", str(e))
            return {"web_results": [], "github_repos": [], "pypi_packages": []}


if __name__ == "__main__":
    import yaml
    from utils import setup_logging
    
    # 加载配置文件
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        # 使用默认配置
        config = {
            "agent": {"primary_llm": "gpt-4o"},
            "api": {
                "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
                "temperature": 0.7,
                "max_tokens": 4096
            },
            "logging": {
                "level": "INFO",
                "log_file": "logs/research_agent.log"
            }
        }
    
    # Setup logging configuration
    setup_logging(config)
    
    retriever = ResearchAgent(config)
    result = retriever.search("video clipping")
    print(result)
