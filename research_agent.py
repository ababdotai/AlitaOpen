"""
This module implements the RetrieverAgent class using LangGraph framework.
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


class RetrieverAgent:
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the RetrieverAgent with configuration settings.
        
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
            
            logging.info("RetrieverAgent initialized with model: %s", 
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
            
            # Create graph builder
            graph_builder = StateGraph(RetrievalState)
            
            # Add nodes
            graph_builder.add_node("decompose_query", self._decompose_query_node)
            graph_builder.add_node("plan_retrieval", self._plan_retrieval_node)
            graph_builder.add_node("call_model", self._call_model_node)
            graph_builder.add_node("tools", ToolNode(mcp_tools))
            graph_builder.add_node("evaluate_sufficiency", self._evaluate_sufficiency_node)
            graph_builder.add_node("format_results", self._format_results_node)
            
            # Add edges
            graph_builder.add_edge(START, "decompose_query")
            graph_builder.add_edge("decompose_query", "plan_retrieval")
            graph_builder.add_edge("plan_retrieval", "call_model")
            
            # Conditional edge from call_model
            graph_builder.add_conditional_edges(
                "call_model",
                tools_condition,
                {
                    "tools": "tools",
                    END: "evaluate_sufficiency",
                }
            )
            
            graph_builder.add_edge("tools", "evaluate_sufficiency")
            
            # Conditional edge from evaluate_sufficiency
            graph_builder.add_conditional_edges(
                "evaluate_sufficiency",
                self._should_continue_retrieval,
                {
                    "continue": "call_model",
                    "finish": "format_results"
                }
            )
            
            graph_builder.add_edge("format_results", END)
            
            # Compile the graph
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
            
            return {
                "retrieval_plan": retrieval_plan,
                "retrieved_info": {"web_results": [], "github_repos": [], "pypi_packages": []},
                "iteration_count": 0,
                "messages": state["messages"] + [AIMessage(content=f"Retrieval plan: {retrieval_plan}")]
            }
            
        except Exception as e:
            logging.error("Error in plan_retrieval_node: %s", str(e))
            return {"retrieval_plan": ["Search for general information"]}
    
    def _call_model_node(self, state: RetrievalState) -> Dict[str, Any]:
        """
        Call the language model to decide on next retrieval action.
        
        Args:
            state: Current retrieval state
            
        Returns:
            Dict[str, Any]: Updated state with model response
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
            
            messages = state["messages"] + [HumanMessage(content=context_prompt)]
            
            # This will be handled by tools_condition to decide whether to use tools or end
            return {
                "messages": messages,
                "iteration_count": iteration_count + 1
            }
            
        except Exception as e:
            logging.error("Error in call_model_node: %s", str(e))
            return {"messages": state["messages"]}
    
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
            
            # Run the retrieval graph
            final_state = await self.graph.ainvoke(initial_state)
            
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
            # Run the async retrieve method
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.retrieve(query))
            loop.close()
            return result
        except Exception as e:
            logging.error("Error in synchronous search wrapper: %s", str(e))
            return {"web_results": [], "github_repos": [], "pypi_packages": []}


if __name__ == "__main__":
    import yaml
    
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
            }
        }
    
    retriever = RetrieverAgent(config)
    result = retriever.search("video clipping")
    print(result)
