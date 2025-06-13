"""mcp_brainstorm.py

This module implements the MCPBrainstorm class, which uses LLM API calls (via OpenAI)
to analyze a given task and the current system context. It detects capability gaps
and generates a structured specification for a new Model Context Protocol (MCP) if needed.

The class is initialized with a configuration dictionary (typically loaded from config.yaml)
and loads the MCP prompt template from the file specified under agent.mcp_prompt_template.
It uses the OpenAI ChatCompletion API to request a brainstorming response from the LLM.
The response is expected to be in a structured JSON format containing keys such as:
    - "capability_gap": a boolean indicating if a new MCP is required.
    - "mcp_spec": details and specifications for the new MCP if a gap is detected.

If any step fails (prompt formatting, API call, or JSON parsing), the code logs the error and
returns a default dictionary indicating no capability gap and providing an error message.
"""

from openai import OpenAI
import json
import logging
import re
from typing import Any, Dict

from utils import read_template, handle_error

class MCPBrainstorm:
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the MCPBrainstorm instance with configuration settings.
        
        This loads the MCP prompt template from the file specified in the configuration under
        agent.mcp_prompt_template, sets up the LLM API parameters (model name, API key, API URL,
        temperature, and max tokens), and configures the OpenAI library accordingly.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary.
                Expected keys:
                    - agent.mcp_prompt_template: File path for the MCP prompt template.
                    - agent.primary_llm: The primary LLM model name.
                    - api.openai_api_key: OpenAI API key.
                    - api.openai_api_url: OpenAI API endpoint.
                    - Optional: api.temperature and api.max_tokens for API call tuning.
        """
        self.config: Dict[str, Any] = config
        
        # Agent settings
        agent_config: Dict[str, Any] = config.get("agent", {})
        self.prompt_template_path: str = agent_config.get("mcp_prompt_template", "templates/mcp_prompt.txt")
        self.prompt_template: str = read_template(self.prompt_template_path)
        self.model: str = agent_config.get("primary_llm", "gpt-4o")
        
        # API settings
        api_config: Dict[str, Any] = config.get("api", {})
        self.api_key: str = api_config.get("openai_api_key", "<YOUR_API_KEY_HERE>")
        self.api_url: str = api_config.get("openai_api_url", "https://api.openai.com/v1")
        self.temperature: float = float(api_config.get("temperature", 0.7))
        self.max_tokens: int = int(api_config.get("max_tokens", 150))
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_url
        )
        
        logging.info("MCPBrainstorm initialized with model: %s, prompt template: %s", 
                     self.model, self.prompt_template_path)

    def brainstorm(self, task: str, context: str) -> Dict[str, Any]:
        """
        Analyze the task and current context to detect capability gaps and generate an MCP specification.
        
        This method constructs a full prompt by inserting the task and context into the loaded prompt
        template. It then calls the OpenAI ChatCompletion API using the constructed prompt. The LLM's
        response is expected to be in a JSON/dictionary format. The method attempts to parse the response,
        ensuring it contains at least the "capability_gap" key. If a gap is detected and no detailed MCP
        specification is provided, a default message is inserted.
        
        Args:
            task (str): The natural language task description.
            context (str): Current context details (e.g., available tools, system status).
        
        Returns:
            Dict[str, Any]: A dictionary containing the MCP specification.
                Expected keys include:
                    - "capability_gap": bool (True if new MCP is needed, else False)
                    - "mcp_spec": Details for the new MCP if a capability gap is detected.
                    - Optionally, an "error" key describing any issues encountered.
        """
        try:
            # Construct the prompt by substituting task and context into the template.
            # Use safe string replacement to avoid issues with special characters in context
            prompt: str = self.prompt_template.replace("{task}", task).replace("{context}", context)
        except Exception as e:
            logging.error("Error formatting prompt template: %s", str(e))
            return {"capability_gap": False, "mcp_spec": None, "error": "Prompt formatting error"}

        logging.debug("Constructed MCP prompt: %s", prompt)
        
        # Call OpenAI Chat Completions API with the constructed prompt.
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            response_text: str = response.choices[0].message.content
            logging.info("Received response from LLM: %s", response_text)
        except Exception as e:
            error_msg: str = f"LLM API call failed: {str(e)}"
            logging.error(error_msg)
            return {"capability_gap": False, "mcp_spec": None, "error": error_msg}
        
        # Parse the response text to extract a structured MCP specification.
        try:
            # Attempt to parse the LLM response directly as JSON.
            mcp_spec: Dict[str, Any] = json.loads(response_text)
        except json.JSONDecodeError as jde:
            logging.warning("Direct JSON parsing failed: %s", str(jde))
            # Attempt to extract a valid JSON substring using regex.
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                json_str: str = json_match.group(0)
                try:
                    mcp_spec = json.loads(json_str)
                except Exception as e:
                    error_msg = f"Failed to parse JSON from extracted substring: {str(e)}"
                    logging.error(error_msg)
                    return {"capability_gap": False, "mcp_spec": None, "error": error_msg}
            else:
                error_msg = "No JSON structure found in LLM response."
                logging.error(error_msg)
                return {"capability_gap": False, "mcp_spec": None, "error": error_msg}
        except Exception as e:
            error_msg = f"Unexpected error during JSON parsing: {str(e)}"
            logging.error(error_msg)
            return {"capability_gap": False, "mcp_spec": None, "error": error_msg}
        
        # Validate that the parsed specification is a dictionary and contains required keys.
        if not isinstance(mcp_spec, dict):
            error_msg = "Parsed MCP specification is not a dictionary."
            logging.error(error_msg)
            return {"capability_gap": False, "mcp_spec": None, "error": error_msg}
        
        if "capability_gap" not in mcp_spec:
            logging.warning("Key 'capability_gap' not found in MCP specification. Assuming no gap.")
            mcp_spec.setdefault("capability_gap", False)
        
        if mcp_spec.get("capability_gap") and "mcp_spec" not in mcp_spec:
            logging.warning("Capability gap indicated but 'mcp_spec' details are missing. Inserting default details.")
            mcp_spec["mcp_spec"] = "No detailed specification provided."
        
        return mcp_spec
