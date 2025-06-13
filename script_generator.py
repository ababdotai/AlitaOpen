"""script_generator.py

This module implements the ScriptGenerator class, which takes a tool specification
(from MCPBrainstorm) and a list of external resource items (from WebAgent) to generate a
self-contained executable Python script. The generated script includes not only the
desired functionality but also the environment setup commands (such as conda environment
creation and dependency installation).

The script generation is powered by an LLM API (via OpenAI's ChatCompletion interface),
using a prompt template loaded from the path specified in the configuration (under
agent.script_gen_prompt_template). This module relies on shared utilities for logging,
configuration loading, and error handling.
"""

import os
from openai import OpenAI
import json
import logging
import re
from typing import Any, Dict, List

from utils import read_template, handle_error

class ScriptGenerator:
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the ScriptGenerator with configuration settings.

        Loads the script generation prompt template from the file specified under
        agent.script_gen_prompt_template in the configuration. Also sets up LLM API settings 
        such as the model name, API key, endpoint URL, temperature, and max tokens.

        Args:
            config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
                Expected keys:
                    - agent.script_gen_prompt_template: File path for script generation prompt template.
                    - agent.primary_llm: The primary LLM model name to use.
                    - api.openai_api_key: API key for the OpenAI API.
                    - api.openai_api_url: API endpoint URL.
                    - Optional: api.temperature: Temperature for API calls (default 0.7).
                    - Optional: api.max_tokens: Maximum tokens for the API response (default 300).
        """
        try:
            # Agent configuration
            agent_config: Dict[str, Any] = config.get("agent", {})
            self.prompt_template_path: str = agent_config.get(
                "script_gen_prompt_template", "templates/script_template.txt"
            )
            self.prompt_template: str = read_template(self.prompt_template_path)
            self.model: str = agent_config.get("primary_llm", "gpt-4o")
            
            # API configuration
            api_config: Dict[str, Any] = config.get("api", {})
            self.api_key: str = api_config.get("openai_api_key", os.environ.get("OPENAI_API_KEY"))
            self.api_url: str = api_config.get("openai_api_url", os.environ.get("OPENAI_API_BASE"))
            self.temperature: float = float(api_config.get("temperature", 0.7))
            self.max_tokens: int = int(api_config.get("max_tokens", 300))
            
            # Initialize OpenAI client
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_url
            )

            logging.info("ScriptGenerator initialized with model: %s, prompt template: %s",
                         self.model, self.prompt_template_path)
        except Exception as e:
            handle_error(e)

    def generate_script(self, spec: Dict[str, Any], resources: List[Dict[str, Any]]) -> str:
        """
        Generate a self-contained Python script based on the given specification and external resources.

        This method constructs a prompt by inserting the task description and tool specifications
        from 'spec' along with external resource context (if any) into the loaded prompt template.
        It then calls the LLM API to generate the script, processes the response to remove any extraneous 
        formatting (like markdown code fences), and returns the cleaned Python script.

        Args:
            spec (Dict[str, Any]): Dictionary containing the tool specification.
                Expected keys include:
                    - "task_description": Description of the task.
                    - "mcp_spec": Functional specifications or design details.
            resources (List[Dict[str, Any]]): List of external resource items.
                Each resource is expected to be a dictionary with keys:
                    - "url": URL of the resource.
                    - "title": Title or description of the resource.
                    - "snippet": (Optional) A snippet or excerpt from the resource.

        Returns:
            str: The generated Python script as a string.
        """
        try:
            # Extract task and tool specification details from the spec dictionary.
            task_description: str = spec.get("task_description", "No task description provided.")
            tool_spec: str = spec.get("mcp_spec", "No specific tool specification provided.")
            
            # Combine external resource items into a coherent text block.
            if resources:
                resource_lines: List[str] = []
                for resource in resources:
                    title: str = resource.get("title", "").strip()
                    url: str = resource.get("url", "").strip()
                    snippet: str = resource.get("snippet", "").strip()
                    resource_entry: str = f"Title: {title}\nURL: {url}\nSnippet: {snippet}"
                    resource_lines.append(resource_entry)
                external_context: str = "\n---\n".join(resource_lines)
            else:
                external_context = "No external resources provided."

            # Format the prompt using the loaded template.
            # The template should have placeholders:
            # {task_description}, {tool_spec}, and {external_context}
            prompt: str = self.prompt_template.format(
                task_description=task_description,
                tool_spec=tool_spec,
                external_context=external_context
            )
            logging.debug("Constructed script generation prompt: %s", prompt)

            # Call the OpenAI Chat Completions API to generate the script.
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            response_text: str = response.choices[0].message.content
            logging.info("Received response from LLM for script generation.")

            # Clean the LLM output to remove markdown fences or extraneous formatting.
            final_script: str = self._clean_script(response_text)
            logging.debug("Generated script (truncated): %s", final_script[:200])
            return final_script

        except Exception as e:
            handle_error(e)
            # In case error handling does not raise, return an empty script.
            return ""

    def _clean_script(self, script_text: str) -> str:
        """
        Clean the generated script text by removing markdown code fences, JSON wrappers, and extraneous formatting.

        Args:
            script_text (str): The raw script text returned by the LLM API.

        Returns:
            str: The cleaned script text.
        """
        try:
            cleaned_script = script_text.strip()
            
            # Check if the response is wrapped in JSON format
            if cleaned_script.startswith('{') and cleaned_script.endswith('}'):
                try:
                    # Try to parse as JSON and extract the script field
                    json_response = json.loads(cleaned_script)
                    if isinstance(json_response, dict) and 'script' in json_response:
                        cleaned_script = json_response['script']
                        logging.info("Extracted script from JSON wrapper")
                except json.JSONDecodeError:
                    logging.warning("Failed to parse JSON wrapper, treating as raw script")
            
            # Remove markdown code fences (``` and ```python)
            cleaned_script = re.sub(r"```(?:python)?\s*", "", cleaned_script)
            cleaned_script = re.sub(r"```\s*", "", cleaned_script)
            
            # Remove any leading/trailing whitespace
            return cleaned_script.strip()
        except Exception as e:
            logging.error("Error during script cleanup: %s", str(e))
            return script_text.strip()
