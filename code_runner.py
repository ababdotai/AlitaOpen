"""code_runner.py

This module defines the CodeRunner class. CodeRunner is responsible for executing a generated Python script
in an isolated Conda environment. It leverages EnvironmentManager to create and configure the environment
(e.g., install required dependencies) and then uses subprocess calls to execute the script. All outputs
(stdout and stderr) are captured and logged, with detailed error handling to support iterative refinement
in the overall CodeReAct loop.
"""

import os
import subprocess
import tempfile
import logging
from typing import Any, Dict, List, Tuple

from env_manager import EnvironmentManager
from utils import handle_error


class CodeRunner:
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the CodeRunner with configuration settings.

        Args:
            config (Dict[str, Any]): Global configuration dictionary loaded from config.yaml.
                Expected keys include environment configurations and optional execution timeout.
        """
        self.config: Dict[str, Any] = config
        # Instantiate EnvironmentManager with provided configuration.
        self.env_manager: EnvironmentManager = EnvironmentManager(config)
        # Set script execution timeout (in seconds) with a default value if not specified.
        self.execution_timeout: int = int(config.get("execution_timeout", 300))
        logging.info("CodeRunner initialized with execution timeout of %d seconds", self.execution_timeout)

    def run_script(self, script: str, env_config: Dict[str, Any]) -> Tuple[str, bool]:
        """
        Execute the provided Python script in an isolated Conda environment.

        The method follows these steps:
            1. Environment Preparation: Parse the env_config for 'env_name' and 'dependencies'.
            2. Use EnvironmentManager to create the environment and install dependencies.
            3. Write the script to a temporary file.
            4. Construct and execute the command using subprocess.
            5. Capture and return the output (stdout/stderr) and a success flag.

        Args:
            script (str): The Python script content to execute.
            env_config (Dict[str, Any]): Dictionary containing environment-related configuration.
                Expected keys:
                    - "env_name": A unique identifier for the Conda environment.
                    - "dependencies": A list of package dependencies required by the script.

        Returns:
            Tuple[str, bool]: A tuple containing:
                - output (str): Combined stdout and/or error messages from the script execution.
                - status (bool): True if execution succeeded (exit code 0), otherwise False.
        """
        temp_script_path: str = ""
        try:
            # Extract environment name and dependencies from env_config with defaults.
            env_name: str = str(env_config.get("env_name", "default"))
            dependencies: List[str] = env_config.get("dependencies", [])
            logging.info("Preparing to run script in environment '%s' with dependencies: %s", env_name, dependencies)

            # Step 1: Create the Conda environment.
            env_created: bool = self.env_manager.create_environment(env_name)
            if not env_created:
                error_msg: str = f"Failed to create Conda environment: {env_name}"
                logging.error(error_msg)
                return (error_msg, False)

            # Step 2: Install the required dependencies.
            deps_installed: bool = self.env_manager.install_dependencies(env_name, dependencies)
            if not deps_installed:
                error_msg: str = f"Failed to install dependencies in environment: {env_name}"
                logging.error(error_msg)
                return (error_msg, False)

            # Step 3: Write the script to a temporary file.
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py", encoding="utf-8") as temp_file:
                temp_file.write(script)
                temp_file.flush()
                temp_script_path = temp_file.name
            logging.info("Script written to temporary file: %s", temp_script_path)

            # Step 4: Construct the execution command.
            # Compute the full environment name using the env_prefix from EnvironmentManager.
            full_env_name: str = f"{self.env_manager.env_prefix}{env_name}"
            cmd: List[str] = ["conda", "run", "-n", full_env_name, "python", temp_script_path]
            logging.info("Executing command: %s", " ".join(cmd))

            # Step 5: Execute the command using subprocess.
            try:
                result: subprocess.CompletedProcess = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.execution_timeout
                )
            except subprocess.TimeoutExpired as te:
                error_msg = f"Script execution timed out after {self.execution_timeout} seconds: {str(te)}"
                logging.error(error_msg)
                return (error_msg, False)
            except Exception as exec_e:
                error_msg = f"Exception during script execution: {str(exec_e)}"
                logging.error(error_msg)
                return (error_msg, False)

            # Step 6: Process execution results.
            if result.returncode == 0:
                output: str = result.stdout.strip()
                logging.info("Script executed successfully. Output: %s", output)
                status: bool = True
            else:
                output = f"Error (exit code {result.returncode}): {result.stderr.strip()}\n{result.stdout.strip()}"
                logging.error("Script execution failed. %s", output)
                status = False

            return (output, status)

        except Exception as e:
            handle_error(e)
            # In case of unexpected error, return error message with failure status.
            return (f"Unexpected error: {str(e)}", False)
        finally:
            # Step 7: Cleanup the temporary script file if it was created.
            if temp_script_path and os.path.exists(temp_script_path):
                try:
                    os.remove(temp_script_path)
                    logging.info("Temporary script file removed: %s", temp_script_path)
                except Exception as cleanup_e:
                    logging.error("Failed to remove temporary script file: %s; Error: %s", temp_script_path, str(cleanup_e))
