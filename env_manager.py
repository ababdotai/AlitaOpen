"""env_manager.py

This module defines the EnvironmentManager class, which provides functions to create
isolated Conda environments and install dependencies into them via subprocess calls.
All configuration values are read from the provided configuration dictionary (typically
parsed from config.yaml using utils.py). This module handles error logging and robustly
manages subprocess calls to ensure reproducibility and isolation.
"""

import subprocess
import logging
import os
from typing import List, Dict, Any


class EnvironmentManager:
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the EnvironmentManager with configuration settings.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing environment settings.
                Expected keys:
                    - environment.conda_base_env: Base Conda environment name.
                    - environment.env_prefix: Prefix for dynamically created environments.
                    - environment.dependency_timeout: Timeout for dependency installation in seconds.
        """
        try:
            env_config = config.get("environment", {})
            self.conda_base_env: str = env_config.get("conda_base_env", "base")
            self.env_prefix: str = env_config.get("env_prefix", "alita_env_")
            self.dependency_timeout: int = int(env_config.get("dependency_timeout", 300))
            logging.info("EnvironmentManager initialized with base_env=%s, env_prefix=%s, "
                         "dependency_timeout=%d",
                         self.conda_base_env, self.env_prefix, self.dependency_timeout)
        except Exception as e:
            logging.error("Failed to initialize EnvironmentManager configuration: %s", str(e))
            raise

    def _get_full_env_name(self, env_name: str) -> str:
        """
        Combine the configured environment prefix with the provided environment name.

        Args:
            env_name (str): The base name for the environment.

        Returns:
            str: The full environment name with prefix.

        Raises:
            ValueError: If env_name is empty.
        """
        if not env_name:
            raise ValueError("Environment name must not be empty.")
        return f"{self.env_prefix}{env_name}"

    def create_environment(self, env_name: str, dependencies: List[str]) -> bool:
        """
        Create a new Conda environment with the provided name and log the process.

        The environment name is combined with the configured prefix to form a unique name.
        Although a list of dependencies is provided, this function handles only the creation
        of the environment. Dependency installation is handled by install_dependencies.

        Args:
            env_name (str): The base name for the environment.
            dependencies (List[str]): List of dependencies (for logging context).

        Returns:
            bool: True if the environment was successfully created; False otherwise.
        """
        if not env_name:
            logging.error("create_environment failed: env_name is empty.")
            return False
        try:
            full_env_name: str = self._get_full_env_name(env_name)
            logging.info("Creating Conda environment: %s", full_env_name)

            # Construct the command to create the Conda environment.
            # e.g., conda create -n <env_full_name> --yes
            cmd: List[str] = ["conda", "create", "-n", full_env_name, "--yes"]
            result: subprocess.CompletedProcess = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logging.error("Failed to create environment %s. Return code: %s, stderr: %s",
                              full_env_name, result.returncode, result.stderr)
                return False

            logging.info("Environment %s created successfully. stdout: %s",
                         full_env_name, result.stdout)
            return True

        except subprocess.CalledProcessError as cpe:
            logging.error("Subprocess error during environment creation: %s", str(cpe))
            return False
        except Exception as e:
            logging.error("Exception during environment creation: %s", str(e))
            return False

    def install_dependencies(self, env_name: str, dependencies: List[str]) -> bool:
        """
        Install a list of dependencies into the specified Conda environment.

        Uses 'conda run' to execute pip install within the environment. The installation
        command is executed with a timeout as specified in the configuration. Errors during
        installation are logged, and the function returns False if any step fails.

        Args:
            env_name (str): The base name for the environment (will be combined with the prefix).
            dependencies (List[str]): List of packages to install.

        Returns:
            bool: True if all dependencies were installed successfully; False otherwise.
        """
        if not dependencies:
            logging.info("No dependencies to install for environment: %s", env_name)
            return True

        try:
            full_env_name: str = self._get_full_env_name(env_name)
            logging.info("Installing dependencies in environment %s: %s",
                         full_env_name, dependencies)

            # Construct the command to install dependencies using pip inside the environment.
            # e.g., conda run -n <full_env_name> pip install dep1 dep2 ...
            cmd: List[str] = ["conda", "run", "-n", full_env_name, "pip", "install"] + dependencies

            result: subprocess.CompletedProcess = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.dependency_timeout
            )
            if result.returncode != 0:
                logging.error("Failed to install dependencies in environment %s. Return code: %s, stderr: %s",
                              full_env_name, result.returncode, result.stderr)
                return False

            logging.info("Dependencies installed successfully in environment %s. stdout: %s",
                         full_env_name, result.stdout)
            return True

        except subprocess.TimeoutExpired:
            logging.error("Installation of dependencies in environment %s timed out after %d seconds.",
                          full_env_name, self.dependency_timeout)
            return False
        except subprocess.CalledProcessError as cpe:
            logging.error("Subprocess error during dependency installation: %s", str(cpe))
            return False
        except Exception as e:
            logging.error("Exception during dependency installation: %s", str(e))
            return False
