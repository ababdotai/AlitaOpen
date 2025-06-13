"""utils.py

This module provides shared utility functions and constants for the Alita project.
It includes functions for configuration parsing, logging setup, custom error handling,
reading template files, and ensuring required directories exist.

Modules depending on configuration settings, logging, or error handling should import this module.
"""

import os
import logging
import yaml
import traceback
from typing import Any, Dict, Optional

# Global variable to cache the configuration once loaded.
GLOBAL_CONFIG: Optional[Dict[str, Any]] = None


class AlitaError(Exception):
    """Custom exception for errors in the Alita system."""
    pass


def load_config(file_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Reads the configuration from the provided file path using PyYAML's safe_load.
    Raises an AlitaError if the file is not found or is invalid.

    Args:
        file_path (str): The path to the configuration file.
                         Defaults to "config.yaml".

    Returns:
        Dict[str, Any]: The loaded configuration as a dictionary.

    Raises:
        AlitaError: If the configuration file is not found or parsing fails.
    """
    if not os.path.exists(file_path):
        raise AlitaError(f"Configuration file not found: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        if config is None:
            raise AlitaError("Configuration file is empty or invalid.")
        return config
    except Exception as e:
        raise AlitaError(f"Failed to load configuration from {file_path}: {e}") from e


def setup_logging(config: Dict[str, Any]) -> None:
    """Set up logging based on configuration settings.

    This function configures the logging module using the settings from the
    'logging' section in the configuration. It ensures that the log directory
    exists and configures both a console and a file handler.

    Args:
        config (Dict[str, Any]): The global configuration dictionary.
    """
    logging_config = config.get("logging", {})
    level_str: str = logging_config.get("level", "INFO")
    log_file: str = logging_config.get("log_file", "logs/alita.log")

    # Create the log directory if it does not exist.
    log_dir: str = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Map the level string to a logging level.
    level = getattr(logging, level_str.upper(), logging.INFO)

    # Configure logging to output to both console and the specified file.
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8")
        ]
    )
    logging.info("Logging setup complete with level %s and output file %s", level_str, log_file)


def handle_error(e: Exception) -> None:
    """Handle errors by logging detailed traceback information and raising an AlitaError.

    This function logs the error message along with a full traceback for debugging,
    then raises an AlitaError to ensure consistent error handling across modules.

    Args:
        e (Exception): The exception to handle.

    Raises:
        AlitaError: Always raised after logging the error details.
    """
    logging.error("An error occurred: %s", str(e))
    logging.error("Traceback details:\n%s", traceback.format_exc())
    raise AlitaError(e) from e


def read_template(file_path: str) -> str:
    """Read and return content from a template file.

    This utility function opens the file at the provided path, reads its contents,
    and returns the content as a string. It raises an AlitaError if the file cannot be found or read.

    Args:
        file_path (str): The path to the template file.

    Returns:
        str: The file content.

    Raises:
        AlitaError: If the file does not exist or cannot be read.
    """
    if not os.path.exists(file_path):
        logging.error("Template file not found: %s", file_path)
        raise AlitaError(f"Template file not found: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as template_file:
            content = template_file.read()
        return content
    except Exception as e:
        logging.error("Failed to read template file: %s", file_path)
        handle_error(e)
        # The following line will never be reached because handle_error raises an exception.
        return ""


def ensure_directory_exists(dir_path: str) -> None:
    """Ensure that a directory exists; create it if it does not.

    Args:
        dir_path (str): The path of the directory to verify or create.
    """
    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
            logging.info("Created directory: %s", dir_path)
        except Exception as e:
            logging.error("Failed to create directory: %s", dir_path)
            handle_error(e)


def get_global_config(file_path: str = "config.yaml") -> Dict[str, Any]:
    """Get or load the global configuration.

    This function returns a cached configuration dictionary if already loaded.
    If not, it loads the configuration from the provided file_path and sets up logging.

    Args:
        file_path (str): Path to the configuration file. Defaults to "config.yaml".

    Returns:
        Dict[str, Any]: The loaded global configuration.
    """
    global GLOBAL_CONFIG
    if GLOBAL_CONFIG is None:
        GLOBAL_CONFIG = load_config(file_path)
        setup_logging(GLOBAL_CONFIG)
        logging.info("Global configuration loaded from %s", file_path)
    return GLOBAL_CONFIG
