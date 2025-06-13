"""benchmark.py

This module defines the Benchmark class to run evaluation experiments on the GAIA, Mathvista,
and PathVQA datasets. It loads the datasets from paths specified in the configuration, selects
a sample (if applicable), and runs each query through the full agent pipeline using the
ManagerAgent.orchestrate() method over a fixed number of rounds (evaluation.rounds).
It then aggregates evaluation metrics such as pass@1 and pass@3, and for GAIA, a further breakdown
by difficulty levels (e.g., Level 1, Level 2, Level 3) is provided.

Usage:
    benchmark = Benchmark(manager_agent, config)
    metrics = benchmark.run_benchmark()
    print(metrics)
"""

import os
import json
import random
import logging
from typing import Any, Dict, List

# Import the ManagerAgent from the corresponding module.
from manager_agent import ManagerAgent


class Benchmark:
    def __init__(self, manager_agent: ManagerAgent, config: Dict[str, Any]) -> None:
        """
        Initialize the Benchmark instance.

        Args:
            manager_agent (ManagerAgent): An instance of ManagerAgent that encapsulates
                the full CodeReAct pipeline (including WebAgent, MCPBrainstorm, ScriptGenerator,
                CodeRunner, EnvironmentManager, and MCPRegistry).
            config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
                Expected keys:
                    - benchmark.gaia.dataset_path (default: "data/gaia.json")
                    - benchmark.mathvista.dataset_path (default: "data/mathvista.json")
                    - benchmark.mathvista.sample_size (default: 100)
                    - benchmark.pathvqa.dataset_path (default: "data/pathvqa.json")
                    - benchmark.pathvqa.sample_size (default: 100)
                    - evaluation.rounds (default: 3)
        """
        self.manager_agent: ManagerAgent = manager_agent
        # Load benchmark dataset paths and sample sizes from configuration with defaults.
        benchmark_config: Dict[str, Any] = config.get("benchmark", {})
        self.gaia_dataset_path: str = benchmark_config.get("gaia", {}).get("dataset_path", "data/gaia.json")
        self.mathvista_dataset_path: str = benchmark_config.get("mathvista", {}).get("dataset_path", "data/mathvista.json")
        self.mathvista_sample_size: int = int(benchmark_config.get("mathvista", {}).get("sample_size", 100))
        self.pathvqa_dataset_path: str = benchmark_config.get("pathvqa", {}).get("dataset_path", "data/pathvqa.json")
        self.pathvqa_sample_size: int = int(benchmark_config.get("pathvqa", {}).get("sample_size", 100))
        # Load evaluation parameters.
        evaluation_config: Dict[str, Any] = config.get("evaluation", {})
        self.rounds: int = int(evaluation_config.get("rounds", 3))
        logging.info("Benchmark initialized with GAIA dataset: %s, Mathvista dataset: %s (sample_size=%d), "
                     "PathVQA dataset: %s (sample_size=%d), rounds=%d",
                     self.gaia_dataset_path, self.mathvista_dataset_path, self.mathvista_sample_size,
                     self.pathvqa_dataset_path, self.pathvqa_sample_size, self.rounds)

    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """
        Load a benchmark dataset from a JSON file.

        Args:
            dataset_path (str): Path to the JSON file containing the dataset.
        
        Returns:
            List[Dict[str, Any]]: A list of query dictionaries.
        """
        if not os.path.exists(dataset_path):
            logging.error("Dataset file not found: %s", dataset_path)
            return []
        try:
            with open(dataset_path, "r", encoding="utf-8") as file:
                dataset = json.load(file)
            if not isinstance(dataset, list):
                logging.error("Dataset file %s does not contain a list.", dataset_path)
                return []
            logging.info("Loaded %d queries from %s", len(dataset), dataset_path)
            return dataset
        except Exception as e:
            logging.error("Failed to load dataset from %s: %s", dataset_path, str(e))
            return []

    def evaluate_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single query by running it through the ManagerAgent pipeline
        for a fixed number of rounds and determine pass@1 and pass@3 metrics.

        The query dictionary is expected to have at least:
            - "question": The natural language query/task.
            - "expected_answer" or "correct_answer": The ground truth answer.
            - For GAIA queries, optionally a "difficulty" field.

        Args:
            query (Dict[str, Any]): A query dictionary.
        
        Returns:
            Dict[str, Any]: A dictionary containing:
                - "pass1": bool, True if the first round was successful.
                - "pass3": bool, True if any round was successful.
                - "difficulty": (Optional) The difficulty level if present.
        """
        question: str = str(query.get("question", "")).strip()
        expected_answer: str = ""
        if "expected_answer" in query:
            expected_answer = str(query["expected_answer"]).strip()
        elif "correct_answer" in query:
            expected_answer = str(query["correct_answer"]).strip()
        else:
            logging.warning("Query missing expected answer: %s", query)
            # If no expected answer, treat it as unsolvable.
            return {"pass1": False, "pass3": False, "difficulty": query.get("difficulty")}

        is_pass1: bool = False
        is_pass3: bool = False

        for round_idx in range(1, self.rounds + 1):
            try:
                logging.info("Evaluating query: '%s' (Round %d)", question, round_idx)
                # Invoke the ManagerAgent orchestration for the task.
                answer: str = self.manager_agent.orchestrate(question)
                normalized_answer: str = str(answer).strip()
                logging.info("Round %d answer: '%s'", round_idx, normalized_answer)
                # For pass@1, check the first round.
                if round_idx == 1 and normalized_answer == expected_answer:
                    is_pass1 = True
                # For pass@3, if any round gives the correct answer.
                if normalized_answer == expected_answer:
                    is_pass3 = True
            except Exception as e:
                logging.error("Exception during evaluation of query '%s' in round %d: %s", question, round_idx, str(e))
                continue

        return {"pass1": is_pass1, "pass3": is_pass3, "difficulty": query.get("difficulty")}

    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run the benchmark evaluation across GAIA, Mathvista, and PathVQA datasets.

        For each dataset:
            - Load the dataset from the specified JSON file.
            - For Mathvista and PathVQA, randomly sample the configured number of queries.
            - For each query, run the ManagerAgent pipeline for self.rounds rounds.
            - Aggregate overall pass@1 and pass@3 metrics.
            - For GAIA, if queries include a difficulty level, aggregate metrics per difficulty level.
        
        Returns:
            Dict[str, Any]: A dictionary containing all aggregated metrics.
                Example structure:
                {
                    "gaia": {
                        "total": int,
                        "pass@1": int,
                        "pass@3": int,
                        "pass@1_percent": float,
                        "pass@3_percent": float,
                        "levels": {
                            "Level 1": {"total": int, "pass@1": int, "pass@1_percent": float, "pass@3": int, "pass@3_percent": float},
                            "Level 2": {...},
                            "Level 3": {...}
                        }
                    },
                    "mathvista": {...},
                    "pathvqa": {...}
                }
        """
        # Initialize results dictionary.
        results: Dict[str, Any] = {}

        # Process each dataset.
        for dataset_key in ["gaia", "mathvista", "pathvqa"]:
            if dataset_key == "gaia":
                dataset_path: str = self.gaia_dataset_path
                sample_size: int = None  # Use full dataset.
            elif dataset_key == "mathvista":
                dataset_path = self.mathvista_dataset_path
                sample_size = self.mathvista_sample_size
            elif dataset_key == "pathvqa":
                dataset_path = self.pathvqa_dataset_path
                sample_size = self.pathvqa_sample_size
            else:
                continue

            dataset: List[Dict[str, Any]] = self.load_dataset(dataset_path)
            if sample_size is not None and isinstance(sample_size, int) and len(dataset) > sample_size:
                dataset = random.sample(dataset, sample_size)
                logging.info("Sampled %d queries for dataset %s", sample_size, dataset_key)

            total_queries: int = len(dataset)
            pass1_count: int = 0
            pass3_count: int = 0
            # For GAIA, maintain per-difficulty counters.
            levels_metrics: Dict[str, Dict[str, int]] = {}

            for query in dataset:
                eval_result: Dict[str, Any] = self.evaluate_query(query)
                if eval_result.get("pass1", False):
                    pass1_count += 1
                if eval_result.get("pass3", False):
                    pass3_count += 1

                # For GAIA, update difficulty level breakdown if available.
                if dataset_key == "gaia":
                    difficulty = eval_result.get("difficulty")
                    if difficulty:
                        # Normalize difficulty string.
                        diff_level: str = str(difficulty).strip()
                        if diff_level not in levels_metrics:
                            levels_metrics[diff_level] = {"total": 0, "pass1": 0, "pass3": 0}
                        levels_metrics[diff_level]["total"] += 1
                        if eval_result.get("pass1", False):
                            levels_metrics[diff_level]["pass1"] += 1
                        if eval_result.get("pass3", False):
                            levels_metrics[diff_level]["pass3"] += 1

            pass1_percent: float = (pass1_count / total_queries * 100) if total_queries > 0 else 0.0
            pass3_percent: float = (pass3_count / total_queries * 100) if total_queries > 0 else 0.0

            dataset_result: Dict[str, Any] = {
                "total": total_queries,
                "pass@1": pass1_count,
                "pass@3": pass3_count,
                "pass@1_percent": round(pass1_percent, 2),
                "pass@3_percent": round(pass3_percent, 2)
            }

            if dataset_key == "gaia":
                # For GAIA, compute per-difficulty percentages.
                levels_results: Dict[str, Any] = {}
                for level, metrics in levels_metrics.items():
                    level_total: int = metrics.get("total", 0)
                    level_pass1: int = metrics.get("pass1", 0)
                    level_pass3: int = metrics.get("pass3", 0)
                    level_pass1_percent: float = (level_pass1 / level_total * 100) if level_total > 0 else 0.0
                    level_pass3_percent: float = (level_pass3 / level_total * 100) if level_total > 0 else 0.0
                    levels_results[level] = {
                        "total": level_total,
                        "pass@1": level_pass1,
                        "pass@1_percent": round(level_pass1_percent, 2),
                        "pass@3": level_pass3,
                        "pass@3_percent": round(level_pass3_percent, 2)
                    }
                dataset_result["levels"] = levels_results

            results[dataset_key] = dataset_result
            logging.info("Completed evaluation for dataset '%s': %s", dataset_key, dataset_result)

        return results


# If this module is run as a script, perform a simple benchmark run.
if __name__ == "__main__":
    from utils import get_global_config
    try:
        # Load the global configuration.
        config: Dict[str, Any] = get_global_config("config.yaml")
        # Initialize ManagerAgent (ensuring it is imported from manager_agent.py).
        from manager_agent import ManagerAgent  # Re-import in __main__ to avoid circular issues.
        manager_agent_instance = ManagerAgent(config)
        # Create and run the Benchmark.
        benchmark = Benchmark(manager_agent_instance, config)
        final_metrics: Dict[str, Any] = benchmark.run_benchmark()
        logging.info("Final Benchmark Metrics: %s", final_metrics)
        print(json.dumps(final_metrics, indent=4))
    except Exception as e:
        logging.error("Benchmark run failed: %s", str(e))
