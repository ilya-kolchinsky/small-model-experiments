import asyncio
import logging
import os
import time
from pathlib import Path
from typing import List, Tuple

import openai
from langgraph.errors import GraphRecursionError
from pydantic import ValidationError

from evaluator.components.mcp_proxy import MCPProxyManager
from evaluator.interfaces.metric_collector import MetricCollector
from evaluator.components.data_provider import get_queries, get_tools_from_queries, QuerySpecification
from evaluator.utils.module_extractor import create_algorithms, create_metric_collectors
from evaluator.eval_spec import EVALUATED_ALGORITHMS, METRIC_COLLECTORS, DATASET_SETTINGS, VERBOSE, EvaluationEnvSpec, \
    EXPERIMENTAL_ENVIRONMENT_SETTINGS
from evaluator.interfaces.tool_rag_algorithm import ToolRagAlgorithm
from evaluator.utils.csv_logger import CSVLogger
from evaluator.components.llm_provider import get_llm
from dotenv import load_dotenv

from evaluator.utils.tool_logger import ToolLogger
from evaluator.utils.utils import print_iterable_verbose

load_dotenv()

if not VERBOSE:
    # in non-verbose mode we want to suppress the excessive output from MCP server and client
    logging.disable(logging.WARNING)

MAX_RETRIES = 5
RETRY_DELAY = 30

# an experiment is defined as a combination of a tool RAG algorithm and an environment specification
ExperimentSpec = Tuple[ToolRagAlgorithm, EvaluationEnvSpec]


async def run_all_experiments() -> None:

    # Set up the necessary components for the experiments:
    # - the language model
    # - the data to evaluate on
    # - the tools to use
    # - the evaluation metrics to collect and calculate
    # - the algorithms to be evaluated

    print(f"Launching evaluation setup with the following parameters:")
    for key, value in DATASET_SETTINGS.items():
        print(f"{key}: {value}")
    print(f"verbose: {VERBOSE}\n")

    print("Loading metric collectors...")
    metric_collectors = create_metric_collectors(METRIC_COLLECTORS)
    print(f"The following metric collectors will be active during evaluation:\n{METRIC_COLLECTORS}\n")

    print("Loading algorithms and environment configurations...")
    algorithms = create_algorithms(EVALUATED_ALGORITHMS)
    print(f"The following algorithms will be evaluated:\n{EVALUATED_ALGORITHMS}\n")
    print(f"The following environment configurations will be evaluated:\n{EXPERIMENTAL_ENVIRONMENT_SETTINGS}\n")

    experiment_specs = _produce_experiment_specs(algorithms, EXPERIMENTAL_ENVIRONMENT_SETTINGS)

    mcp_proxy_manager = MCPProxyManager(int(os.getenv("MCP_PROXY_LOCAL_PORT")))

    # Actually run the experiments
    with CSVLogger(metric_collectors,
                   Path(os.getenv("OUTPUT_PATH")),
                   metadata_columns=["Experiment ID", "Algorithm", "Environment"]) as logger:
        for i, spec in enumerate(experiment_specs):
            algorithm, environment = spec
            print(f"{'-' * 60}\nRunning Experiment {i+1} of {len(experiment_specs)}: {_spec_to_str(spec)}...\n{'-' * 60}")
            await _run_experiment(i+1, spec, metric_collectors, mcp_proxy_manager)
            print(f"{'-' * 60}\nSummary of Experiment {i+1} - {_spec_to_str(spec)}\n{'-' * 60}")
            logger.log_experiment(meta_values={"Experiment ID": i+1, "Algorithm": algorithm.get_unique_id(), "Environment": environment.model_dump()})

    mcp_proxy_manager.stop_server()
    print(f"Successfully executed {len(experiment_specs)} experiment(s).")


def _spec_to_str(spec: ExperimentSpec) -> str:
    algorithm, environment = spec
    return f"{algorithm.get_unique_id()}:{environment.model_dump()}"


def _produce_experiment_specs(algorithms: List[ToolRagAlgorithm], env_specs: List[EvaluationEnvSpec]) -> List[ExperimentSpec]:
    result = []
    for algo in algorithms:
        for spec in env_specs:
            result.append((algo, spec))
    return result


async def _run_experiment(exp_index: int,
                          spec: ExperimentSpec,
                          metric_collectors: List[MetricCollector],
                          mcp_proxy_manager: MCPProxyManager,
                          ) -> None:
    queries = await _set_up_experiment(spec, metric_collectors, mcp_proxy_manager)
    algorithm, environment = spec

    for i, query_spec in enumerate(queries):
        print(f"Processing query #{query_spec.id} (Experiment {exp_index}, query {i+1} of {len(queries)})...")

        for mc in metric_collectors:
            mc.prepare_for_measurement(query_spec)

        response = None
        retrieved_tools = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response, retrieved_tools = await algorithm.process_query(query_spec)
                break  # success, go to next query
            except (GraphRecursionError, ValidationError, openai.BadRequestError) as e:
                # if we hit it, the model obviously failed to adequately address the query
                # TODO: execution errors must be tracked as a separate metric and categorized according to the error type
                print(f"Exception while processing query {i+1}: {e}")
                if attempt < MAX_RETRIES:
                    print(f"Retrying query {i+1}...")
                    continue
                else:
                    print(f"All {MAX_RETRIES} retries failed. Marking query {i+1} as failed.")
                    response = {"response": "Query execution failed."}
                    break
            except openai.InternalServerError as e:
                # detect gateway timeout (504) specifically
                if "504" in str(e) or "Gateway Time-out" in str(e):
                    print(f"Timeout on query {i+1} (attempt {attempt}/{MAX_RETRIES})")
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        print(f"All {MAX_RETRIES} retried failed, marking query {i+1} as failed.")
                        response = {"response": "Query execution failed."}
                        break
                # it's not a 504 / timeout - raise the exception and abort the experiment
                raise

        executed_tools = ToolLogger(os.getenv("TOOL_LOG_PATH")).get_executed_tools()

        for mc in metric_collectors:
            mc.register_measurement(
                query_spec,
                response=response,
                executed_tools=executed_tools,
                retrieved_tools=retrieved_tools
            )

    algorithm.tear_down()
    for mc in metric_collectors:
        mc.tear_down()


async def _set_up_experiment(spec: ExperimentSpec,
                             metric_collectors: List[MetricCollector],
                             mcp_proxy_manager: MCPProxyManager,
                             ) -> List[QuerySpecification]:
    algorithm, environment = spec

    print(f"Initializing LLM connection: {environment.model_id}")
    llm = get_llm(model_id=environment.model_id)
    print("Connection established successfully.\n")

    print("Fetching queries for the current experiment...")
    queries = get_queries(environment)
    print(f"Successfully loaded {len(queries)} queries.\n")
    print_iterable_verbose("The following queries will be executed:\n", queries)

    print("Retrieving tool definitions for the current experiment...")
    tool_specs = get_tools_from_queries(queries)
    tools = await mcp_proxy_manager.run_mcp_proxy(tool_specs, init_client=True).get_tools()
    print_iterable_verbose("The following tools will be available during evaluation:\n", tools)
    print(f"The experiment will proceed with {len(tools)} tool(s).\n")

    print("Setting up the algorithm and the metric collectors...")
    algorithm.set_up(llm, tools)
    for mc in metric_collectors:
        mc.set_up()
    print("All set!\n")

    return queries


if __name__ == "__main__":
    asyncio.run(run_all_experiments())
