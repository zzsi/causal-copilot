import os
import re
import sys
import json
import argparse
import asyncio
from openai import AsyncOpenAI

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

BASELINE_MODEL = "o1-mini"

def generate_dummy_args():
    # Create a dummy args object with necessary attributes for simulation.
    parser = argparse.ArgumentParser(description="Tiny Copilot - Simplified Version")
    parser.add_argument('--data-file', type=str, default="dataset/sachs/sachs.csv", help='Path to dataset file')
    parser.add_argument('--initial_query', type=str, default="selected algorithm: FGES; filter: continuous", help='Initial algorithm query')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode (skip heavy preprocessing)')
    # Modes: normal, skip, simulate
    parser.add_argument('--mode', type=str, choices=["normal", "skip", "simulate"], default="simulate", help='Mode of operation')
    parser.add_argument('--evaluate', action='store_true', default=False, help='Turn on metrics evaluation if ground truth is available')
    # Additional arguments required by the global initialization
    parser.add_argument('--organization', type=str, default="org-Xa9VGT8plP28JBRKtqBRjG5l")
    parser.add_argument('--project', type=str, default="proj_t78xpJomilJJu5qtuOK4vWfR")
    parser.add_argument('--apikey', type=str, default=None)
    parser.add_argument('--simulation_mode', type=str, default="offline")
    parser.add_argument('--data_mode', type=str, default="real")
    parser.add_argument('--demo_mode', type=bool, default=False)

    # Quick test of simulation output
    args = parser.parse_args()
    args.apikey = os.getenv("OPENAI_API_KEY")
    return args

async def simulate_task(task_id):
    """
    Generates one user simulation data by calling the simulate_user_query function.
    The fake_data (a pandas DataFrame) is converted to a JSON string.
    """
    try:
        from algorithm.tests.user_simulation import simulate_user_query
    except ImportError:
        raise ImportError("Could not import simulate_user_query from user_simulation.py.")
    args = generate_dummy_args()
    # Run the potentially blocking simulate_user_query in a separate thread.
    simulation = await asyncio.to_thread(simulate_user_query, args)
    # If fake_data is a DataFrame, convert it to a JSON serializable string.
    # Remove the actual fake data since it's not needed for the selection task.
    simulation.pop("fake_data")
    simulation["simulation_id"] = task_id
    return simulation

async def select_algorithm(simulation):
    """
    For a given simulation, use the o1-mini baseline to select the appropriate algorithm
    from the available candidate list in algorithm.wrappers.__all__, and explain the choice.
    """
    import json
    try:
        import algorithm.wrappers as wrappers
    except ImportError:
        raise ImportError("Could not import algorithm.wrappers module.")

    args = generate_dummy_args()
    # Get the list of candidate algorithm names
    candidate_algorithms = wrappers.__all__

    user_query = simulation.get("initial_query", "")
    statistics = simulation.get("statistics", {})

    # Build a prompt that includes the user query, statistics and candidate algorithms.
    prompt = (
        f"Given the following user query:\n{user_query}\n\n"
        f"And the following statistics:\n{json.dumps(statistics)}\n\n"
        f"Candidate algorithms:\n{json.dumps(candidate_algorithms)}\n\n"
        "Please choose the most suitable causal discovery algorithm for the data user provided and explain why this algorithm was chosen over the others.\n\n"
        "Please return the selected algorithm and the explanation in a JSON object. Wrap the JSON object in <json>...</json> tags. For"
        "example:\n"
        "<json>\n"
        "{{"
        "  'selected_algorithm': '...', "
        "  'selection_response': '...'"
        "}}"
        "</json>"
    )

    # Use asyncopenai's asynchronous chat completion with the o1-mini model.
    try:
        args = generate_dummy_args()
        client = AsyncOpenAI(organization=args.organization, project=args.project, api_key=args.apikey)
        response = await client.chat.completions.create(
            model=BASELINE_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        answer = response.choices[0].message.content.strip()
        answer = re.search(r'<json>(.*?)</json>', answer, re.DOTALL).group(1)
        answer = json.loads(answer)
    except Exception as e:
        answer = f"Error during selection: {str(e)}"
    
    selection = {
        "simulation_id": simulation.get("simulation_id"),
        "initial_query": user_query,
        "statistics": statistics,
        "selected_algorithm": answer.get("selected_algorithm"),
        "selection_response": answer.get("selection_response")
    }
    return selection

async def baseline():
    # Number of user simulation data points to generate (adjust as needed)
    NUM_SIMULATIONS = 10

    # Step 1: Generate N user simulation data in parallel.
    if os.path.exists("simulations.json"):
        with open("simulations.json", "r") as sim_file:
            valid_simulations = json.load(sim_file)
    else:
        simulation_tasks = [simulate_task(i) for i in range(NUM_SIMULATIONS)]
        simulation_results = await asyncio.gather(*simulation_tasks, return_exceptions=True)

        # Process simulation results, handling any exceptions.
        valid_simulations = []
        for idx, result in enumerate(simulation_results):
            if isinstance(result, Exception):
                print(f"Simulation task {idx} generated an exception: {result}")
            else:
                valid_simulations.append(result)

        # Save all simulation data into a JSON file.
        with open(f"simulations.json", "w") as sim_file:
            json.dump(valid_simulations, sim_file, indent=4)

    # Step 2: For each simulation, perform the algorithm selection in parallel.
    selection_tasks = [select_algorithm(sim) for sim in valid_simulations]
    selection_results = await asyncio.gather(*selection_tasks, return_exceptions=True)

    # Process selection results, handling any exceptions.
    final_selections = []
    for idx, result in enumerate(selection_results):
        if isinstance(result, Exception):
            print(f"Selection task for simulation {valid_simulations[idx].get('simulation_id')} generated an exception: {result}")
        else:
            final_selections.append(result)

    # Save the selection responses into a JSON file.
    with open(f"selection_results_{BASELINE_MODEL}.json", "w") as sel_file:
        json.dump(final_selections, sel_file, indent=4)

if __name__ == '__main__':
    asyncio.run(baseline())
