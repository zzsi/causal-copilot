import json
import os
import re

def add_benchmarking_results_to_all(benchmarking_json, directory="."):
    """Add benchmarking results to all algorithm description text files in directory.
    
    Args:
        benchmarking_json (dict): Dictionary containing benchmarking results
        directory (str): Directory containing algorithm description text files
    """
    
    # Find all .txt files in directory
    for filename in os.listdir(directory):
        if re.match(r'.*\.txt$', filename):
            # Extract algorithm name from filename
            algo_name = filename.replace('.txt', '')
            print(algo_name, list(benchmarking_json.values())[0].keys())
            if not algo_name in list(benchmarking_json.values())[0].keys():
                continue
            # Build full file path
            file_path = os.path.join(directory, filename)
            
            # Add benchmarking results to this file
            add_benchmarking_results(file_path, benchmarking_json)
    
    # Create a consolidated benchmarking results file
    # create_consolidated_benchmarking_results(benchmarking_json, directory)

def add_benchmarking_results(algorithm_desc_path, benchmarking_json):
    """Add benchmarking results to algorithm description text file.
    
    Args:
        algorithm_desc_path (str): Path to algorithm description text file
        benchmarking_json (dict): Dictionary containing benchmarking results
    """
    with open(algorithm_desc_path, 'r') as f:
        content = f.read()
        
    # Remove any existing benchmarking sections to prevent duplication
    if "Benchmarking Results" in content:
        # Find all occurrences of benchmarking sections
        pattern = r"────+\nBenchmarking Results\n────+[\s\S]*?(?=────+|$)"
        # Replace with empty string
        content = re.sub(pattern, "", content)
        print("Found and removed existing benchmarking sections")
        
    # Clean up any potential double newlines or trailing whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = content.rstrip()
        
    with open(algorithm_desc_path, 'w') as f:
        f.write(content + "\n\n")
        f.write("────────────────────────────────────────────────────────\n")
        f.write("Benchmarking Results\n") 
        f.write("────────────────────────────────────────────────────────\n\n")

        # Simplified benchmarking context
        f.write("• Benchmarking Context\n")
        f.write("  – Simulations with various node counts (5-100), sample sizes (500-5000), edge probabilities (0.1-0.3)\n")
        f.write("  – Tests included linear/non-linear functions, different noise types, discrete variables\n")
        f.write("  – Multiple scenarios: measurement error, missing values, multi-domain data\n\n")
        
        f.write("• Performance Metrics\n")
        f.write("  – Performance level (1-10): Based on F1 score, higher is better\n")
        f.write("  – Efficiency level (0-5): Based on runtime, higher is better\n")
        f.write("  – Overall Score (1-10): Combined measure of performance and efficiency\n\n")
        
        # Rankings table with simplified format
        f.write("• Algorithm Performance by Scenario\n\n")
        
        # First determine which metrics are available for this algorithm
        algo_name = algorithm_desc_path.split('/')[-1].replace('.txt', '')
        has_efficiency = False
        has_composite = False
        
        for scenario in benchmarking_json:
            if algo_name in benchmarking_json[scenario]:
                result = benchmarking_json[scenario][algo_name]
                if 'efficiency' in result['levels'] and result['levels']['efficiency'] != 'N/A':
                    has_efficiency = True
                if 'composite' in result['levels'] and result['levels']['composite'] != 'N/A':
                    has_composite = True
        
        # Create the appropriate table header based on available metrics
        header = "| Scenario | Performance |"
        if has_efficiency:
            header += " Efficiency |"
        if has_composite:
            header += " Overall Score |"
        f.write(header + "\n")
        
        # Create the separator line with the right number of columns
        separator = "|----------|------------|"
        if has_efficiency:
            separator += "------------|"
        if has_composite:
            separator += "------------|"
        f.write(separator + "\n")
        
        # Add data rows - group scenarios to keep related ones together
        scenario_groups = {
            "general": [],
            "scaling": ["Variable Scaling", "Sample Scaling"],
            "function": ["Linear Function", "Non-Linear Function"],
            "noise": ["Gaussian Noise", "Non-Gaussian Noise"],
            "other": ["Heterogeneity", "Measurement Error", "Missing Data", "Edge Probability",
                    "Discrete Ratio", "Dense Graph", "Sparse Graph", 
                    "High Missing Data", "High Measurement Error", "Highly Heterogeneous"]
        }
        
        # First, determine which scenarios are actually available
        available_scenarios = {}
        for group, scenarios in scenario_groups.items():
            if group == "general":
                continue
            available = [s for s in scenarios if s in benchmarking_json and algo_name in benchmarking_json[s]]
            if available:
                available_scenarios[group] = available
            
            # Add scenarios that aren't in any specific group to general
            for scenario in benchmarking_json:
                is_in_group = False
                for group_scenarios in scenario_groups.values():
                    if scenario in group_scenarios:
                        is_in_group = True
                        break
                if not is_in_group and algo_name in benchmarking_json[scenario]:
                    scenario_groups["general"].append(scenario)
            
        # Now print tables in groups
        groups_to_print = ["scaling", "function", "noise", "other", "general"]
        
        for group in groups_to_print:
            scenarios = scenario_groups[group]
            if not scenarios:
                continue
                
            # Only add section headers for non-empty groups
            if group == "scaling":
                f.write("\n• Scaling Scenarios\n")
            elif group == "function":
                f.write("\n• Function Type Scenarios\n")
            elif group == "noise":
                f.write("\n• Noise Type Scenarios\n")
            elif group == "other" and scenario_groups["other"]:
                f.write("\n• Other Scenarios\n")
            elif group == "general" and scenario_groups["general"]:
                f.write("\n• General Scenarios\n")
                
            for scenario in scenarios:
                if scenario in benchmarking_json and algo_name in benchmarking_json[scenario]:
                    result = benchmarking_json[scenario][algo_name]
                    
                    # Get performance level (treat 'N/A' appropriately)
                    performance = result['levels'].get('performance', 'N/A')
                    performance_str = f"{float(performance):.1f}" if performance != 'N/A' else 'N/A'
                    
                    # Create the row with performance
                    row = f"| {scenario} | {performance_str} |"
                    
                    # Add efficiency if available
                    if has_efficiency:
                        efficiency = result['levels'].get('efficiency', 'N/A')
                        eff_str = f"{float(efficiency):.1f}" if efficiency != 'N/A' else 'N/A'
                        row += f" {eff_str} |"
                    
                    # Add overall score (formerly composite) if available
                    if has_composite:
                        composite = result['levels'].get('composite', 'N/A')
                        comp_str = f"{float(composite):.1f}" if composite != 'N/A' else 'N/A'
                        row += f" {comp_str} |"
                    
                    f.write(row + "\n")
        
        # Add a special linear vs non-linear comparison if both exist
        if ("Linear Function" in benchmarking_json and algo_name in benchmarking_json["Linear Function"] and
            "Non-Linear Function" in benchmarking_json and algo_name in benchmarking_json["Non-Linear Function"]):
            
            linear_perf = benchmarking_json["Linear Function"][algo_name]['levels'].get('performance', 'N/A')
            nonlinear_perf = benchmarking_json["Non-Linear Function"][algo_name]['levels'].get('performance', 'N/A')
            
            if linear_perf != 'N/A' and nonlinear_perf != 'N/A':
                linear_perf = float(linear_perf)
                nonlinear_perf = float(nonlinear_perf)
                f.write("\n• Linear vs Non-Linear Performance\n")
                f.write(f"  – Linear function performance: {linear_perf:.1f}\n")
                f.write(f"  – Non-linear function performance: {nonlinear_perf:.1f}\n")
                
                if linear_perf > nonlinear_perf:
                    diff = linear_perf - nonlinear_perf
                    f.write(f"  – This algorithm performs {diff:.1f} points better on linear functions\n")
                elif nonlinear_perf > linear_perf:
                    diff = nonlinear_perf - linear_perf
                    f.write(f"  – This algorithm performs {diff:.1f} points better on non-linear functions\n")
                else:
                    f.write("  – This algorithm performs equally well on linear and non-linear functions\n")
        
        # Simplified Analysis section
        f.write("\n• Analysis\n")
        
        # Calculate overall stats
        total_scenarios = len(benchmarking_json)
        avg_ranks = []
        for scenario in benchmarking_json:
            if algo_name in benchmarking_json[scenario]:
                avg_ranks.append(benchmarking_json[scenario][algo_name]['ranking']['mean'])
        
        if avg_ranks:
            overall_rank = sum(avg_ranks) / len(avg_ranks)
            f.write(f"\n  – Overall ranking across scenarios: {overall_rank:.2f}\n")
        
        # Add scenario-specific insights (keeping the most relevant ones)
        # Check if algorithm is in top performers
        top_performers = ["GRaSP", "XGES", "FCI"]
        if algo_name in top_performers:
            f.write("  – Consistently ranks among top performers across multiple scenarios\n")
            
        # Add simplified specific scenario strengths
        for scenario_type, threshold in [
            ("Variable Scaling", 5),
            ("Sample Scaling", 5),
            ("Non-Gaussian Noise", 5),
            ("Heterogeneity", 5)
        ]:
            if (scenario_type in benchmarking_json and 
                algo_name in benchmarking_json[scenario_type] and 
                benchmarking_json[scenario_type][algo_name]["ranking"]["mean"] < threshold):
                f.write(f"  – Strong performance in {scenario_type.lower()} scenarios\n")

def create_consolidated_benchmarking_results(benchmarking_json, directory):
    """Create a consolidated file with benchmarking results for all algorithms.
    
    Args:
        benchmarking_json (dict): Dictionary containing benchmarking results
        directory (str): Directory to save the consolidated file
    """
    output_path = os.path.join(directory, "consolidated_benchmarking_results.txt")
    
    with open(output_path, 'w') as f:
        f.write("# Consolidated Benchmarking Results for All Algorithms\n\n")
        f.write("────────────────────────────────────────────────────────\n")
        f.write("Overview\n")
        f.write("────────────────────────────────────────────────────────\n\n")
        
        # Simplified benchmarking context
        f.write("• Benchmarking Context\n")
        f.write("  – Simulations with various node counts (5-100), sample sizes (500-5000), edge probabilities (0.1-0.3)\n")
        f.write("  – Tests included linear/non-linear functions, different noise types, discrete variables\n")
        f.write("  – Multiple scenarios: measurement error, missing values, multi-domain data\n\n")
        
        f.write("• Performance Metrics\n")
        f.write("  – Performance level (1-10): Based on F1 score, higher is better\n")
        f.write("  – Efficiency level (0-5): Based on runtime, higher is better\n")
        f.write("  – Overall Score (1-10): Combined measure of performance and efficiency\n\n")
        
        # Get all algorithms across all scenarios
        all_algorithms = set()
        for scenario in benchmarking_json:
            all_algorithms.update(benchmarking_json[scenario].keys())
        
        # Calculate overall stats for each algorithm
        algorithm_overall_stats = {}
        for algo_name in all_algorithms:
            scenarios_present = 0
            avg_rank = 0
            avg_performance = 0
            has_efficiency = False
            avg_efficiency = 0
            has_composite = False
            avg_composite = 0
            
            # Track linear and non-linear performance separately
            linear_perf = None
            nonlinear_perf = None
            
            for scenario in benchmarking_json:
                if algo_name in benchmarking_json[scenario]:
                    result = benchmarking_json[scenario][algo_name]
                    scenarios_present += 1
                    avg_rank += result['ranking']['mean']
                    
                    # Handle performance
                    performance = result['levels']['performance']
                    if performance != 'N/A':
                        avg_performance += float(performance)
                    
                    # Check and handle efficiency
                    if 'efficiency' in result['levels'] and result['levels']['efficiency'] != 'N/A':
                        has_efficiency = True
                        avg_efficiency += float(result['levels']['efficiency'])
                    
                    # Check and handle composite
                    if 'composite' in result['levels'] and result['levels']['composite'] != 'N/A':
                        has_composite = True
                        avg_composite += float(result['levels']['composite'])
                    
                    # Store linear/non-linear performance specifically
                    if scenario == "Linear Function" and performance != 'N/A':
                        linear_perf = float(performance)
                    elif scenario == "Non-Linear Function" and performance != 'N/A':
                        nonlinear_perf = float(performance)
            
            if scenarios_present > 0:
                stats = {
                    'avg_rank': avg_rank / scenarios_present,
                    'avg_performance': avg_performance / scenarios_present,
                    'scenarios_present': scenarios_present,
                }
                
                if has_efficiency:
                    stats['avg_efficiency'] = avg_efficiency / scenarios_present
                
                if has_composite:
                    stats['avg_composite'] = avg_composite / scenarios_present
                
                if linear_perf is not None:
                    stats['linear_perf'] = linear_perf
                
                if nonlinear_perf is not None:
                    stats['nonlinear_perf'] = nonlinear_perf
                
                algorithm_overall_stats[algo_name] = stats
        
        # Sort algorithms by overall rank for the summary table
        sorted_algos_by_rank = sorted(
            algorithm_overall_stats.items(),
            key=lambda x: x[1]['avg_rank']
        )
        
        # For each algorithm, display overall stats
        f.write("────────────────────────────────────────────────────────\n")
        f.write("Overall Algorithm Performance\n")
        f.write("────────────────────────────────────────────────────────\n\n")
        
        # Create header based on available metrics
        header = "| Algorithm | Performance |"
        if any('avg_efficiency' in stats for _, stats in sorted_algos_by_rank):
            header += " Efficiency |"
        if any('avg_composite' in stats for _, stats in sorted_algos_by_rank):
            header += " Overall Score |"
        f.write(header + "\n")
        
        # Create separator
        separator = "|-----------|-------------|"
        if any('avg_efficiency' in stats for _, stats in sorted_algos_by_rank):
            separator += "-------------|"
        if any('avg_composite' in stats for _, stats in sorted_algos_by_rank):
            separator += "-------------|"
        f.write(separator + "\n")
        
        # Add data rows
        for algo_name, stats in sorted_algos_by_rank:
            row = f"| {algo_name} | {stats['avg_performance']:.2f} |"
            
            if any('avg_efficiency' in s for _, s in sorted_algos_by_rank):
                eff_val = stats.get('avg_efficiency', 'N/A')
                eff_str = f"{eff_val:.2f}" if eff_val != 'N/A' else 'N/A'
                row += f" {eff_str} |"
            
            if any('avg_composite' in s for _, s in sorted_algos_by_rank):
                comp_val = stats.get('avg_composite', 'N/A')
                comp_str = f"{comp_val:.2f}" if comp_val != 'N/A' else 'N/A'
                row += f" {comp_str} |"
            
            f.write(row + "\n")
        
        # Add a section that compares linear vs non-linear performance
        f.write("\n\n────────────────────────────────────────────────────────\n")
        f.write("Linear vs Non-Linear Function Performance\n")
        f.write("────────────────────────────────────────────────────────\n\n")
        
        f.write("| Algorithm | Linear | Non-Linear | Difference |\n")
        f.write("|-----------|--------|------------|------------|\n")
        
        # Only include algorithms with both linear and non-linear data
        for algo_name, stats in sorted_algos_by_rank:
            if 'linear_perf' in stats and 'nonlinear_perf' in stats:
                linear = stats['linear_perf']
                nonlinear = stats['nonlinear_perf']
                diff = linear - nonlinear
                diff_str = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"
                f.write(f"| {algo_name} | {linear:.1f} | {nonlinear:.1f} | {diff_str} |\n")
        
        # Explain the difference values
        f.write("\nNote: Positive difference values indicate better performance on linear functions,\n")
        f.write("negative values indicate better performance on non-linear functions.\n")
        
        # Simplified scenario-specific results
        f.write("\n\n────────────────────────────────────────────────────────\n")
        f.write("Top and Bottom Performers by Scenario\n")
        f.write("────────────────────────────────────────────────────────\n\n")
        
        for scenario in benchmarking_json:
            f.write(f"\n• {scenario}\n")
            
            # Sort algorithms by their ranking in this scenario
            sorted_algos = sorted(
                [(algo, benchmarking_json[scenario][algo]['ranking']['mean']) 
                 for algo in benchmarking_json[scenario]],
                key=lambda x: x[1]
            )
            
            # Show top 5 algorithms for each scenario
            f.write("  Top performers:\n")
            top_algos = sorted_algos[:5]
            for i, (algo_name, rank) in enumerate(top_algos):
                result = benchmarking_json[scenario][algo_name]
                perf = result['levels']['performance']
                perf_str = f"{float(perf):.1f}" if perf != 'N/A' else 'N/A'
                
                # For Variable Scaling and Sample Scaling, also show efficiency
                if scenario in ["Variable Scaling", "Sample Scaling"]:
                    eff = result['levels'].get('efficiency', 'N/A')
                    eff_str = f"{float(eff):.1f}" if eff != 'N/A' else 'N/A'
                    f.write(f"  {i+1}. {algo_name}: Performance {perf_str}, Efficiency {eff_str}\n")
                else:
                    f.write(f"  {i+1}. {algo_name}: Performance {perf_str}\n")
            
            # Show bottom 5 algorithms for each scenario
            f.write("\n  Bottom performers:\n")
            bottom_algos = sorted_algos[-5:] if len(sorted_algos) >= 5 else sorted_algos
            bottom_algos.reverse()  # Display from worst to better
            for i, (algo_name, rank) in enumerate(bottom_algos):
                result = benchmarking_json[scenario][algo_name]
                perf = result['levels']['performance']
                perf_str = f"{float(perf):.1f}" if perf != 'N/A' else 'N/A'
                
                # For Variable Scaling and Sample Scaling, also show efficiency
                if scenario in ["Variable Scaling", "Sample Scaling"]:
                    eff = result['levels'].get('efficiency', 'N/A')
                    eff_str = f"{float(eff):.1f}" if eff != 'N/A' else 'N/A'
                    f.write(f"  {i+1}. {algo_name}: Performance {perf_str}, Efficiency {eff_str}\n")
                else:
                    f.write(f"  {i+1}. {algo_name}: Performance {perf_str}\n")
    
    print(f"Consolidated benchmarking results saved to {output_path}")

def create_filtered_benchmarking_results(benchmarking_json, algorithm_list=None, output_filename=None):
    """Create a consolidated benchmarking results text with only the specified algorithms.
    
    Args:
        benchmarking_json (dict): Dictionary containing benchmarking results
        algorithm_list (list): List of algorithm names to include in the results
        output_filename (str, optional): Name of the output file if saving is desired
    
    Returns:
        str: Formatted text containing the filtered benchmarking results
    """
    if algorithm_list is None or len(algorithm_list) == 0:
        print("No algorithms specified for filtering. Using all algorithms.")
        return None
    
    # Filter the benchmarking_json to only include the specified algorithms
    filtered_json = {}
    for scenario, algorithms in benchmarking_json.items():
        filtered_algorithms = {}
        for algo, data in algorithms.items():
            # Check if any algorithm in algorithm_list is a substring of algo
            # This handles cases like "PC_alpha=0.01" matching "PC"
            if any(target_algo in algo for target_algo in algorithm_list):
                filtered_algorithms[algo] = data
        
        if filtered_algorithms:  # Only include scenarios that have at least one of the specified algorithms
            filtered_json[scenario] = filtered_algorithms
    
    if not filtered_json:
        return "No results found for the specified algorithms."
    
    # Create the output text
    output_text = ""
    output_text += "# ALGORITHM BENCHMARKING RESULTS\n\n"
    output_text += "• Benchmarking Context\n"
    output_text += "  – Simulations with various node counts (5-1000), sample sizes (500-10000), edge probabilities (0.11-0.78, average degree 1 - 7)\n"
    output_text += "  – Tests included linear/non-linear functions, gaussian/uniform noise types, discrete variables (0-20%)\n"
    output_text += "  – Multiple scenarios: measurement error (10-50%), missing values (10-30%), multi-domain data (1-10 domains)\n\n"
    
    output_text += "• Performance Metrics\n"
    output_text += "  – Performance level (1-10): Based on F1 score, higher is better\n"
    output_text += "  – Efficiency level (0-5): Based on runtime, higher is better\n"
    output_text += "  – Overall Score (1-10): Combined measure of performance and efficiency\n\n"
    
    output_text += "• Important Note on Efficiency Scoring\n"
    output_text += "  – Benchmarks include large-scale systems with up to 1000 nodes and may timeout for some algorithms\n"
    output_text += "  – For large-scale systems (node size > 200), prioritize algorithms that can utilize available GPUs\n"
    output_text += "  – GPU-accelerated methods provide significant efficiency advantages in large-scale scenarios\n\n"
    output_text += "────────────────────────────────────────────────────────\n"
    output_text += "Filtered Benchmarking Results\n"
    output_text += "────────────────────────────────────────────────────────\n\n"
    
    output_text += f"Algorithms included: {', '.join(algorithm_list)}\n\n"
    
    # Overall algorithm performance across all scenarios
    output_text += "────────────────────────────────────────────────────────\n"
    output_text += "Overall Algorithm Performance\n"
    output_text += "────────────────────────────────────────────────────────\n\n"
    
    # Get all unique algorithm names across all scenarios
    all_algos = set()
    for scenario in filtered_json:
        all_algos.update(filtered_json[scenario].keys())
    
    # Calculate average performance for each algorithm
    algo_avg_performance = {}
    for algo in all_algos:
        performances = []
        for scenario in filtered_json:
            if algo in filtered_json[scenario]:
                perf = filtered_json[scenario][algo]['levels'].get('performance', 'N/A')
                if perf != 'N/A':
                    performances.append(float(perf))
        
        if performances:
            algo_avg_performance[algo] = sum(performances) / len(performances)
    
    # Sort algorithms by average performance
    sorted_algos = sorted(algo_avg_performance.items(), key=lambda x: x[1], reverse=True)
    
    # Display overall ranking
    output_text += "Overall ranking based on average performance across all scenarios:\n\n"
    for i, (algo, avg_perf) in enumerate(sorted_algos):
        output_text += f"{i+1}. {algo}: {avg_perf:.1f}\n"
    
    # Algorithm-specific detailed results
    output_text += "\n\n────────────────────────────────────────────────────────\n"
    output_text += "Detailed Results by Algorithm\n"
    output_text += "────────────────────────────────────────────────────────\n\n"
    
    for algo_name in all_algos:
        output_text += f"• {algo_name}\n\n"
        
        # Check which metrics are available for this algorithm
        has_efficiency = False
        has_composite = False
        for scenario in filtered_json:
            if algo_name in filtered_json[scenario]:
                result = filtered_json[scenario][algo_name]
                if 'efficiency' in result['levels'] and result['levels']['efficiency'] != 'N/A':
                    has_efficiency = True
                if 'composite' in result['levels'] and result['levels']['composite'] != 'N/A':
                    has_composite = True
        
        # Create the appropriate table header based on available metrics
        header = "| Scenario | Performance |"
        if has_efficiency:
            header += " Efficiency |"
        if has_composite:
            header += " Overall Score |"
        output_text += header + "\n"
        
        # Create the separator line with the right number of columns
        separator = "|----------|------------|"
        if has_efficiency:
            separator += "------------|"
        if has_composite:
            separator += "------------|"
        output_text += separator + "\n"
        
        # Group scenarios to keep related ones together
        scenario_groups = {
            "general": [],
            "scaling": ["Variable Scaling", "Sample Scaling"],
            "function": ["Linear Function", "Non-Linear Function"],
            "noise": ["Gaussian Noise", "Non-Gaussian Noise"],
            "other": ["Heterogeneity", "Measurement Error", "Missing Data", "Edge Probability",
                    "Discrete Ratio", "Dense Graph", "Sparse Graph", 
                    "High Missing Data", "High Measurement Error", "Highly Heterogeneous"]
        }
        
        # First, determine which scenarios are actually available
        available_scenarios = {}
        for group, scenarios in scenario_groups.items():
            if group == "general":
                continue
            available = [s for s in scenarios if s in filtered_json and algo_name in filtered_json[s]]
            if available:
                available_scenarios[group] = available
            
        # Add scenarios that aren't in any specific group to general
        for scenario in filtered_json:
            is_in_group = False
            for group_scenarios in scenario_groups.values():
                if scenario in group_scenarios:
                    is_in_group = True
                    break
            if not is_in_group and algo_name in filtered_json[scenario]:
                scenario_groups["general"].append(scenario)
            
        # Now print tables in groups
        groups_to_print = ["scaling", "function", "noise", "other", "general"]
        
        has_printed_any_scenario = False
        for group in groups_to_print:
            scenarios = [s for s in scenario_groups[group] if s in filtered_json and algo_name in filtered_json[s]]
            if not scenarios:
                continue
                
            has_printed_any_scenario = True
                
            # Only add section headers for non-empty groups
            if group == "scaling":
                output_text += "\n• Scaling Scenarios\n"
            elif group == "function":
                output_text += "\n• Function Type Scenarios\n"
            elif group == "noise":
                output_text += "\n• Noise Type Scenarios\n"
            elif group == "other" and scenarios:
                output_text += "\n• Other Scenarios\n"
            elif group == "general" and scenarios:
                output_text += "\n• General Scenarios\n"
                
            for scenario in scenarios:
                result = filtered_json[scenario][algo_name]
                
                # Get performance level (treat 'N/A' appropriately)
                performance = result['levels'].get('performance', 'N/A')
                performance_str = f"{float(performance):.1f}" if performance != 'N/A' else 'N/A'
                
                # Create the row with performance
                row = f"| {scenario} | {performance_str} |"
                
                # Add efficiency if available
                if has_efficiency:
                    efficiency = result['levels'].get('efficiency', 'N/A')
                    eff_str = f"{float(efficiency):.1f}" if efficiency != 'N/A' else 'N/A'
                    row += f" {eff_str} |"
                
                # Add overall score (formerly composite) if available
                if has_composite:
                    composite = result['levels'].get('composite', 'N/A')
                    comp_str = f"{float(composite):.1f}" if composite != 'N/A' else 'N/A'
                    row += f" {comp_str} |"
                
                output_text += row + "\n"
        
        if not has_printed_any_scenario:
            output_text += "No scenario data available for this algorithm.\n"
        
        output_text += "\n"
    
    # Simplified scenario-specific results
    output_text += "\n\n────────────────────────────────────────────────────────\n"
    output_text += "Top and Bottom Performers by Scenario\n"
    output_text += "────────────────────────────────────────────────────────\n\n"
    
    for scenario in filtered_json:
        output_text += f"\n• {scenario}\n"
        
        # Sort algorithms by their ranking in this scenario
        sorted_algos = sorted(
            [(algo, filtered_json[scenario][algo]['ranking']['mean']) 
             for algo in filtered_json[scenario]],
            key=lambda x: x[1]
        )
        
        # Show top algorithms for each scenario (up to 5 or all if less than 5)
        output_text += "  Top performers:\n"
        top_count = min(5, len(sorted_algos))
        top_algos = sorted_algos[:top_count]
        for i, (algo_name, rank) in enumerate(top_algos):
            result = filtered_json[scenario][algo_name]
            perf = result['levels']['performance']
            perf_str = f"{float(perf):.1f}" if perf != 'N/A' else 'N/A'
            
            # For Variable Scaling and Sample Scaling, also show efficiency
            if scenario in ["Variable Scaling", "Sample Scaling"]:
                eff = result['levels'].get('efficiency', 'N/A')
                eff_str = f"{float(eff):.1f}" if eff != 'N/A' else 'N/A'
                output_text += f"  {i+1}. {algo_name}: Performance {perf_str}, Efficiency {eff_str}\n"
            else:
                output_text += f"  {i+1}. {algo_name}: Performance {perf_str}\n"
        
        # Show bottom algorithms for each scenario (up to 5 or all if less than 5)
        if len(sorted_algos) > 1:  # Only show if there's more than one algorithm
            output_text += "\n  Bottom performers:\n"
            bottom_count = min(5, len(sorted_algos))
            bottom_algos = sorted_algos[-bottom_count:] 
            bottom_algos.reverse()  # Display from worst to better
            for i, (algo_name, rank) in enumerate(bottom_algos):
                result = filtered_json[scenario][algo_name]
                perf = result['levels']['performance']
                perf_str = f"{float(perf):.1f}" if perf != 'N/A' else 'N/A'
                
                # For Variable Scaling and Sample Scaling, also show efficiency
                if scenario in ["Variable Scaling", "Sample Scaling"]:
                    eff = result['levels'].get('efficiency', 'N/A')
                    eff_str = f"{float(eff):.1f}" if eff != 'N/A' else 'N/A'
                    output_text += f"  {i+1}. {algo_name}: Performance {perf_str}, Efficiency {eff_str}\n"
                else:
                    output_text += f"  {i+1}. {algo_name}: Performance {perf_str}\n"
    
    # Save to file if output_filename is provided
    if output_filename:
        with open(output_filename, 'w') as f:
            f.write(output_text)
        print(f"Filtered benchmarking results saved to {output_filename}")
    
    return output_text

def create_ranking_benchmarking_results(benchmarking_json, algorithm_list=None, output_filename=None):
    """Create a consolidated benchmarking results text optimized for LLM readability,
    focusing on rankings rather than raw values.
    
    Args:
        benchmarking_json (dict): Dictionary containing benchmarking results
        algorithm_list (list): List of algorithm names to include in the results
        output_filename (str, optional): Name of the output file if saving is desired
    
    Returns:
        str: Formatted text containing the ranking-based benchmarking results
    """
    if algorithm_list is None or len(algorithm_list) == 0:
        print("No algorithms specified for filtering. Using all algorithms.")
        return None
    
    # Filter the benchmarking_json to only include the specified algorithms
    filtered_json = {}
    for scenario, algorithms in benchmarking_json.items():
        filtered_algorithms = {}
        for algo, data in algorithms.items():
            # Check if any algorithm in algorithm_list is a substring of algo
            if any(target_algo in algo for target_algo in algorithm_list):
                filtered_algorithms[algo] = data
        
        if filtered_algorithms:  # Only include scenarios that have at least one of the specified algorithms
            filtered_json[scenario] = filtered_algorithms
    
    if not filtered_json:
        return "No results found for the specified algorithms."
    
    # Create the output text
    output_text = ""
    output_text += "# ALGORITHM BENCHMARKING RESULTS\n\n"
    output_text += f"Algorithms included: {', '.join(algorithm_list)}\n\n"
    
    # Get all unique algorithm names across all scenarios
    all_algos = set()
    for scenario in filtered_json:
        all_algos.update(filtered_json[scenario].keys())
    all_algos = list(all_algos)
    
    # Format algorithm names to be more readable
    def format_algorithm_name(algo_name):
        # Handle names like 'FCI_indep_test=fisherz'
        if '_' in algo_name:
            parts = algo_name.split('_')
            base_algo = parts[0]
            params = []
            for part in parts[1:]:
                if '=' in part:
                    param, value = part.split('=')
                    # Format as "param: value" instead of "param=value"
                    params.append(f"{param}: {value}")
                else:
                    params.append(part)
            
            if params:
                return f"{base_algo} ({'_'.join(params)})"
            else:
                return base_algo
        return algo_name
    
    # 1. Overall algorithm ranking across all scenarios
    output_text += "## 1. OVERALL ALGORITHM RANKING\n\n"
    
    # Calculate average performance for each algorithm
    algo_avg_performance = {}
    for algo in all_algos:
        performances = []
        for scenario in filtered_json:
            if algo in filtered_json[scenario]:
                perf = filtered_json[scenario][algo]['levels'].get('performance', 'N/A')
                if perf != 'N/A':
                    performances.append(float(perf))
        
        if performances:
            algo_avg_performance[algo] = sum(performances) / len(performances)
    
    # Sort algorithms by average performance
    sorted_algos = sorted(algo_avg_performance.items(), key=lambda x: x[1], reverse=True)
    
    # Create a ranking table
    output_text += "| Rank | Algorithm | Overall Score |\n"
    output_text += "|------|-----------|---------------|\n"
    for i, (algo, avg_perf) in enumerate(sorted_algos):
        formatted_algo = format_algorithm_name(algo)
        output_text += f"| {i+1} | {formatted_algo} | {avg_perf:.1f} |\n"
    
    output_text += "\n"
    
    # 2. Best algorithm by scenario category
    output_text += "## 2. BEST ALGORITHM BY SCENARIO CATEGORY\n\n"
    
    # Define scenario categories
    scenario_categories = {
        "Linear Function": "Function Type",
        "Non-Linear Function": "Function Type",
        "Gaussian Noise": "Noise Type",
        "Non-Gaussian Noise": "Noise Type",
        "Variable Scaling": "Scaling",
        "Sample Scaling": "Scaling",
        "Dense Graph": "Graph Structure",
        "Sparse Graph": "Graph Structure",
        "High Missing Data": "Data Quality",
        "High Measurement Error": "Data Quality",
        "Highly Heterogeneous": "Data Quality",
        "Heterogeneity": "Data Quality",
        "Measurement Error": "Data Quality",
        "Missing Data": "Data Quality",
        "Edge Probability": "Graph Structure",
        "Discrete Ratio": "Data Type"
    }
    
    # Get unique categories
    categories = sorted(set(scenario_categories.values()))
    
    # Find best algorithm for each category
    for category in categories:
        output_text += f"### {category}\n\n"
        output_text += "| Scenario | Best Algorithm | Score | Runner-up | Score |\n"
        output_text += "|----------|---------------|-------|-----------|-------|\n"
        
        # Get all scenarios in this category
        category_scenarios = [s for s in filtered_json if any(s.startswith(sc) for sc in scenario_categories if scenario_categories[sc] == category)]
        
        # Process each scenario
        for scenario in sorted(category_scenarios):
            # Sort algorithms by performance for this scenario
            scenario_algos = []
            for algo, data in filtered_json[scenario].items():
                perf = data['levels'].get('performance', 'N/A')
                if perf != 'N/A':
                    scenario_algos.append((algo, float(perf)))
            
            if not scenario_algos:
                continue
                
            # Sort algorithms by performance (highest first)
            scenario_algos.sort(key=lambda x: x[1], reverse=True)
            
            # Get the best and runner-up algorithms
            best_algo, best_score = scenario_algos[0]
            runner_up = "N/A"
            runner_up_score = "N/A"
            if len(scenario_algos) > 1:
                runner_up, runner_up_score = scenario_algos[1]
                
            formatted_best_algo = format_algorithm_name(best_algo)
            formatted_runner_up = format_algorithm_name(runner_up) if runner_up != "N/A" else "N/A"
            
            output_text += f"| {scenario} | {formatted_best_algo} | {best_score:.1f} | {formatted_runner_up} | {runner_up_score if runner_up_score == 'N/A' else f'{runner_up_score:.1f}'} |\n"
        
        output_text += "\n"
    
    # 3. Algorithm Performance Matrix
    output_text += "## 3. ALGORITHM PERFORMANCE MATRIX\n\n"
    output_text += "Performance rating: Excellent (9-10), Good (7-8.9), Moderate (5-6.9), Poor (3-4.9), Very Poor (0-2.9)\n\n"
    
    # Group scenarios by type
    scenario_groups = {
        "Function Type": ["Linear Function", "Non-Linear Function"],
        "Noise Type": ["Gaussian Noise", "Non-Gaussian Noise"],
        "Scaling": ["Variable Scaling", "Sample Scaling"],
        "Graph Structure": ["Dense Graph", "Sparse Graph", "Edge Probability"],
        "Data Quality": ["High Missing Data", "High Measurement Error", "Highly Heterogeneous", 
                         "Heterogeneity", "Measurement Error", "Missing Data"],
        "Data Type": ["Discrete Ratio"]
    }
    
    # Create a matrix table with algorithms as rows and scenario types as columns
    output_text += "| Algorithm |"
    for group_name in scenario_groups:
        output_text += f" {group_name} |"
    output_text += "\n|-----------|"
    for _ in scenario_groups:
        output_text += "------------|"
    output_text += "\n"
    
    # For each algorithm, calculate average performance for each scenario group
    for algo in sorted(all_algos):
        formatted_algo = format_algorithm_name(algo)
        output_text += f"| {formatted_algo} |"
        
        for group_name, group_scenarios in scenario_groups.items():
            # Calculate average performance for this algorithm in this scenario group
            performances = []
            for scenario in filtered_json:
                if algo in filtered_json[scenario] and any(scenario.startswith(s) for s in group_scenarios):
                    perf = filtered_json[scenario][algo]['levels'].get('performance', 'N/A')
                    if perf != 'N/A':
                        performances.append(float(perf))
            
            if performances:
                avg_perf = sum(performances) / len(performances)
                # Convert numerical score to rating
                if avg_perf >= 9:
                    rating = "Excellent"
                elif avg_perf >= 7:
                    rating = "Good"
                elif avg_perf >= 5:
                    rating = "Moderate"
                elif avg_perf >= 3:
                    rating = "Poor"
                else:
                    rating = "Very Poor"
                output_text += f" {rating} ({avg_perf:.1f}) |"
            else:
                output_text += " N/A |"
        
        output_text += "\n"
    
    output_text += "\n"
    
    # 4. Special Considerations
    output_text += "## 4. SPECIAL CONSIDERATIONS\n\n"
    
    # Efficiency analysis for scaling scenarios
    output_text += "### Efficiency Analysis (for scaling scenarios)\n\n"
    output_text += "| Algorithm | Efficiency Score | Notes |\n"
    output_text += "|-----------|-----------------|-------|\n"
    
    scaling_scenarios = ["Variable Scaling", "Sample Scaling"]
    for algo in sorted(all_algos):
        efficiency_scores = []
        for scenario in filtered_json:
            if (any(scenario.startswith(s) for s in scaling_scenarios) and 
                algo in filtered_json[scenario] and
                'efficiency' in filtered_json[scenario][algo]['levels'] and 
                filtered_json[scenario][algo]['levels']['efficiency'] != 'N/A'):
                efficiency_scores.append(float(filtered_json[scenario][algo]['levels']['efficiency']))
        
        if efficiency_scores:
            avg_efficiency = sum(efficiency_scores) / len(efficiency_scores)
            if avg_efficiency >= 4.5:
                notes = "Excellent efficiency, suitable for large-scale problems"
            elif avg_efficiency >= 3.5:
                notes = "Good efficiency, handles most scaling challenges well"
            elif avg_efficiency >= 2.5:
                notes = "Moderate efficiency, may struggle with very large problems"
            elif avg_efficiency >= 1.5:
                notes = "Poor efficiency, not recommended for large-scale problems"
            else:
                notes = "Very poor efficiency, will likely fail on large-scale problems"
            
            formatted_algo = format_algorithm_name(algo)
            output_text += f"| {formatted_algo} | {avg_efficiency:.1f} | {notes} |\n"
    
    output_text += "\n"
    
    # 5. Recommendations
    output_text += "## 5. ALGORITHM RECOMMENDATIONS\n\n"
    
    # Best overall algorithm
    if sorted_algos:
        best_algo, _ = sorted_algos[0]
        formatted_best_algo = format_algorithm_name(best_algo)
        output_text += f"- **Best Overall Algorithm**: {formatted_best_algo}\n"
    
    # Best algorithms for specific scenario types
    output_text += "- **Best for Specific Use Cases**:\n"
    
    for category in categories:
        category_scores = {}
        
        # Get all scenarios in this category
        category_scenarios = [s for s in filtered_json if any(s.startswith(sc) for sc in scenario_categories if scenario_categories[sc] == category)]
        
        # Calculate average performance for each algorithm in this category
        for algo in all_algos:
            scores = []
            for scenario in category_scenarios:
                if algo in filtered_json[scenario]:
                    perf = filtered_json[scenario][algo]['levels'].get('performance', 'N/A')
                    if perf != 'N/A':
                        scores.append(float(perf))
            
            if scores:
                category_scores[algo] = sum(scores) / len(scores)
        
        # Find the best algorithm for this category
        if category_scores:
            best_algo_for_category = max(category_scores.items(), key=lambda x: x[1])
            formatted_best_algo = format_algorithm_name(best_algo_for_category[0])
            output_text += f"  - **{category}**: {formatted_best_algo} (Score: {best_algo_for_category[1]:.1f})\n"
    
    # Save to file if output_filename is provided
    if output_filename:
        with open(output_filename, 'w') as f:
            f.write(output_text)
        print(f"Ranking benchmarking results saved to {output_filename}")
    
    return output_text

if __name__ == "__main__":
    benchmarking_json = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "benchmarking", "algorithm_performance_analysis.json")))
    add_benchmarking_results_to_all(benchmarking_json, os.path.join(os.path.dirname(os.path.dirname(__file__))))
    create_consolidated_benchmarking_results(benchmarking_json, os.path.join(os.path.dirname(os.path.dirname(__file__))))
