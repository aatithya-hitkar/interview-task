def calculate_tool_selection_accuracy(expected_tools: list, actual_tools: list) -> float:
    """
    Calculates Tool Selection Accuracy.
    Returns 1.0 if all expected tools were called, 0.0 otherwise.
    (We ensure that the agent calls at least the required sequence).
    """
    if not expected_tools and not actual_tools:
        return 1.0
    if not expected_tools and actual_tools:
        return 0.0
        
    actual_names = [t.get("tool") for t in actual_tools]
    
    # Check if all expected tools are in actual tools
    matches = sum(1 for et in expected_tools if et in actual_names)
    return matches / len(expected_tools) if expected_tools else 1.0

def calculate_parameter_extraction_accuracy(expected_id: str, actual_tools: list) -> float:
    """
    Calculates Parameter Extraction Accuracy.
    Verifies if the agent successfully extracted the correct customer_id and passed it to the tools.
    """
    if not expected_id:
        return 1.0 # Not expected to extract a parameter
    
    if not actual_tools:
        return 0.0 # Failed to call tools to extract parameter
        
    # Check if any tool got the right ID
    for t in actual_tools:
        if t.get("customer_id") == expected_id:
            return 1.0
    return 0.0
    
def measure_latency(start_time: float, end_time: float) -> float:
    """
    Calculates the latency of the agent execution in seconds.
    """
    return round(end_time - start_time, 2)
