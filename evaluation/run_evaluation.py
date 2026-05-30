import json
import time
import uuid
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from agent import run_agent
from tools import get_and_clear_trace
from eval_metrics import calculate_tool_selection_accuracy, calculate_parameter_extraction_accuracy, measure_latency
from llm_judge import evaluate_response

def run_evaluations():
    print("Loading test suite...")
    suite_path = os.path.join(os.path.dirname(__file__), "test_suite.json")
    with open(suite_path, "r") as f:
        test_suite = json.load(f)
        
    results = []
    
    for case in test_suite:
        print(f"\nEvaluating Case: {case['id']} ({case['category']})")
        
        # Ensure clean state for tools
        get_and_clear_trace()
        
        # Unique session to prevent memory bleed between tests
        session_id = str(uuid.uuid4())
        
        # Measure latency
        start_time = time.time()
        try:
            agent_response = run_agent(case['user_input'], session_id=session_id)
        except Exception as e:
            agent_response = f"Agent Failed: {str(e)}"
        end_time = time.time()
        
        latency = measure_latency(start_time, end_time)
        actual_tools = get_and_clear_trace()
        
        # Calculate automated metrics
        tool_acc = calculate_tool_selection_accuracy(case['expected_tool_calls'], actual_tools)
        param_acc = calculate_parameter_extraction_accuracy(case['expected_customer_id'], actual_tools)
        
        # LLM as Judge
        judge_scores = evaluate_response(
            user_input=case['user_input'],
            expected_criteria=case['quality_criteria'],
            actual_response=agent_response
        )
        
        # Compile result
        case_result = {
            "id": case['id'],
            "category": case['category'],
            "latency_seconds": latency,
            "tool_selection_accuracy": tool_acc,
            "parameter_extraction_accuracy": param_acc,
            "judge_factual_correctness": judge_scores.get("factual_correctness", 0),
            "judge_tool_use": judge_scores.get("tool_use", 0),
            "judge_actionability": judge_scores.get("actionability", 0),
            "judge_hallucination": judge_scores.get("hallucination", 0),
            "judge_reasoning": judge_scores.get("reasoning", ""),
            "raw_agent_response": agent_response,
            "raw_tools_called": [t.get("tool") for t in actual_tools]
        }
        
        results.append(case_result)
        print(f"Finished {case['id']} - Latency: {latency}s | Tool Acc: {tool_acc} | Judge Score (Factual): {case_result['judge_factual_correctness']}")
        
    # Save raw JSON results
    results_path = os.path.join(os.path.dirname(__file__), "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
        
    # Save scorecard CSV
    df = pd.DataFrame(results)
    
    # Calculate Summary Scorecard
    summary = {
        "Total Cases": len(df),
        "Avg Latency (s)": round(df["latency_seconds"].mean(), 2),
        "Avg Tool Selection Acc": round(df["tool_selection_accuracy"].mean(), 2),
        "Avg Parameter Extraction Acc": round(df["parameter_extraction_accuracy"].mean(), 2),
        "Avg Judge Factual Correctness (1-5)": round(df["judge_factual_correctness"].mean(), 2),
        "Avg Judge Tool Use (1-5)": round(df["judge_tool_use"].mean(), 2),
        "Avg Judge Actionability (1-5)": round(df["judge_actionability"].mean(), 2),
        "Avg Judge Hallucination (1-5)": round(df["judge_hallucination"].mean(), 2),
    }
    
    print("\n--- AGGREGATE SCORECARD ---")
    for k, v in summary.items():
        print(f"{k}: {v}")
        
    df.to_csv(os.path.join(os.path.dirname(__file__), "evaluation_scorecard.csv"), index=False)
    print("\nResults saved to evaluation_results.json and evaluation_scorecard.csv")

if __name__ == "__main__":
    run_evaluations()
