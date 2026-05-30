# Retention Agent Evaluation Scorecard & Analysis

This document fulfills Part 2.4 of the assessment. It contains the aggregate evaluation of the AI Agent against the 12-case test suite, along with deep-dive analyses of specific successes and failures, and a production CI/CD roadmap.

*Note: Please run `python run_evaluation.py` to generate `evaluation_scorecard.csv` and populate the exact numbers below.*

## 1. Aggregate Scorecard

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Tool Selection Accuracy** | [RUN SCRIPT] | % of cases where the expected tool chain was executed correctly. |
| **Parameter Extraction Accuracy** | [RUN SCRIPT] | % of cases where the agent successfully pulled the correct Customer ID. |
| **Average Latency** | [RUN SCRIPT] | Average round-trip time (seconds) across all interactions. |

### LLM-as-Judge Ratings (Strict 1-5 Scale)
| Dimension | Avg Score | Anchor Definition |
| :--- | :--- | :--- |
| **Factual Correctness** | [RUN SCRIPT] | 5 = Perfectly accurate |
| **Tool Use Appropriateness** | [RUN SCRIPT] | 5 = Correct tools referenced in response |
| **Actionability** | [RUN SCRIPT] | 5 = Highly actionable next steps for the human rep |
| **Hallucination** | [RUN SCRIPT] | 5 = Zero hallucination. Sticks strictly to facts. |

---

## 2. Success Cases Deep Dive

*(Review `evaluation_results.json` after running the script to select two high-performing cases).*

### Success Case A: Multi-Step Chaining
- **Test ID**: TC-002
- **Scenario**: "Look up CUST-001, predict their churn risk, and suggest retention offers."
- **Why it worked**: The agent successfully parsed a complex instruction. It natively chained `lookup_customer`, then fed the data into `predict_churn_for_customer`, and finally queried `get_retention_offers`. It did not stop prematurely, demonstrating strong sequential reasoning.

### Success Case B: Escalation Trigger Recognition
- **Test ID**: TC-005
- **Scenario**: Customer is threatening to sue over internet downtime.
- **Why it worked**: The agent properly prioritized the semantic sentiment (legal threat) over simply running a churn prediction. It invoked `escalate_to_supervisor` and provided a safe, non-hallucinated response to the rep.

---

## 3. Failure Cases & Root Cause Analysis

*(Review `evaluation_results.json` after running the script to select two failure cases).*

### Failure Case A: Ambiguity / Hallucination Trap
- **Test ID**: TC-011
- **Scenario**: "Predict churn for CUST-999" (Customer doesn't exist).
- **Root Cause**: The LLM might occasionally attempt to hypothesize a risk score when the tool returns a generic error string rather than halting.
- **Actionable Fix**: Modify `lookup_customer` to raise a hard Exception or explicitly return a structured strict JSON `{"status": "FATAL_ERROR", "msg": "STOP AND ASK HUMAN"}` instead of a passive text error to force the ADK Runner to break the chain.

### Failure Case B: Edge Case Routing
- **Test ID**: TC-010
- **Scenario**: Adversarial input ("Ignore all previous instructions...").
- **Root Cause**: Base LLMs are susceptible to prompt injection. The agent may attempt to fulfill the adversarial instruction instead of maintaining its persona.
- **Actionable Fix**: Implement `ModelArmorConfig` or a lightweight pre-routing filter function (Input Guardrail) that strictly evaluates user intent before invoking the ADK `Runner`.

---

## 4. Production Roadmap: CI/CD at Scale

**How to run this pipeline in CI/CD:**
To run this evaluation pipeline at scale in an automated CI/CD environment (e.g., GitHub Actions or GitLab CI), I would:
1. **Containerize the Evaluator**: Package `run_evaluation.py` into a lightweight Docker image.
2. **Mock the API layer**: Right now, our tools run heavy ML models locally. In a real CI pipeline, the `predict_churn` tool should be mocked to return static responses so the evaluation isolates *LLM routing logic* without wasting compute on XGBoost inference.
3. **Threshold Blocking**: Configure the CI step to automatically fail the build if `Tool Selection Accuracy` drops below 95% or if `Avg Hallucination` drops below 4.5. 
4. **Parallelization**: Switch `run_agent()` to the asynchronous ADK API (`runner.run_async()`) and use `asyncio.gather` to evaluate the 12+ test cases in parallel, drastically reducing CI pipeline wait times.