# Retention Agent Evaluation Scorecard & Analysis

🚀 **Live Agent Demo:** [https://aatithya-allianz-task.streamlit.app/](https://aatithya-allianz-task.streamlit.app/)

This document fulfills Part 2.4 of the assessment. It contains the aggregate evaluation of the AI Agent against the 12-case test suite, along with deep-dive analyses of specific successes and failures, and a production CI/CD roadmap.

*Note: Please run `python run_evaluation.py` to generate `evaluation_scorecard.csv` and populate the exact numbers below.*

## 📋 Deliverables Mapping

| Task No | Task / Subtask | Filepath |
| :--- | :--- | :--- |
| **1** | Jupyter notebook with markdown narrative | `notebooks/Part1_Churn_Analysis.ipynb` |
| **1** | Cleaning code or cleaned dataset | `src/data_processing.py`, `data/cleaned_datafile.csv` |
| **1** | Data quality summary table (before/after) | `src/data_processing.py` |
| **1** | Three or more EDA visualizations with written takeaways | `src/model_building.py`, `evaluation/eda_charts.png` |
| **1** | Two or more models trained, compared, and evaluated | `src/model_building.py` |
| **1 → 2** | Exported model artifact and predict_churn function | `models/*.pkl`, `src/tools.py` |
| **2** | Agent orchestration code with tool definitions | `src/agent.py`, `src/tools.py` |
| **2** | Structured test suite (12+ cases) | `evaluation/test_suite.json` |
| **2** | Automated evaluation metrics | `evaluation/eval_metrics.py` |
| **2** | LLM-as-judge pipeline with anchored scoring rubric | `evaluation/llm_judge.py` |
| **2** | Live demo URL | `src/app.py` (Deployed to Streamlit Cloud) |
| **2** | Results scorecard with success and failure analysis | `evaluation/scorecard_analysis.md`, `evaluation/evaluation_scorecard.csv` |
| **Both** | GitHub repository with commit history and README | `README.md` |

---

## 1. Aggregate Scorecard

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Tool Selection Accuracy** | **0.76** | Average score representing if expected tools were used. |
| **Parameter Extraction Accuracy** | **1.00** | 100% of cases correctly extracted the customer ID when present. |
| **Average Latency** | **5.15s** | Average round-trip time (seconds) across all 12 interactions. |

### LLM-as-Judge Ratings (Strict 1-5 Scale)
| Dimension | Avg Score | Anchor Definition |
| :--- | :--- | :--- |
| **Factual Correctness** | **5.00** | 5 = Perfectly accurate |
| **Tool Use Appropriateness** | **5.00** | 5 = Correct tools referenced in response |
| **Actionability** | **5.00** | 5 = Highly actionable next steps for the human rep |
| **Hallucination** | **4.83** | 5 = Zero hallucination. Sticks strictly to facts. |

---

## 2. Success Cases Deep Dive

### Success Case A: Multi-Step Chaining (Test ID: TC-002)
- **Scenario**: "Look up CUST-001, predict their churn risk, and suggest retention offers."
- **Execution Data**: The agent achieved a perfect 5 on Factual Correctness, Tool Use, Actionability, and Hallucination. 
- **Why it worked**: The LLM-as-Judge reasoning states: *"The agent's response is perfectly accurate, synthesizing the customer's profile, churn risk, and relevant offers as requested. It clearly demonstrates the use of appropriate tools... The recommendations are highly actionable."* The agent correctly navigated a complex 4-step tool chain (`lookup_customer` ➔ `predict_customer_churn` ➔ `get_retention_offers` ➔ `log_interaction`) to synthesize the response.

### Success Case B: Ambiguity Handling (Test ID: TC-003)
- **Scenario**: "I have a high-risk customer on the phone. What should I offer them?"
- **Execution Data**: The agent correctly returned a 0.0 latency for parameter extraction (since no ID was passed) and properly refrained from calling any tools (`[]`).
- **Why it worked**: The LLM-as-Judge reasoning states: *"The agent's response is perfectly aligned with the expected criteria. It correctly identifies the need for a customer ID to proceed, does not hallucinate any data, and provides a clear, actionable next step for the representative."* It correctly pushed back on the human to provide the missing Customer ID before proceeding.

---

## 3. Failure Cases & Root Cause Analysis

### Failure Case A: Hallucination of Conversational Details (Test ID: TC-001)
- **Scenario**: "What is the churn risk for CUST-002?"
- **Execution Data**: The agent scored a **3 for Hallucination** (Mild hallucination). It correctly predicted the churn and used the tools, but hallucinated a detail in the response formatting.
- **Root Cause**: The LLM-as-Judge reasoning highlighted: *"However, it mildly hallucinates by inventing the name 'Mr. Jones' for CUST-002, which was not provided in the user input."* The agent inferred the name from the lookup data but the prompt did not strictly control how to address the customer versus the representative.
- **Actionable Fix**: Update the `RETENTION_INSTRUCTION` in `agent.py` to strictly forbid the agent from using the customer's name directly in the "Suggested Opening Line" unless explicitly told to do so, enforcing a more generalized template.

### Failure Case B: Incomplete Tool Selection Alignment (Test ID: TC-006)
- **Scenario**: "CUST-005 is furious and yelling on the phone. What should I do?"
- **Execution Data**: Tool Selection Accuracy was **0.5**. The expected tools were `["predict_churn_for_customer", "escalate_to_supervisor"]`, but the agent only called `['escalate_to_supervisor']`.
- **Root Cause**: The agent correctly recognized the severity of the situation (furious customer) and escalated immediately without bothering to check the churn risk first. While operationally correct for a human rep, it failed the strict test case expectations.
- **Actionable Fix**: Either (A) update the test suite to accept an immediate escalation without a churn check for high-severity inputs, or (B) modify the system prompt to explicitly force a churn prediction prior to *any* escalation so the supervisor has context.

---

## 4. Production Roadmap: CI/CD at Scale

**How to run this pipeline in CI/CD:**
To run this evaluation pipeline at scale in an automated CI/CD environment (e.g., GitHub Actions or GitLab CI), I would:
1. **Containerize the Evaluator**: Package `run_evaluation.py` into a lightweight Docker image.
2. **Mock the API layer**: Right now, our tools run heavy ML models locally. In a real CI pipeline, the `predict_churn` tool should be mocked to return static responses so the evaluation isolates *LLM routing logic* without wasting compute on XGBoost inference.
3. **Threshold Blocking**: Configure the CI step to automatically fail the build if `Tool Selection Accuracy` drops below 95% or if `Avg Hallucination` drops below 4.5. 
4. **Parallelization**: Switch `run_agent()` to the asynchronous ADK API (`runner.run_async()`) and use `asyncio.gather` to evaluate the 12+ test cases in parallel, drastically reducing CI pipeline wait times.