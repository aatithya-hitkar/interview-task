import os
import json
from google import genai
from google.genai import types
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from config import GEMINI_API_KEY, MODEL_NAME

# Initialize the Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)

JUDGE_PROMPT = """
You are an expert AI quality assurance judge. Your task is to evaluate an AI Retention Agent's response to a customer service representative's query.

You will be given:
1. The User Input (the representative's query).
2. The Expected Quality Criteria.
3. The Agent's Actual Response.

You must score the Agent's Response across 4 dimensions on a strict 1, 3, 5 scale based on the following anchors:

# Dimension 1: Factual Correctness
- 5: The response is perfectly accurate according to the tools/expected criteria and does not contradict known information.
- 3: The response is mostly accurate but omits a minor detail or is slightly vague.
- 1: The response is factually incorrect or contradicts the expected criteria.

# Dimension 2: Tool Use Appropriateness
- 5: The agent correctly references the tools it should have used (e.g. clearly indicates it looked up the profile, checked churn, found offers, escalated).
- 3: The agent used tools but the output suggests it might have missed one step (e.g. predicted churn but forgot to mention offers when asked).
- 1: The agent completely failed to reference the required actions (e.g. didn't escalate when it should have).

# Dimension 3: Actionability
- 5: The response provides a clear, highly actionable recommendation or next step for the human representative.
- 3: The response gives good information but leaves the representative slightly unsure of the exact next step.
- 1: The response is a raw data dump or completely useless for a representative on a live call.

# Dimension 4: Hallucination
- 5: Zero hallucination. The agent strictly sticks to provided data or explicitly states if data is missing.
- 3: Mild hallucination (e.g., invents a minor, harmless conversational detail).
- 1: Severe hallucination (e.g., invents a fake customer ID, fake risk score, or fake retention offer).

Respond strictly in valid JSON format matching this schema exactly:
{
  "factual_correctness": <int>,
  "tool_use": <int>,
  "actionability": <int>,
  "hallucination": <int>,
  "reasoning": "<string summarizing why you gave these scores>"
}
"""

def evaluate_response(user_input: str, expected_criteria: str, actual_response: str) -> dict:
    """
    Evaluates an agent's response using an LLM-as-Judge.
    
    Reliability Meta-Question Discussion:
    To ensure this judge is reliable:
    1. We use anchored scoring (1, 3, 5) rather than a continuous 1-10 scale. Anchors drastically reduce inter-rater variance and prevent "positivity bias" (where LLMs tend to default to 7 or 8 out of 10).
    2. We explicitly instruct the LLM on exactly what constitutes a failure (e.g., defining "Severe hallucination").
    3. The prompt asks for reasoning alongside the scores, which forces the LLM to justify its evaluation (Chain-of-Thought), increasing calibration accuracy against human labels.
    """
    
    content = f"USER INPUT:\n{user_input}\n\nEXPECTED CRITERIA:\n{expected_criteria}\n\nAGENT RESPONSE:\n{actual_response}"
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Content(role="user", parts=[types.Part.from_text(text=JUDGE_PROMPT + "\n\n" + content)])
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error calling LLM Judge: {e}")
        return {
            "factual_correctness": 0,
            "tool_use": 0,
            "actionability": 0,
            "hallucination": 0,
            "reasoning": f"Judge error: {str(e)}"
        }
