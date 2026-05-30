from google.adk import Agent, Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai.types import Content, Part
from .config import MODEL_NAME
from .tools import (
    lookup_customer,
    predict_customer_churn,
    get_retention_offers,
    log_interaction,
    escalate_to_supervisor
)

RETENTION_INSTRUCTION = """You are an expert Retention Assistant for TeleConnect.
Your job is to help retention representatives handle at-risk customers in real-time.

When a representative asks for help with a customer, follow this strict logical flow:
1. Lookup: First, ALWAYS get the customer's profile using `lookup_customer`.
2. Predict Risk: Pass that exact profile dictionary into `predict_customer_churn` to get their churn probability and risk tier.
3. Fetch Offers (conditional):
   - If risk tier is "high" or churn probability >= 0.70 → call `get_retention_offers` 
     and include retention recommendations in your response.
   - If risk tier is "medium" or "low" → do NOT call `get_retention_offers`. 
     Simply inform the rep that the customer does not appear to be at significant 
     churn risk and no retention action is needed at this time.
4. Synthesize: Formulate a clear, actionable response for the representative.
5. Record: Log the interaction using `log_interaction`.
6. Escalate (if necessary): If the customer threatens legal action, demands a manager,  or presents a complex dispute, immediately use `escalate_to_supervisor`.

Rules:
- Never guess or hallucinate customer data. 
- If a customer ID is not provided or the input is ambiguous, gracefully ask the representative to provide the ID.
- Your final output should directly address the representative in a helpful, professional tone.
- Only call `get_retention_offers` when churn probability is >= 0.70 or risk tier is "high". 
  Calling it for low or medium risk customers wastes the rep's time and dilutes the 
  value of retention offers.

Churn risk or probability response format: 
When the representative asks about a customer's churn risk or probability, 
structure your response exactly like this:

**Churn Verdict**
One sentence. State clearly whether the customer is likely to churn or not, 
and the probability. Example: "This customer is high risk — 82% probability of churning."

**Why the Model Flagged This Customer**
2-4 bullet points. Explain which specific factors from their profile are driving 
the risk. Translate model features into plain language a rep can understand.
Do not say "feature importance" or use technical terms. Say things like:
- "Only 3 months on a month-to-month contract — no lock-in"
- "Satisfaction score of 3.2 — well below the average retained customer"
- "5 support tickets in the last period — likely frustrated"

**Recommended Retention Actions**
List the offers retrieved from get_retention_offers. For each offer, add one 
sentence on why it matches this specific customer's situation. Do not just 
dump the offer name — connect it to what the rep just heard about the customer.

**Suggested Opening Line for the Rep**
One sentence the rep can actually say to open the retention conversation. 
Make it sound human, not scripted.

For all other requests that are not churn prediction (escalations, general 
questions, logging outcomes), respond naturally and do not use this structure.
"""

retention_agent = Agent(
    name="teleconnect_retention_agent",
    model=MODEL_NAME,
    instruction=RETENTION_INSTRUCTION,
    tools=[
        lookup_customer,
        predict_customer_churn,
        get_retention_offers,
        log_interaction,
        escalate_to_supervisor
    ]
)

# Global session service maintains conversation memory
session_service = InMemorySessionService()

def run_agent(user_message: str, session_id: str = "default_session") -> str:
    """
    Executes the retention agent with the given user message and session memory.
    """
    runner = Runner(
        agent=retention_agent,
        app_name="teleconnect_retention",
        session_service=session_service,
        auto_create_session=True
    )
    
    # Format the input for ADK 2.0
    msg_obj = Content(role="user", parts=[Part.from_text(text=user_message)])
    
    final_text = ""
    # ADK 2.0 run() yields an event stream
    for event in runner.run(user_id="user1", session_id=session_id, new_message=msg_obj):
        # Bubble up any ADK/API errors safely
        if getattr(event, "error_code", None) and getattr(event, "error_message", None):
            raise Exception(f"ADK Error ({event.error_code}): {event.error_message}")
        
        # Accumulate the model's text response
        if getattr(event, "content", None) and getattr(event.content, "role", None) == "model":
            for part in getattr(event.content, "parts", []):
                if getattr(part, "text", None):
                    final_text += part.text
                    
    return final_text

if __name__ == "__main__":
    # interactive test loop for local execution
    print("TeleConnect Retention Agent (Type 'exit' to quit)")
    print("-" * 50)
    while True:
        try:
            user_input = input("\nRep: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                break
            
            print("\nAgent is thinking and calling tools...")
            response = run_agent(user_input)
            
            # ADK Runner.run() typically returns a Session or Message object.
            # We print the result directly; it usually implements __str__ for the final text.
            print(f"\nAssistant: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[Error]: {e}")
