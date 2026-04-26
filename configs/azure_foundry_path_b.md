# Path B: Azure AI Foundry Agents Implementation
=================================================

## Key Differences from LangGraph (Path A)

| Aspect | Path A (LangGraph) | Path B (Azure Foundry) |
|--------|-------------------|------------------------|
| State | Explicit TypedDict | Azure Thread object |
| Memory | File-backed JSON store | Azure Cosmos / Table Storage |
| HITL | Conditional edge + terminal | human_input_required tool call |
| Observability | LangSmith / custom | Azure Monitor + App Insights |
| Critique | Separate graph node | Tool call in single agent |

## Setup

```bash
pip install azure-ai-projects azure-identity
export AZURE_AI_PROJECT_ENDPOINT="https://your-project.api.azureml.ms"
export AZURE_SUBSCRIPTION_ID="..."
export AZURE_RESOURCE_GROUP="..."
```

## Core Agent Implementation

```python
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    Agent, Thread, Message, MessageRole,
    FunctionTool, ToolSet
)
from azure.identity import DefaultAzureCredential

client = AIProjectClient(
    endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
    credential=DefaultAzureCredential()
)

# Define tools
episodic_memory_tool = FunctionTool(
    name="load_patient_history",
    description="Load all prior visit records for a patient from episodic memory",
    parameters={
        "type": "object",
        "properties": {"patient_id": {"type": "string"}},
        "required": ["patient_id"]
    }
)

semantic_search_tool = FunctionTool(
    name="search_guidelines",
    description="Search medical guideline KB for relevant protocols",
    parameters={
        "type": "object", 
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 3}
        }
    }
)

critique_tool = FunctionTool(
    name="run_critique",
    description="Run safety critique: check contraindications, allergies, drug interactions",
    parameters={
        "type": "object",
        "properties": {
            "patient_id": {"type": "string"},
            "proposed_treatment": {"type": "string"},
            "current_labs": {"type": "object"}
        }
    }
)

toolset = ToolSet()
toolset.add(episodic_memory_tool)
toolset.add(semantic_search_tool)
toolset.add(critique_tool)

# Create agent
agent = client.agents.create_agent(
    model="gpt-4o",
    name="ClinicalDSS-Foundry",
    instructions=CLINICAL_SYSTEM_PROMPT,
    toolset=toolset
)

# Create thread per patient visit
thread = client.agents.create_thread()

# Add visit message
message = client.agents.create_message(
    thread_id=thread.id,
    role=MessageRole.USER,
    content=format_visit_prompt(patient, visit)
)

# Run with auto tool execution
run = client.agents.create_and_process_run(
    thread_id=thread.id,
    agent_id=agent.id,
    instructions="Use all available tools for complete assessment"
)

# Get final response
messages = client.agents.list_messages(thread_id=thread.id)
final_response = messages.get_last_text_message_by_role(MessageRole.ASSISTANT)
```

## HITL with Foundry

```python
from azure.ai.projects.models import RequiredFunctionToolCall

run = client.agents.create_run(
    thread_id=thread.id,
    agent_id=agent.id
)

# Poll for HITL trigger
while run.status in ["queued", "in_progress", "requires_action"]:
    if run.status == "requires_action":
        tool_calls = run.required_action.submit_tool_outputs.tool_calls
        tool_outputs = []
        for tc in tool_calls:
            if tc.function.name == "request_physician_review":
                # Surface to physician UI
                physician_response = get_physician_input(tc.function.arguments)
                tool_outputs.append({
                    "tool_call_id": tc.id,
                    "output": physician_response
                })
        run = client.agents.submit_tool_outputs_to_run(
            thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs
        )
    else:
        time.sleep(0.5)
        run = client.agents.get_run(thread_id=thread.id, run_id=run.id)
```
