from typing import TypedDict, List, Annotated, Dict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class ResearchState(TypedDict):
    initial_query: str
    current_task: str
    research_history: Annotated[List[BaseMessage], add_messages]
    accumulated_findings: str
    final_report: str
    max_iterations: int
    current_iteration: int
