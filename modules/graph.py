from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from modules.agents import rag_agent, summarization_agent, reasoning_agent, llm

# CHANGED: Added 'file_names' to the state
class AgentState(TypedDict):
    question: str
    messages: List[str]
    documents: List[Document]
    intent: str
    file_names: List[str] # New field to track all uploaded files

def plan_route(state):
    question = state["question"]
    system_prompt = "Classify user intent as 'summarize', 'reason' (for comparison/timeline), or 'rag' (for lookup). Return ONLY the word."
    response = llm.invoke(f"{system_prompt}\nUser Input: {question}")
    intent = response.content.strip().lower()
    
    if intent not in ["summarize", "reason", "rag"]: intent = "rag"
    
    print(f"--- PLANNER DECISION: {intent} ---")
    return {"intent": intent}

workflow = StateGraph(AgentState)
workflow.add_node("planner", plan_route)
workflow.add_node("rag_node", rag_agent)
workflow.add_node("summarize_node", summarization_agent)
workflow.add_node("reason_node", reasoning_agent)

workflow.set_entry_point("planner")

def route_from_planner(state):
    if state["intent"] == "summarize": return "summarize_node"
    return "rag_node"

workflow.add_conditional_edges("planner", route_from_planner)

def route_from_rag(state):
    if state["intent"] == "reason": return "reason_node"
    return END

workflow.add_conditional_edges("rag_node", route_from_rag)
workflow.add_edge("summarize_node", END)
workflow.add_edge("reason_node", END)

app_graph = workflow.compile()