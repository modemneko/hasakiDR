from langgraph.graph import StateGraph, END
from research_state import ResearchState
from nodes import planner_node, researcher_node, synthesizer_node, final_report_node

# --- 定义条件边 ---
def should_continue(state: ResearchState) -> str:
    current_iteration = state.get("current_iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    current_task = state.get("current_task", "")

    if current_task == "FINAL_REPORT_TASK":
        print("条件判断: 任务为生成报告，流程转向 final_report_generator")
        return "generate_report"
    if current_task == "ERROR_LLM_UNAVAILABLE":
        print("条件判断: LLM 不可用，流程转向 final_report_generator (生成错误报告)")
        return "generate_report"
    if current_iteration >= max_iterations:
        print(f"条件判断: 达到最大迭代次数 ({max_iterations})，流程转向 final_report_generator")
        return "generate_report"

    print("条件判断: 继续研究，流程转向 planner")
    return "continue_research"

# --- 构建工作流 ---
def build_research_graph():
    workflow = StateGraph(ResearchState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("final_report_generator", final_report_node)

    workflow.set_entry_point("planner")

    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "synthesizer")

    workflow.add_conditional_edges(
        "synthesizer",
        should_continue,
        {"continue_research": "planner", "generate_report": "final_report_generator"}
    )
    workflow.add_edge("final_report_generator", END)

    return workflow.compile()
