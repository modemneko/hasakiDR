import os
import asyncio
from datetime import datetime
from typing import TypedDict, List, Annotated, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai import types as genai_types
from browser_use import Agent, Browser, BrowserConfig 

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import time


# proxy = "http://127.0.0.1:7890"
# os.environ["HTTP_PROXY"] = proxy


browser = Browser(
    BrowserConfig(
        headless=True,
        disable_security=True
    )
)


GOOGLE_API_KEY = ""
os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY

class ResearchState(TypedDict):
    initial_query: str
    current_task: str
    research_history: Annotated[List[BaseMessage], add_messages]
    accumulated_findings: str
    final_report: str
    max_iterations: int
    current_iteration: int


llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        safety_settings={
            genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai_types.HarmBlockThreshold.BLOCK_NONE,
            genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai_types.HarmBlockThreshold.BLOCK_NONE,
            genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai_types.HarmBlockThreshold.BLOCK_NONE,
            genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai_types.HarmBlockThreshold.BLOCK_NONE,
        },
        temperature=0.5,
    )
    
async def planner_node(state: ResearchState) -> Dict[str, Any]:
    print("\n--- Planner ---")
    current_iteration = state.get("current_iteration", 0)
    initial_query = state["initial_query"]
    accumulated_findings = state.get("accumulated_findings", "无初始发现")
    research_history_messages = state.get("research_history", [])

    history_context_for_prompt = ""
    if current_iteration > 0:
        history_context_for_prompt = f"当前已积累的研究发现概要：\n{accumulated_findings}\n"

        if research_history_messages:
            last_ai_message_content = None
            for msg in reversed(research_history_messages):
                if isinstance(msg, AIMessage) and msg.content:
                     last_ai_message_content = msg.content
                     break

            if last_ai_message_content:
                 history_context_for_prompt += f"\n最近完成的研究步骤回顾：\n{last_ai_message_content[:500]}...\n"


    if current_iteration == 0:
        prompt = f"""
        研究主题: "{initial_query}"

        请为网页浏览代理 BrowserAgent 生成一个任务指令。
        指令应引导其完成以下步骤：
        1. 针对上述研究主题进行关键词搜索。
        2. 从搜索结果中识别并选择1-2个最相关的网页链接。
        3. 访问这些链接，并从每个页面中提取与研究主题直接相关的核心信息。
        4. 总结收集到的所有信息。

        直接输出这个任务指令。
        """
    else:
        prompt = f"""
        总体研究主题: "{initial_query}"

        {history_context_for_prompt}
        请为网页浏览代理 BrowserAgent 生成下一步的研究任务指令。
        该指令应旨在深化已有发现、填补信息空白或探索新角度。
        指令应引导其完成类似以下步骤：
        1. 根据需要补充或深化的具体信息点，进行有针对性的搜索或访问已知网站。
        2. 从结果中选择最合适的链接访问（如果进行了搜索）。
        3. 从目标页面提取与当前特定研究焦点相关的详细信息。
        4. 总结新收集到的信息。

        如果分析认为当前信息已足够全面回答总体研究主题，请直接输出 "生成最终报告"。
        否则，请直接输出给 BrowserAgent 的下一步任务指令。
        """

    response = await llm.ainvoke(prompt)
    next_task_for_browser_use = response.content.strip()
    print(f"Planner generated task for Agent: {next_task_for_browser_use}")

    if "生成最终报告" in next_task_for_browser_use:
        if accumulated_findings and len(accumulated_findings) > 50:
            print("Planner suggests generating final report.")
            return {"current_task": "FINAL_REPORT_TASK", "current_iteration": current_iteration + 1}
        else:
            print("Planner suggested report, but findings are insufficient. Forcing further research.")
            fallback_task = f"继续深入研究 '{initial_query}' 的核心方面，寻找更具体的细节、例子或证据。"
            return {"current_task": fallback_task, "current_iteration": current_iteration + 1}


    return {
        "current_task": next_task_for_browser_use,
        "current_iteration": current_iteration + 1
    }


async def researcher_node(state: ResearchState) -> Dict[str, Any]:
    print("\n--- Researcher ---")
    task_to_research = state.get("current_task", "")
    if not task_to_research or task_to_research in ["FINAL_REPORT_TASK", "ERROR_LLM_UNAVAILABLE"]:
        print("无新研究任务、准备生成报告或LLM错误，跳过网页搜索。")
        return {"research_history": [AIMessage(content=f"跳过研究节点。任务: {task_to_research}")]}

    if browser is None:
         print("错误: 浏览器未初始化，无法执行 Researcher 节点。")
         return {"research_history": [AIMessage(content=f"研究任务 '{task_to_research}' 执行失败: 浏览器未初始化。")]}

    if llm is None:
        print("错误: LLM 未初始化，无法执行 Researcher 节点。")
        return {"research_history": [AIMessage(content=f"研究任务 '{task_to_research}' 执行失败: LLM 未初始化。")]}

    print(f"Researching: {task_to_research}")
    try:
        agent = Agent(
            browser=browser,
            task=task_to_research,
            llm=llm,
            use_vision=False,
            max_failures=10,
            retry_delay=5,
            )
        result_text = await agent.run()

        summary = str(result_text) if result_text else "未能获取明确信息。"
        message_content = f"研究任务: {task_to_research}\n研究结果:\n{summary[:5000]}"
        return {"research_history": [AIMessage(content=message_content)]}
    except Exception as e:
        print(f"Error in researcher_node for task '{task_to_research}': {e}")
        return {"research_history": [AIMessage(content=f"研究任务 '{task_to_research}' 执行失败: {e}")]}


async def synthesizer_node(state: ResearchState) -> Dict[str, Any]:
    print("\n--- Synthesizer ---")
    initial_query = state["initial_query"]
    research_history_messages = state.get("research_history", [])
    # 从历史消息中过滤出 AI 的研究结果
    research_results = [msg.content for msg in research_history_messages if isinstance(msg, AIMessage) and "研究结果:" in msg.content]

    if not research_results:
         print("无新的研究结果可供合成。")
         return {"accumulated_findings": state.get("accumulated_findings", "无")}

    all_results_text = "\n\n".join(research_results)

    if llm is None:
         print("错误: LLM 未初始化，无法执行 Synthesizer 节点。")
         # 返回当前积累的发现，不进行更新
         return {"accumulated_findings": state.get("accumulated_findings", "LLM不可用，未能合成新发现。")}

    prompt = f"""整合以下关于 "{initial_query}" 的研究信息，生成更新的累积发现概要。如果已有一些累积发现，请在新的发现基础上更新它。

    研究信息片段:
    {all_results_text}

    请输出简洁、连贯的累积发现概要。"""
    response = await llm.ainvoke(prompt)
    updated_findings = response.content.strip()
    print(f"Synthesized findings length: {len(updated_findings)}")
    return {"accumulated_findings": updated_findings}


async def final_report_node(state: ResearchState) -> Dict[str, Any]:
    print("\n--- Final Report Generator ---")
    initial_query = state["initial_query"]
    accumulated_findings = state.get("accumulated_findings", "未能积累发现。")

    if llm is None:
         print("错误: LLM 未初始化，无法生成最终报告。")
         return {"final_report": "错误: LLM 未初始化，未能生成最终报告。\n" + accumulated_findings}


    history_summary_for_report = ""
    # 仅在有足够历史时包含详细步骤
    if state.get("research_history") and len(state["research_history"]) > 2:
        history_summary_for_report = "\n\n详细研究步骤和发现概要：\n"
        step_counter = 1
        for msg in state["research_history"]:
             if isinstance(msg, AIMessage) and msg.content and "研究任务:" in msg.content:
                  # 提取任务和结果的概览
                  parts = msg.content.split("\n研究结果:\n", 1)
                  task_summary = parts[0].replace("研究任务:", "").strip()
                  result_preview = parts[1][:300] + "..." if len(parts) > 1 and parts[1] else "无明确结果。"
                  history_summary_for_report += f"\n步骤 {step_counter}:\n 任务: {task_summary}\n 结果预览: {result_preview}\n"
                  step_counter += 1
        if step_counter == 1: # 如果没有提取到有效的研究步骤
            history_summary_for_report = ""


    prompt = f"""研究主题: "{initial_query}"

    核心累积发现:
    {accumulated_findings}
    {history_summary_for_report if history_summary_for_report else "无详细研究步骤回顾。"}

    请基于以上信息，撰写一份全面、结构清晰的研究报告。"""
    response = await llm.ainvoke(prompt)
    report = response.content.strip()
    print(f"Final Report generated, length: {len(report)}")
    return {"final_report": report}

# --- 4. 定义条件边 ---

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

app_graph = workflow.compile()

# Dictionary to store ongoing and completed research progress
research_progress = {}
# Dictionary to store completed reports for retrieval
completed_reports = {}

async def run_hasaki_research(initial_query: str, max_iterations: int = 3):
    global research_progress, completed_reports
    # Use a unique key for the query, maybe sanitize it or use a hash if queries can be very long/complex
    query_key = initial_query # Simple key for now

    # Initialize progress for the new query
    research_progress[query_key] = {
        "progress": "Starting...",
        "final_report": None,
        "logs": [], # Initialize logs list
        "token_usage": None, # Placeholder, actual token usage tracking would require more integration
        "synthesizer_info": None, # Placeholder for synthesizer summary
        "final_report_info": None, # Placeholder for final report summary
        "start_time": time.time(),
        "elapsed_time": 0,
        "is_complete": False, # Add a completion flag
        "error": None, # Add an error field
    }

    inputs = {
        "initial_query": initial_query,
        "max_iterations": max_iterations,
        "current_iteration": 0,
        "research_history": [],
        "accumulated_findings": "无初始发现。",
        "final_report": "",
    }
    print(f"\n🚀 Starting hasaki research for: '{initial_query}' (max {max_iterations} iterations)")

    try:
        async for output in app_graph.astream(inputs):
            # Capture and log all intermediate outputs
            for key, value in output.items():
                if key != '__end__':
                    # Format log entry to include node name and a snippet of the value
                    log_entry_content = str(value)
                    if len(log_entry_content) > 500:
                         log_entry_content = log_entry_content[:500] + "..."
                    log_entry = f"[{key}] {log_entry_content}"
                    research_progress[query_key]["logs"].append(log_entry)
                    print(f"Stream output: {log_entry}") # Optional: print to console

            # Update progress based on the latest state
            # Get the latest state from the last output chunk
            last_output_key = list(output.keys())[-1]
            current_state = output.get('__end__') or output.get(last_output_key)

            if current_state:
                 if current_state.get('final_report'):
                    research_progress[query_key]["progress"] = "研究完成，生成报告。"
                    research_progress[query_key]["final_report"] = current_state['final_report']
                    research_progress[query_key]["final_report_info"] = f"Final Report generated, length: {len(current_state['final_report'])}"
                    research_progress[query_key]["is_complete"] = True # Mark as complete
                    # Store the completed report
                    completed_reports[query_key] = {
                        "query": initial_query,
                        "report": current_state['final_report'],
                        "timestamp": datetime.now().isoformat(),
                        "elapsed_time": time.time() - research_progress[query_key]["start_time"],
                    }
                 elif current_state.get('accumulated_findings'):
                    # Update progress with a summary of findings
                    research_progress[query_key]["progress"] = "研究进行中... 累积发现概要: " + (current_state['accumulated_findings'][:200] + "..." if current_state['accumulated_findings'] else "无")
                    research_progress[query_key]["synthesizer_info"] = f"Synthesized findings length: {len(current_state['accumulated_findings'])}"
                 elif current_state.get('current_task'):
                     task_preview = str(current_state['current_task'])
                     if len(task_preview) > 100:
                         task_preview = task_preview[:100] + "..."
                     research_progress[query_key]["progress"] = f"研究进行中... 当前任务: {task_preview}"
                 else:
                    research_progress[query_key]["progress"] = "研究进行中..." # Default progress

            research_progress[query_key]["elapsed_time"] = time.time() - research_progress[query_key]["start_time"]
            # No need for asyncio.sleep(1) here, astream yields as it progresses

        # After the astream loop finishes, the final state should be available
        # The last output from astream should contain the final state if it reached END
        # A final check to ensure completion status and report are captured
        if not research_progress[query_key]["is_complete"]:
             # If astream finished but didn't mark as complete, it might have ended without reaching the final node
             print("\n⚠️ hasaki research finished stream but did not reach the final report node.")
             research_progress[query_key]["progress"] = "研究完成，但未能生成报告。"
             research_progress[query_key]["is_complete"] = True # Mark as complete even if no report
             research_progress[query_key]["elapsed_time"] = time.time() - research_progress[query_key]["start_time"]


        # Return the final report or a status message
        return research_progress[query_key].get('final_report') if research_progress[query_key].get('final_report') else "研究未能生成报告。"


    except Exception as e:
        print(f"\n❌ An error occurred during graph execution: {e}")
        research_progress[query_key]["progress"] = f"研究执行期间发生错误: {e}"
        research_progress[query_key]["elapsed_time"] = time.time() - research_progress[query_key]["start_time"]
        research_progress[query_key]["is_complete"] = True # Mark as complete on error
        research_progress[query_key]["error"] = str(e)
        # Store error state in completed reports as well
        completed_reports[query_key] = {
            "query": initial_query,
            "report": f"研究执行期间发生错误: {e}",
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": time.time() - research_progress[query_key]["start_time"],
            "error": str(e),
        }
        return f"研究执行期间发生错误: {e}"


# --- 7. 构建 FastAPI Web UI ---
app = FastAPI()

# Mount static files (like index.html, CSS, JS if separate)
app.mount("/static", StaticFiles(directory="e:/HakusAI"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Serve index.html from the static directory
    return FileResponse("e:/HakusAI/index.html")

@app.post("/research")
async def start_research(request: Request):
    global research_progress
    try:
        data = await request.json()
        query = data.get("query")
        max_iterations = data.get("max_iterations", 3)
        deep_research = data.get("deep_research", False)

        if not query:
            return {"report": "错误：未提供研究主题。"}

        print(f"Web UI Received Query: {query}, Max Iterations: {max_iterations}, Deep Research: {deep_research}")

        # Use the query itself as the key for progress tracking
        query_key = query

        # Prevent starting research if one is already running for this query
        if query_key in research_progress and not research_progress[query_key]["is_complete"] and research_progress[query_key]["error"] is None:
             return {"report": "该研究主题已在进行中。"}

        if deep_research:
            # Run hasaki research in a background task
            asyncio.create_task(run_hasaki_research(query, max_iterations=max_iterations))
            return {"report": "hasaki研究已启动，请稍后查看研究进度。"}
        else:
            # Run regular research (single pass or limited iterations)
            # For simplicity, let's make non-deep research just run the graph once or with minimal iterations
            # Or, we can make the graph itself handle the non-deep case based on max_iterations=1
            # Let's reuse run_hasaki_research but with max_iterations=1 for non-deep
            final_report = await run_hasaki_research(query, max_iterations=1) # Use max_iterations=1 for non-deep
            # For non-deep, we can return the report directly
            report_data = completed_reports.get(query_key, {})
            return {"report": report_data.get("report", final_report), "error": report_data.get("error")}


    except Exception as e:
        print(f"Error in /research endpoint: {e}")
        return {"report": f"处理请求时发生错误: {e}"}

@app.get("/research_progress")
async def get_research_progress(query: str):
    global research_progress
    # Return a copy to avoid external modification issues
    # Use the query itself as the key
    query_key = query
    progress_data = research_progress.get(query_key, {"report": "研究尚未启动或未找到研究进度。", "final_report": None, "is_complete": True, "error": None, "logs": [], "elapsed_time": 0})
    return progress_data

@app.get("/completed_reports")
async def list_completed_reports():
    global completed_reports
    # Return a list of completed report summaries (query and timestamp)
    report_list = []
    for query, data in completed_reports.items():
        report_list.append({
            "query": data["query"],
            "timestamp": data["timestamp"],
            "elapsed_time": data["elapsed_time"],
            "has_error": "error" in data,
        })
    # Sort by timestamp, newest first
    report_list.sort(key=lambda x: x["timestamp"], reverse=True)
    return report_list

@app.get("/report/{query:path}") # Use path converter to handle queries with slashes
async def get_report(query: str):
    global completed_reports
    # Use the query itself as the key
    query_key = query
    report_data = completed_reports.get(query_key)
    if report_data:
        return {"query": report_data["query"], "report": report_data["report"], "timestamp": report_data["timestamp"], "elapsed_time": report_data["elapsed_time"], "error": report_data.get("error")}
    else:
        return {"report": "未找到该研究报告。", "error": "Report not found"}


if __name__ == "__main__":
    # Ensure the static directory exists (where index.html is)
    if not os.path.exists("e:/HakusAI"):
        print("Error: e:/HakusAI directory not found.")
        exit()

    uvicorn.run(app, host="127.0.0.1", port=7860)