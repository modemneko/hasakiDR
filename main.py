import os
import asyncio
import time
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from graph import build_research_graph
from research_state import ResearchState
from nodes import planner_node

# Compile the graph once at startup
app_graph = build_research_graph()

# Dictionary to store ongoing and completed research progress
research_progress = {}
# Dictionary to store completed reports for retrieval
completed_reports = {}

# --- FastAPI Web UI ---
app = FastAPI()

# 新增 /plan 接口
@app.post("/plan")
async def generate_plan(request: Request):
    data = await request.json()
    query = data.get("query")
    if not query:
        return {"plan": "错误：未提供研究主题。"}
    # 只调用 planner_node，生成 plan
    state = {
        "initial_query": query,
        "current_task": "",
        "max_iterations": 1,
        "current_iteration": 0,
        "research_history": [],
        "accumulated_findings": "无初始发现。",
        "final_report": "",
    }
    plan_result = await planner_node(state)
    plan_text = plan_result.get("current_task", "未能生成研究计划。")
    return {"plan": plan_text}

# /research 接口，接收 plan，分步产出进度
@app.post("/research")
async def start_research(request: Request):
    global research_progress
    data = await request.json()
    query = data.get("query")
    plan = data.get("plan")
    max_iterations = data.get("max_iterations", 3)
    if not query or not plan:
        return {"report": "错误：未提供研究主题或计划。"}
    query_key = query
    # 初始化进度
    research_progress[query_key] = {
        "progress": "研究已启动...",
        "final_report": None,
        "logs": [],
        "token_usage": None,
        "synthesizer_info": None,
        "final_report_info": None,
        "start_time": time.time(),
        "elapsed_time": 0,
        "is_complete": False,
        "error": None,
    }
    # 研究主流程
    async def run():
        inputs: ResearchState = {
            "initial_query": query,
            "current_task": plan,
            "max_iterations": max_iterations,
            "current_iteration": 0,
            "research_history": [],
            "accumulated_findings": "无初始发现。",
            "final_report": "",
        }
        try:
            async for output in app_graph.astream(inputs):
                for key, value in output.items():
                    if key != '__end__':
                        log_entry_content = str(value)
                        if len(log_entry_content) > 500:
                            log_entry_content = log_entry_content[:500] + "..."
                        # 结构化日志，带 step type
                        log_entry = {"type": key, "content": log_entry_content}
                        research_progress[query_key]["logs"].append(log_entry)
                        print(f"Stream output: {log_entry}")
                last_output_key = list(output.keys())[-1]
                current_state = output.get('__end__') or output.get(last_output_key)
                if current_state:
                    if current_state.get('final_report'):
                        research_progress[query_key]["progress"] = "研究完成，生成报告。"
                        research_progress[query_key]["final_report"] = current_state['final_report']
                        research_progress[query_key]["final_report_info"] = f"Final Report generated, length: {len(current_state['final_report'])}"
                        research_progress[query_key]["is_complete"] = True
                        completed_reports[query_key] = {
                            "query": query,
                            "report": current_state['final_report'],
                            "timestamp": datetime.now().isoformat(),
                            "elapsed_time": time.time() - research_progress[query_key]["start_time"],
                        }
                    elif current_state.get('accumulated_findings'):
                        research_progress[query_key]["progress"] = "研究进行中... 累积发现概要: " + (current_state['accumulated_findings'][:200]) + "..." if current_state['accumulated_findings'] else "无"
                        research_progress[query_key]["synthesizer_info"] = f"Synthesized findings length: {len(current_state['accumulated_findings'])}"
                    elif current_state.get('current_task'):
                        task_preview = str(current_state['current_task'])
                        if len(task_preview) > 100:
                            task_preview = task_preview[:100] + "..."
                        research_progress[query_key]["progress"] = f"研究进行中... 当前任务: {task_preview}"
                    else:
                        research_progress[query_key]["progress"] = "研究进行中..."
                research_progress[query_key]["elapsed_time"] = time.time() - research_progress[query_key]["start_time"]
            if not research_progress[query_key]["is_complete"]:
                research_progress[query_key]["progress"] = "研究完成，但未能生成报告。"
                research_progress[query_key]["is_complete"] = True
                research_progress[query_key]["elapsed_time"] = time.time() - research_progress[query_key]["start_time"]
        except Exception as e:
            import traceback
            print(f"\n❌ An error occurred during graph execution: {e}")
            print(traceback.format_exc())
            research_progress[query_key]["progress"] = f"研究执行期间发生错误: {e}"
            research_progress[query_key]["elapsed_time"] = time.time() - research_progress[query_key]["start_time"]
            research_progress[query_key]["is_complete"] = True
            research_progress[query_key]["error"] = str(e)
            completed_reports[query_key] = {
                "query": query,
                "report": f"研究执行期间发生错误: {e}",
                "timestamp": datetime.now().isoformat(),
                "elapsed_time": time.time() - research_progress[query_key]["start_time"],
                "error": str(e),
            }
    asyncio.create_task(run())
    return {"report": "研究已启动，请稍后查看研究进度。"}

@app.get("/research_progress")
async def get_research_progress(query: str):
    global research_progress
    query_key = query
    progress_data = research_progress.get(query_key, {
        "report": "研究尚未启动或未找到研究进度。",
        "final_report": None,
        "is_complete": True,
        "error": None,
        "logs": [],
        "elapsed_time": 0
    })
    return JSONResponse(content=progress_data)

@app.get("/completed_reports")
async def list_completed_reports():
    global completed_reports
    report_list = []
    for query, data in completed_reports.items():
        report_list.append({
            "query": data["query"],
            "timestamp": data["timestamp"],
            "elapsed_time": data["elapsed_time"],
            "has_error": "error" in data,
        })
    report_list.sort(key=lambda x: x["timestamp"], reverse=True)
    return report_list

@app.get("/report/{query:path}")
async def get_report(query: str):
    global completed_reports
    query_key = query
    report_data = completed_reports.get(query_key)
    if report_data:
        return {"query": report_data["query"], "report": report_data["report"], "timestamp": report_data["timestamp"], "elapsed_time": report_data["elapsed_time"], "error": report_data.get("error")}
    else:
        return {"report": "未找到该研究报告。", "error": "Report not found"}

# Mount static files (like index.html, CSS, JS if separate)
# IMPORTANT: Ensure 'static' directory exists and contains your index.html and other static assets.
STATIC_FILES_DIR = "static"
app.mount("/static", StaticFiles(directory=STATIC_FILES_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse(os.path.join(STATIC_FILES_DIR, "index.html"))

if __name__ == "__main__":
    if not os.path.exists("static"):
        print("Error: static directory not found.")
        exit()
    uvicorn.run(app, host="127.0.0.1", port=11451)
