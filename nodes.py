import asyncio
from typing import Dict, Any
from langchain_core.messages import AIMessage
from config import LLM
from research_state import ResearchState
from browser_use import Agent, Browser, BrowserConfig
from browser_use.llm.messages import UserMessage
import requests

# 初始化 browser
browser = Browser()

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
研究主题: \"{initial_query}\"
请为联网浏览代理生成一个任务指令，要求如下：
1. 分别在 Google、Bing、DuckDuckGo、百度等主流搜索引擎上针对上述研究主题进行关键词搜索。
2. 从每个搜索引擎的结果中各自识别并选择1-2个最相关的网页链接。
3. 访问这些链接，并从每个页面中提取与研究主题直接相关的核心信息。
4. 总结并对比各搜索引擎收集到的信息，突出异同点。
直接输出这个任务指令。
"""
    else:
        prompt = f"""
总体研究主题: \"{initial_query}\"
{history_context_for_prompt}
请为联网浏览代理生成下一步的研究任务指令，要求如下：
1. 继续在 Google、Bing、DuckDuckGo、百度等主流搜索引擎上补充检索，聚焦于已有发现的空白点或新角度。
2. 从每个搜索引擎的新结果中各自识别并选择最相关的网页链接，访问并提取关键信息。
3. 总结并对比各搜索引擎新收集到的信息，突出与前述发现的异同。
如果分析认为当前信息已足够全面回答总体研究主题，请直接输出 \"生成最终报告\"。
否则，请直接输出下一步任务指令。
"""
    response = await LLM.ainvoke([UserMessage(content=prompt)])
    next_task = response.completion.strip() if hasattr(response, "completion") else str(response).strip()
    print(f"Planner generated task: {next_task}")

    if "生成最终报告" in next_task:
        if accumulated_findings and len(accumulated_findings) > 50:
            print("Planner suggests generating final report.")
            return {"current_task": "FINAL_REPORT_TASK", "current_iteration": current_iteration + 1}
        else:
            print("Planner suggested report, but findings are insufficient. Forcing further research.")
            fallback_task = f"继续深入研究 '{initial_query}' 的核心方面，寻找更具体的细节、例子或证据。"
            return {"current_task": fallback_task, "current_iteration": current_iteration + 1}

    return {
        "current_task": next_task,
        "current_iteration": current_iteration + 1
    }

def searxng_search(query, searxng_url="https://search.mdosch.de", num_results=10):
    params = {
        "q": query,
        "format": "json",
        "language": "zh-CN",
        "safesearch": 0,
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    }
    resp = requests.get(f"{searxng_url}/search", params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return [
        {
            "title": r.get("title"),
            "content": r.get("content"),
            "url": r.get("url")
        }
        for r in data.get("results", [])[:num_results]
    ]

async def researcher_node(state: ResearchState) -> Dict[str, Any]:
    print("\n--- Researcher ---")
    # 只用纯关键词搜索，避免 403
    search_query = state.get("initial_query", "")
    if not search_query or search_query == "FINAL_REPORT_TASK":
        return {"research_history": [AIMessage(content=f"跳过研究节点。任务: {search_query}")]}
    search_results = searxng_search(search_query)
    search_text = "\n\n".join([f"{item['title']}\n{item['content']}\n{item['url']}" for item in search_results])
    prompt = f"请根据以下多引擎搜索结果，提取与 '{search_query}' 相关的核心信息，并对比不同来源：\n{search_text}\n请用中文总结。"
    response = await LLM.ainvoke([UserMessage(content=prompt)])
    summary = response.completion.strip()
    return {"research_history": [AIMessage(content=f"研究任务: {search_query}\n研究结果:\n{summary}")]}

async def synthesizer_node(state: ResearchState) -> Dict[str, Any]:
    print("\n--- Synthesizer ---")
    initial_query = state["initial_query"]
    research_history_messages = state.get("research_history", [])
    research_results = [msg.content for msg in research_history_messages if isinstance(msg, AIMessage) and isinstance(msg.content, str) and "研究结果:" in msg.content]
    if not research_results:
        print("无新的研究结果可供合成。")
        return {"accumulated_findings": state.get("accumulated_findings", "无")}
    all_results_text = "\n\n".join(research_results)
    prompt = f"""整合以下关于 \"{initial_query}\" 的研究信息，生成更新的累积发现概要。如果已有一些累积发现，请在新的发现基础上更新它。
研究信息片段:
{all_results_text}
请输出简洁、连贯的累积发现概要。"""
    response = await LLM.ainvoke([UserMessage(content=prompt)])
    updated_findings = response.completion.strip() if hasattr(response, "completion") else str(response).strip()
    print(f"Synthesized findings length: {len(updated_findings)}")
    return {"accumulated_findings": updated_findings}

async def final_report_node(state: ResearchState) -> Dict[str, Any]:
    print("\n--- Final Report Generator ---")
    initial_query = state["initial_query"]
    accumulated_findings = state.get("accumulated_findings", "未能积累发现。")
    history_summary_for_report = ""
    if state.get("research_history") and len(state["research_history"]) > 2:
        history_summary_for_report = "\n\n详细研究步骤和发现概要：\n"
        step_counter = 1
        for msg in state["research_history"]:
            if isinstance(msg, AIMessage) and isinstance(msg.content, str) and "研究任务:" in msg.content:
                parts = msg.content.split("\n研究结果:\n", 1)
                task_summary = parts[0].replace("研究任务:", "").strip()
                result_preview = parts[1][:300] + "..." if len(parts) > 1 and parts[1] else "无明确结果。"
                history_summary_for_report += f"\n步骤 {step_counter}:\n 任务: {task_summary}\n 结果预览: {result_preview}\n"
                step_counter += 1
        if step_counter == 1:
            history_summary_for_report = ""
    prompt = f"""研究主题: \"{initial_query}\"
核心累积发现:
{accumulated_findings}
{history_summary_for_report if history_summary_for_report else "无详细研究步骤回顾。"}
请基于以上信息，撰写一份全面、结构清晰的研究报告."""
    response = await LLM.ainvoke([UserMessage(content=prompt)])
    report = response.completion.strip() if hasattr(response, "completion") else str(response).strip()
    print(f"Final Report generated, length: {len(report)}")
    return {"final_report": report}
