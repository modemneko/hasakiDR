# hasakiDR Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [API Reference](#api-reference)
5. [Installation and Setup](#installation-and-setup)
6. [Usage Guide](#usage-guide)
7. [Technical Details](#technical-details)
8. [Future Enhancements](#future-enhancements)

## Project Overview

hasakiDR is an AI-based research assistant that can automatically collect, analyze, and synthesize information from multiple sources to generate comprehensive research reports. It combines LLM, AgentBrowser, and Graph node workflows.

### Key Features

- **Multi-engine search**: Leverages multiple search engines through SearXNG to gather diverse information
- **AI-powered analysis**: Uses large language models to analyze and synthesize information
- **Structured workflow**: Implements a state-based workflow for systematic research
- **Real-time progress tracking**: Provides updates on research progress through a web interface
- **Comprehensive reporting**: Generates detailed research reports with findings and insights
- **User-friendly interface**: Offers an intuitive web UI for submitting queries and viewing results

## Architecture

The hasakiDR system follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Web Interface                              │
│   (HTML/CSS/JavaScript with modern styling framework)              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                        API Layer                                   │
│          (API endpoints for plan/research management)              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                      Workflow Engine                               │
│               (State-based workflow system)                        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                        AI Layer                                    │
│         (Large language models for analysis and generation)        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                   Search & Browser Layer                           │
│   (Multi-engine search capabilities with web browsing support)     │
└─────────────────────────────────────────────────────────────────────┘
```

### Workflow Diagram

The research process follows a structured state-based workflow:

1. **Planner Node**: Generates a research plan based on the initial query
2. **Researcher Node**: Performs multi-engine searches and extracts relevant information
3. **Synthesizer Node**: Analyzes and synthesizes findings from different sources
4. **Final Report Node**: Generates a comprehensive research report

The workflow includes conditional logic that determines whether to continue research iterations or proceed to report generation based on the quality and completeness of the information gathered.

## Core Components

### 1. Web Interface (`static/index.html`)

The user-facing component built with HTML, CSS, and JavaScript, enhanced with Tailwind CSS for styling. It provides:

- A chat-like interface for submitting research queries
- Real-time display of research progress and intermediate results
- Plan generation and confirmation functionality
- Report viewing and copying capabilities
- Responsive design for different screen sizes

### 2. API Layer (`main.py`)

The FastAPI-based backend that handles HTTP requests and coordinates the research process:

- `/plan`: Generates a research plan for a given query
- `/research`: Initiates the research process with a plan
- `/research_progress`: Returns the status and results of ongoing research
- `/completed_reports`: Lists all completed research reports
- `/report/{query}`: Retrieves a specific completed report

### 3. Workflow Engine (`graph.py`)

Implements the state-based workflow using LangGraph:

- **State Management**: Maintains the research state throughout the process
- **Node Coordination**: Manages the execution of different workflow nodes
- **Conditional Logic**: Determines when to continue research or generate a report

### 4. AI Layer

#### Nodes (`nodes.py`)

- **Planner Node**: Creates research tasks based on the query and previous findings
- **Researcher Node**: Performs web searches and analyzes results
- **Synthesizer Node**: Integrates findings from multiple sources
- **Final Report Node**: Generates comprehensive research reports

#### Language Model Configuration (`config.py`)

Supports multiple AI models for different use cases and performance requirements

### 5. Search Layer (`nodes.py`)

- **searxng_search function**: Queries the SearXNG API to gather results from multiple search engines
- **Result Processing**: Extracts and formats search results for analysis
- **Agent Browser Integration**: Uses `browser_use` module for potential web browsing capabilities
  - Initialized `Browser` instance for web navigation
  - Designed to support web browsing agents for more complex research tasks
  - Currently using direct SearXNG API calls for search functionality

### 6. State Management (`research_state.py`)

- **ResearchState**: TypedDict structure for maintaining research context
- **State Transitions**: Defines how state changes between workflow nodes

## API Reference

### 1. POST /plan

Generates a research plan for a given query.

**Request Body**:
```json
{
  "query": "Your research topic"
}
```

**Response**:
```json
{
  "plan": "Generated research plan text"
}
```

### 2. POST /research

Initiates the research process with a plan.

**Request Body**:
```json
{
  "query": "Your research topic",
  "plan": "Research plan text",
  "max_iterations": 3
}
```

**Response**:
```json
{
  "report": "Research initiated, please check progress later."
}
```

### 3. GET /research_progress

Returns the status and results of ongoing research.

**Query Parameters**:
- `query`: The original research query

**Response**:
```json
{
  "progress": "Current progress status",
  "final_report": "Generated report (if complete)",
  "logs": [
    {
      "type": "node_type",
      "content": "Log content"
    }
  ],
  "is_complete": true/false,
  "error": "Error message (if any)",
  "elapsed_time": 10.5
}
```

### 4. GET /completed_reports

Lists all completed research reports.

**Response**:
```json
[
  {
    "query": "Research topic",
    "timestamp": "2024-01-01T12:00:00",
    "elapsed_time": 15.2,
    "has_error": false
  }
]
```

### 5. GET /report/{query}

Retrieves a specific completed report.

**Path Parameters**:
- `query`: The original research query

**Response**:
```json
{
  "query": "Research topic",
  "report": "Generated report content",
  "timestamp": "2024-01-01T12:00:00",
  "elapsed_time": 15.2,
  "error": "Error message (if any)"
}
```

## Installation and Setup

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Access to Ollama or Google Gemini API
- Access to a SearXNG instance

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd hasakiDR
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**:
   - For Google Gemini: Update `GOOGLE_API_KEY` in `config.py`
   - For Ollama: Ensure Ollama is running with the desired model

4. **Configure search engine**:
   - Update the `searxng_url` in `nodes.py` if needed

5. **Start the server**:
   ```bash
   python main.py
   ```

6. **Access the application**:
   Open a web browser and navigate to `http://localhost:11451`

## Usage Guide

### 1. Submitting a Research Query

1. Open the hasakiDR web interface
2. Enter your research topic in the input field
3. Adjust the maximum number of iterations if needed (default: 3)
4. Click "开始研究" (Start Research)

### 2. Reviewing the Research Plan

1. The system will generate a research plan
2. Review the plan and either:
   - Click "同意" (Agree) to proceed with the plan
   - Click "修改" (Edit) to customize the plan

### 3. Monitoring Research Progress

- The chat interface will display real-time updates from each workflow node
- The progress indicator will show the current status
- Wait for the research to complete

### 4. Viewing the Research Report

1. Once research is complete, click "点击查看报告" (Click to View Report)
2. The report will open in a side drawer
3. Use the "📋 复制" (Copy) button to copy the report
4. Close the drawer when finished

## Technical Details

### Workflow Execution

The research workflow is implemented using a state-based approach:

1. **Initialization**: The workflow is compiled once at startup
2. **State Management**: Context is maintained throughout the research process
3. **Node Execution**: Each processing node handles the current state and returns updates
4. **Conditional Transitions**: Logic determines whether to continue research or generate a report
5. **Stream Processing**: Results are streamed back to the client for real-time updates

### Search Implementation

- Uses a meta-search engine to gather results from multiple sources
- Implements error handling for robust search operations
- Processes and formats search results for AI analysis
- Includes web browsing capabilities for potential complex research tasks
  - Web navigation functionality for comprehensive data collection
  - Designed to support interactive browsing and data extraction
  - Currently configured for efficient search functionality

### AI Integration

- Integrates with various AI models for analysis and generation
- Supports both cloud-based and local model deployment options
- Implements asynchronous processing for improved performance

### Frontend Architecture

- Pure HTML/CSS/JavaScript implementation
- Tailwind CSS for responsive design
- Marked.js for Markdown rendering
- Real-time updates via polling

## Future Enhancements

1. **Advanced Search Capabilities**
   - Support for more specialized search engines
   - Custom search filters and parameters
   - Direct URL submission for specific sources
   - Full utilization of web browsing capabilities
     - Implement web navigation agents for complex research tasks
     - Support for interactive web browsing and data extraction
     - Enhanced capabilities for handling dynamic web content

2. **Enhanced AI Capabilities**
   - Multi-model comparison and ensemble approaches
   - Fine-tuned models for specific research domains
   - Improved reasoning and citation capabilities

3. **User Experience Improvements**
   - Real-time web socket updates instead of polling
   - Interactive report editing and customization
   - Export options for different formats (PDF, DOCX, etc.)

4. **Scalability and Performance**
   - Background task processing with Celery
   - Caching for common queries and search results
   - Parallel execution for multi-topic research

5. **Security Enhancements**
   - API key protection and rotation
   - Input validation and sanitization
   - Rate limiting and access control

6. **Additional Features**
   - Citation and reference management
   - Research history and favorites
   - Collaborative research capabilities

## Conclusion

hasakiDR represents a powerful approach to automated research, combining the strengths of web search, AI analysis, and structured workflows. Its modular architecture and extensible design make it well-suited for a wide range of research applications, from academic studies to market analysis and beyond.

By leveraging modern AI technologies and a user-centric design, hasakiDR streamlines the research process, enabling users to focus on analysis and decision-making rather than information gathering and synthesis.
