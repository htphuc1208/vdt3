---
license: mit
configs:
  - config_name: TS1
    data_files:
      - split: test
        path: TS1/test.json
  - config_name: TS2
    data_files:
      - split: test
        path: TS2/test.json
  - config_name: TS3
    data_files:
      - split: test
        path: TS3/test.json
language:
- en
tags:
- benchmark
- tool-use
- telecommunications
pretty_name: TeleLogsAgent
task_categories:
- question-answering
size_categories:
- n<2K
---

<div align="center" style="font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; padding: 25px 15px; max-width: 720px; margin: 20px auto;">
  <div style="font-size: 3.2em; font-weight: bold; margin-bottom: 5px;">
    <span style="background: -webkit-linear-gradient(45deg, #5a03bdff, #9f10f2ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
      TeleLogsAgent
    </span>
  </div>
  <div style="font-size: 1.0em; color: #4a4a4a; margin-bottom: 12px; line-height: 1.45; padding: 0 10px;">
    A Benchmark for LLM Tool-Use in 5G Network Root Cause Analysis
  </div>
  <div style="font-size: 0.80em; color: #777; margin-bottom: 10px;">
    Developed by the <strong>NetOp Team, Huawei Paris Research Center</strong>
  </div>
  <hr style="border: 0; height: 1px; background: #ddd; margin-top: 10px; margin-bottom: 15px; width: 60%;">
  <div style="font-size: 0.80em; color: #6c757d; line-height: 1.5; margin-bottom: 15px;">
    Mohamed Sana · Nicola Piovesan · Antonio De Domenico · Fadhel Ayed
  </div>
  <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; font-size: 1.1em; margin-bottom: 0px;">
  <a href="https://arxiv.org/abs/2506.10674" target="_blank"
     style="text-decoration: none; background-color: #007bff; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; text-align: center;">
    📄 Read the Paper
  </a>
  <a href="https://huggingface.co/datasets/netop/TeleLogsAgent"
     style="text-decoration: none; background-color: #ffc107; color: black; padding: 10px 20px; border-radius: 5px; font-weight: bold; text-align: center;">
    🤗 Explore the Dataset
  </a>
  </div>
</div>


> [!NOTE]
> IMPORTANT: Please help us protect the integrity of this benchmark by not publicly sharing, re-uploading, or distributing the dataset.



## Dataset Description

- **Repository (Dataset & Evaluation Code):** https://huggingface.co/datasets/netop/TeleLogsAgent  
- **Paper:** https://arxiv.org/abs/2506.10674  

TeleLogsAgent is a benchmark and evaluation framework designed to measure the ability of Large Language Model (LLM) agents to perform **structured tool-use** in the telecommunications domain.

It simulates the workflow of a 5G network engineer diagnosing performance degradation during drive testing, requiring agents to:
- inspect configuration data,
- analyze time-series KPIs,
- reason across multiple tools,
- identify the most plausible root cause.



## Overview

The benchmark consists of **two main components**:

1. **FastAPI Server (`fastapi_server.py`)**  
   Exposes realistic analytical tools (HTTP endpoints) to access 5G drive-test scenarios.  
   Agents interact with this environment using OpenAI-style function calls.

2. **LLM Evaluation Agent (`benchmark.py`)**  
   Connects to either the FastAPI server and evaluates LLMs on their ability to plan, call tools, and reason over multiple steps.

In addition, we conveniently provide a **FastMCP Server (`fastmcp_server.py`)** as an alternative implementation of the FastAPI server using **FastMCP**.  This version is especially convenient for MCP-native LLM agents.

## Project Structure

```text
TeleLogsAgent/
├── fastapi_server.py      # FastAPI benchmark server (HTTP tools)
├── fastmcp_server.py      # FastMCP benchmark server (MCP tools)
├── benchmark.py           # LLM evaluation / benchmarking script
├── TS1/test.json          # Scenario 1: root cause identification based on high-level network configuration and user-plane data.
├── TS2/test.json          # Scenario 2: root cause identification based on high-level and low-level network configuration, signaling-plane and user-plane data.
├── TS3/test.json          # Scenario 3: root cause remediation based on high-level and low-level network configuration, signaling-plane and user-plane data.
├── requirements.txt       # Dependencies
├── README.md              # This file
````

Main dependencies include:

* fastapi
* uvicorn
* fastmcp
* pandas
* requests
* openai
* numpy
* tqdm



## Running the Benchmark Environment

### Option A — FastAPI Server (HTTP Tools)

```bash
export TELELOGS_AGENT_CONFIG="TS1"; python fastapi_server.py
```

Server address:

```
http://localhost:7861
```

Scenario context is managed using the HTTP header:

```
X-Scenario-Id: <scenario_id>
```

Available endpoints include:

* `/scenario`
* `/signaling-plane-event-log` (only available in scenario TS1 & TS2)
* `/throughput-logs`
* `/cell-info`
* `/gnodeb-location`
* `/user-location`
* `/user-speed`
* `/serving-cell-pci`
* `/serving-cell-rsrp`
* `/serving-cell-sinr`
* `/rbs-allocated-to-user`
* `/neighboring-cells-pci`
* `/neighboring-cell-rsrp`
* `/beam-scenario-info`
* `/tools`



### Option B — FastMCP Server

```bash
python fastmcp_server.py
```

MCP endpoint:

```
http://localhost:7860
```

**Advantages of FastMCP**

* Native MCP protocol
* Session-scoped scenario context
* Cleaner agent logic
* Seamless integration with MCP-compatible agents

The FastMCP server exposes the **same logical tools** as the FastAPI server.



## Running the Agent Evaluation

The evaluation script supports only the FastAPI backend. Adapting to FastMCP is however straighforward.

### Using FastAPI Tools

```bash
export TELELOGS_AGENT_API_KEY=xxxx
python benchmark.py \
  --server_url http://localhost:7860 \
  --model_url http://localhost:7865/v1 \
  --model_name qwen8B \
  --num_attempts 4 \
  --max_samples 20 \
  --save_dir ./results
```

## Evaluation and Scoring

Agents are evaluated along multiple dimensions:

1. **Task Success** – Correct root cause identification
2. **Tool Call Efficiency** –  Average accuracy per number of tool calls
3. **Tool Call Failure Rate**
4. **Average number of iterations per task** 



## Citation

If you use TeleLogsAgent in your research, please cite:

```bibtex
@article{Sana2026TeleLogsAgent,
  title={{TeleLogsAgent: A Benchmark for LLM Tool-Use in 5G Network Root Cause Analysis}},
  author={Mohamed Sana and Nicola Piovesan and Antonio De Domenico and Fadhel Ayed},
  year={2026},
  eprint={arXiv:2506.10674},
  url={https://arxiv.org/abs/2506.10674}
}
```