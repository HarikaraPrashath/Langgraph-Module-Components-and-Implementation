# LangGraph Module — Components and Implementation

A hands-on, progressive tutorial series that teaches the core building blocks of **LangGraph** — a framework for building stateful, multi-actor applications powered by large language models (LLMs). Each notebook builds on the previous one, taking you from the very first imports all the way to a fully functioning conversational agent with conditional branching and loop-back logic.

---

## Table of Contents

- [Overview](#overview)
- [What Is LangGraph?](#what-is-langgraph)
- [Core Concepts Covered](#core-concepts-covered)
- [Repository Structure](#repository-structure)
- [Notebook Walkthroughs](#notebook-walkthroughs)
  - [03-02 — Importing Relevant Classes](#03-02--importing-relevant-classes)
  - [03-03 — Defining State and Nodes](#03-03--defining-state-and-nodes)
  - [03-04 — Building the First Graph](#03-04--building-the-first-graph)
  - [03-05 — Conditional Edges (Setup)](#03-05--conditional-edges-setup)
  - [03-06 — Conditional Edges (Complete)](#03-06--conditional-edges-complete)
- [Architecture Deep Dive](#architecture-deep-dive)
  - [State](#state)
  - [Nodes](#nodes)
  - [Edges](#edges)
  - [Conditional Edges and Routing](#conditional-edges-and-routing)
  - [Graph Compilation](#graph-compilation)
- [Graph Visualizations](#graph-visualizations)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
- [Environment Variables](#environment-variables)
- [Running the Notebooks](#running-the-notebooks)
- [Key Patterns and Best Practices](#key-patterns-and-best-practices)
- [Further Reading](#further-reading)

---

## Overview

This module is part of a structured curriculum on LangGraph. It introduces the framework's fundamental primitives through a series of progressively more complex examples:

| Notebook | Topic | Complexity |
|---|---|---|
| `03-02` | Imports and environment setup | ⭐ |
| `03-03` | State schema and node functions | ⭐⭐ |
| `03-04` | Assembling and compiling a graph | ⭐⭐ |
| `03-05` | Conditional edge routing (partial) | ⭐⭐⭐ |
| `03-06` | Complete conditional looping graph | ⭐⭐⭐⭐ |

By the end of this series you will be able to design, build, compile, and run LangGraph-powered agents that can call LLMs, branch on state, and loop back to earlier nodes.

---

## What Is LangGraph?

[LangGraph](https://github.com/langchain-ai/langgraph) is an open-source library built on top of the [LangChain](https://github.com/langchain-ai/langchain) ecosystem. It models your application as a **directed graph** where:

- **Nodes** are the units of work (LLM calls, tool invocations, user I/O, business logic…).
- **Edges** define the order in which nodes execute.
- **State** is a typed dictionary that flows through the graph, carrying all information that nodes read from and write to.
- **Conditional edges** let the graph branch or loop depending on the current state — unlocking multi-step reasoning, human-in-the-loop flows, and cyclic agent behaviour.

Unlike simple LangChain chains, LangGraph graphs can contain **cycles**, which is essential for agentic workflows where an LLM decides whether to keep working or stop.

---

## Core Concepts Covered

| Concept | Description | First Seen |
|---|---|---|
| `TypedDict` state | Strongly-typed state schema passed between nodes | `03-03` |
| Node functions | Python callables that receive state and return a (partial) state update | `03-03` |
| `StateGraph` | The graph builder object | `03-04` |
| `START` / `END` | Special sentinel nodes marking the entry and exit of the graph | `03-04` |
| `add_node` | Registers a node function under a name | `03-04` |
| `add_edge` | Creates an unconditional edge between two nodes | `03-04` |
| `compile()` | Converts the graph definition into an executable `Runnable` | `03-04` |
| Routing functions | Python functions that inspect state and return the next node name | `03-05` |
| `add_conditional_edges` | Registers a routing function so the graph can branch dynamically | `03-05` / `03-06` |
| Looping graphs | Graphs that cycle back to earlier nodes based on user or model output | `03-06` |
| ASCII graph drawing | `get_graph().draw_ascii()` for quick visual debugging | `03-06` |

---

## Repository Structure

```
Langgraph-Module-Components-and-Implementation/
│
├── README.md                       ← You are here
│
├── 03-02+First+Graph.ipynb         ← Imports and environment setup
├── 03-03+First+Graph.ipynb         ← State schema and chatbot node
├── 03-04+First+Graph.ipynb         ← Building and compiling the first graph
├── 03-05+Conditional+Edges.ipynb   ← Conditional edge routing (partial)
└── 03-06+Conditional+Edges.ipynb   ← Complete conditional looping graph
```

All notebooks use a shared kernel named `langgraph_env` (a Python 3.11 virtual environment — see [Installation and Setup](#installation-and-setup)).

---

## Notebook Walkthroughs

### 03-02 — Importing Relevant Classes

**Goal:** Get your environment ready and understand what you will be working with.

This notebook introduces every import used throughout the series:

```python
from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import Runnable
from collections.abc import Sequence
```

**Key takeaway:** LangGraph sits on top of the LangChain ecosystem. The two central objects are `StateGraph` (the graph builder) and `TypedDict` (used to define the state schema). `START` and `END` are special sentinel strings that mark where execution enters and exits the graph.

---

### 03-03 — Defining State and Nodes

**Goal:** Define the state your graph will carry and write your first node.

#### State schema

```python
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from collections.abc import Sequence

class State(TypedDict):
    messages: Sequence[BaseMessage]
```

Every node in the graph receives a `State` dictionary and returns a dictionary containing the keys it wants to update. Keeping the schema explicit lets type checkers and the framework validate data flowing through the graph.

#### Node function

```python
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o", seed=365, temperature=0, max_completion_tokens=100)

def chatbot(state: State) -> State:
    response = llm.invoke(state["messages"])
    print(response)
    return {"messages": [response]}
```

A node is just a Python function. It:
1. Accepts the current `State` as its only argument.
2. Does some work (here, an LLM call).
3. Returns a dictionary with the updated state keys.

**Key takeaway:** Nodes are plain Python functions — there is no special base class to inherit from. The framework infers the input/output contract from the `State` type annotation.

---

### 03-04 — Building the First Graph

**Goal:** Wire the node into a graph, compile it, and invoke it end-to-end.

```python
from langgraph.graph import StateGraph, START, END

# 1. Create the graph builder
graph = StateGraph(State)

# 2. Register the node
graph.add_node("chatbot", chatbot)

# 3. Define execution order with edges
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# 4. Compile into an executable Runnable
graph_compiled = graph.compile()
```

Invoking the compiled graph:

```python
result = graph_compiled.invoke({"messages": [HumanMessage(content="Tell me a grook by Piet Hein.")]})
```

Linear flow:

```
START ──► chatbot ──► END
```

**Key takeaway:** Calling `compile()` converts the mutable `StateGraph` builder into an immutable `Runnable`. A compiled LangGraph graph implements the same `invoke / stream / batch` interface as any other LangChain `Runnable`, so it integrates seamlessly with the rest of the ecosystem.

---

### 03-05 — Conditional Edges (Setup)

**Goal:** Introduce the building blocks needed for conditional branching — node functions that capture user input and a routing function that decides where to go next.

Three new nodes are defined:

```python
def ask_question(state: State) -> State:
    user_input = input("Ask a question: ")
    return {"messages": [HumanMessage(content=user_input)]}

def chatbot(state: State) -> State:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def ask_another_question(state: State) -> State:
    user_input = input("Do you want to ask another question? (yes/no): ")
    return {"messages": [HumanMessage(content=user_input)]}
```

A routing function inspects state and returns the **name of the next node** (or `"__end__"` to finish):

```python
from typing import Literal

def routing_function(state: State) -> Literal["ask_question", "__end__"]:
    if state["messages"] and state["messages"][-1].content.strip().lower() == "yes":
        return "ask_question"
    else:
        return "__end__"
```

**Key takeaway:** A routing function is a regular Python function that returns a `str`. `Literal[...]` type hints are optional but recommended — they make it easy to catch typos and to auto-generate documentation.

---

### 03-06 — Conditional Edges (Complete)

**Goal:** Assemble all the pieces into a looping, interactive conversational agent.

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(State)

# Register all three nodes
graph.add_node("ask_question", ask_question)
graph.add_node("chatbot", chatbot)
graph.add_node("ask_another_question", ask_another_question)

# Unconditional edges
graph.add_edge(START, "ask_question")
graph.add_edge("ask_question", "chatbot")
graph.add_edge("chatbot", "ask_another_question")

# Conditional edge: the routing function decides what comes after ask_another_question
graph.add_conditional_edges("ask_another_question", routing_function)

graph_compiled = graph.compile()

# Visualise the graph structure
print(graph_compiled.get_graph().draw_ascii())
```

Execution flow:

```
START
  │
  ▼
ask_question  ◄──────────────────┐
  │                              │
  ▼                              │ (user answers "yes")
chatbot                          │
  │                              │
  ▼                              │
ask_another_question             │
  │                              │
  ├── "yes" ──────────────────── ┘
  │
  └── "no" ──► END
```

**Key takeaway:** `add_conditional_edges(source_node, routing_fn)` is all that is needed to branch. The routing function receives the full current state and returns the name of the next node. Because the graph can cycle back to `ask_question`, this is a proper **agentic loop** — not just a linear chain.

---

## Architecture Deep Dive

### State

The state is a `TypedDict` subclass that acts as the single source of truth for the entire graph execution. Every node reads from it and writes partial updates back to it. LangGraph merges those partial updates into the canonical state object before passing it to the next node.

```python
class State(TypedDict):
    messages: Sequence[BaseMessage]
```

For more advanced use cases, LangGraph supports **reducers** — functions that control how updates are merged (e.g., appending to a list instead of replacing it).

---

### Nodes

A node is any Python callable with the signature:

```python
def my_node(state: State) -> dict:
    ...
    return {"key": updated_value}
```

Nodes can:
- Call LLMs or other APIs.
- Read from or write to external storage.
- Prompt the user for input (human-in-the-loop).
- Run arbitrary business logic.

The returned dictionary only needs to contain the keys the node wants to change. LangGraph merges it with the existing state.

---

### Edges

Unconditional edges are static connections:

```python
graph.add_edge("node_a", "node_b")   # node_a always goes to node_b
graph.add_edge(START, "entry_node")  # execution always starts at entry_node
graph.add_edge("exit_node", END)     # execution always ends after exit_node
```

---

### Conditional Edges and Routing

A conditional edge is attached to a **routing function** that inspects the current state and returns the name of the next node:

```python
def routing_function(state: State) -> Literal["node_x", "node_y", "__end__"]:
    if some_condition(state):
        return "node_x"
    return "node_y"

graph.add_conditional_edges("decision_node", routing_function)
```

The routing function can return any registered node name or the special string `"__end__"` to terminate the graph.

---

### Graph Compilation

`graph.compile()` validates the graph (checks for unreachable nodes, missing edges, etc.) and returns an immutable `Runnable` object. Once compiled, you interact with the graph via the standard LangChain interface:

| Method | Description |
|---|---|
| `invoke(input)` | Run the graph once and return the final state |
| `stream(input)` | Stream state updates as they happen (node by node) |
| `batch(inputs)` | Run multiple inputs in parallel |

---

## Graph Visualizations

### Simple Linear Graph (03-04)

```
         +-----------+
         | __start__ |
         +-----------+
               *
               *
               *
          +--------+
          | chatbot |
          +--------+
               *
               *
               *
          +---------+
          | __end__ |
          +---------+
```

### Conditional Looping Graph (03-06)

```
              +-----------+
              | __start__ |
              +-----------+
                    *
                    *
                    *
             +--------------+
             | ask_question |
             +--------------+
                    *
                    *
                    *
              +---------+
              | chatbot |
              +---------+
                    *
                    *
                    *
       +----------------------+
       | ask_another_question |
       +----------------------+
          **             **
        **                 **
       *                     *
+--------------+         +---------+
| ask_question |         | __end__ |
+--------------+         +---------+
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.11 |
| OpenAI API key | — |
| Jupyter (Lab or Notebook) | Any recent version |

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/HarikaraPrashath/Langgraph-Module-Components-and-Implementation.git
cd Langgraph-Module-Components-and-Implementation
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv langgraph_env
source langgraph_env/bin/activate   # macOS / Linux
# langgraph_env\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install langgraph langchain-core langchain-openai python-dotenv jupyter
```

Full list of packages used across all notebooks:

| Package | Purpose |
|---|---|
| `langgraph` | Graph construction, compilation, and execution |
| `langchain-core` | `BaseMessage`, `HumanMessage`, `Runnable` base types |
| `langchain-openai` | `ChatOpenAI` integration for GPT models |
| `python-dotenv` | Loading API keys from a `.env` file |
| `typing_extensions` | `TypedDict` is available from the built-in `typing` module on Python ≥ 3.8; `typing_extensions` provides newer annotation features (`Annotated`, protocol extras) used in some LangGraph internals |
| `jupyter` | Running the `.ipynb` notebooks |

### 4. Register the virtual environment as a Jupyter kernel

```bash
pip install ipykernel
python -m ipykernel install --user --name=langgraph_env --display-name "langgraph_env"
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
```

The notebooks load this file automatically via `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv()
```

> **Never commit your `.env` file or API keys to version control.** Add `.env` to `.gitignore`.

---

## Running the Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# — or —

# Start classic Jupyter Notebook
jupyter notebook
```

Open the notebooks in order (03-02 → 03-03 → 03-04 → 03-05 → 03-06) and run each cell from top to bottom. The notebooks are cumulative — later notebooks reuse definitions from earlier ones.

> **Tip:** Notebooks 03-05 and 03-06 use `input()` to capture user responses at runtime. Make sure you are running them in a Jupyter environment (not a non-interactive script runner) so the prompts appear correctly.

---

## Key Patterns and Best Practices

### 1. Keep state minimal and explicit

Only store in state what nodes genuinely need to share. Bloated state makes graphs harder to reason about. Use `TypedDict` (or Pydantic models for validation) so every field is documented and type-checked.

### 2. One responsibility per node

Nodes are easier to test, debug, and reuse when they do one thing. A node that calls an LLM should not also write to a database — keep those concerns separate.

### 3. Use `Literal` types in routing functions

```python
def route(state: State) -> Literal["node_a", "node_b", "__end__"]:
    ...
```

This makes the set of possible routes explicit and catches typos at development time rather than at runtime.

### 4. Compile once, invoke many times

`graph.compile()` is a relatively expensive operation. Call it once during initialisation and reuse the compiled graph for all subsequent invocations.

### 5. Visualise early and often

```python
print(graph_compiled.get_graph().draw_ascii())
```

Printing the ASCII graph representation is a quick sanity check that your edges are wired the way you intended.

---

## Further Reading

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph GitHub Repository](https://github.com/langchain-ai/langgraph)
- [LangChain Documentation](https://python.langchain.com/docs/introduction/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [LangGraph How-To Guides](https://langchain-ai.github.io/langgraph/how-tos/)
- [LangGraph Conceptual Guides](https://langchain-ai.github.io/langgraph/concepts/)
