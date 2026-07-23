## Problem: Inconsistent Tool Calling and API Compatibility Across LLMs

Developers building agentic AI systems frequently encounter issues with inconsistent tool calling behavior and API compatibility when integrating various Large Language Model (LLM) providers (e.g., OpenAI, Gemini, custom local models) or proxies. This leads to problems such as:
- `OutputParserException` errors due to mismatched tool schemas or incorrect function call formats.
- Difficulties in porting agents between different LLM backends.
- Challenges in standardizing tool definitions and ensuring they are correctly interpreted by diverse models.
- Increased development time and maintenance overhead.

This problem affects anyone building multi-LLM agent systems, particularly those aiming for portability, robustness, and easy integration of new tools or LLM providers.

## Why this project shape/stack was chosen

This prototype adopts a Python package structure, specifically focusing on a `ToolSchemaAdapter` class. Python is the de facto language for most agentic AI development, and a class-based approach allows for:
- **Encapsulation**: Centralizing logic for schema conversion and validation.
- **Extensibility**: Easily adding new adapters for different LLM providers or tool specifications.
- **Integration**: Seamlessly fitting into existing Python-based agent frameworks (e.g., LangChain, LlamaIndex).
- **Clarity**: Providing a clear, well-defined interface for managing tool schemas.

The core idea is to provide a standardized way to define tools and then adapt them to the specific requirements of different LLM APIs, minimizing the "impedance mismatch" that causes parsing errors. We demonstrate this with a simple, provider-agnostic `Tool` definition and an adapter for the Gemini API's `FunctionDeclaration` format.

## Setup and Usage Instructions

1.  **Clone the repository (or create the files manually).**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the example:**
    ```bash
    python main.py
    ```
    This will demonstrate how a generic `Tool` definition is adapted to Gemini's `FunctionDeclaration` format and then used to simulate a tool call.

## GEMINI_API_KEY Requirement

GEMINI_API_KEY is **NOT** required to run this prototype. The prototype focuses on the *schema adaptation* aspect, not on making live LLM calls. It simulates how an LLM *would* call a tool based on the adapted schema.
