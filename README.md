# LangGraph Learning Repository

A comprehensive collection of LangGraph examples and implementations for building AI agents with complex workflows and state management.

## 🚀 Overview

This repository contains practical examples and implementations using LangGraph, a library for building stateful, multi-actor applications with Large Language Models (LLMs). It demonstrates various patterns including tool calling, conditional routing, human-in-the-loop interactions, and persistent memory.

## 📋 Prerequisites

- Python 3.8+
- Azure OpenAI or OpenAI API credentials
- Basic understanding of LangChain and AI agents

## 🛠 Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd LangGraph
   ```

2. **Create and activate virtual environment:**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**

   ```powershell
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory:
   ```env
   AZURE_OPENAI_API_KEY=your_api_key_here
   AZURE_OPENAI_ENDPOINT=your_endpoint_here
   AZURE_OPENAI_API_VERSION=2024-02-15-preview
   AZURE_OPENAI_DEPLOYMENT=your_deployment_name
   ```

## 📁 Project Structure

```
├── assignments/           # Learning assignments and exercises
├── asyncandstreaming/     # Asynchronous operations and streaming examples
├── basics/               # Fundamental LangGraph concepts
├── conditional_routing/   # Conditional logic and routing examples
├── hitl/                 # Human-in-the-loop implementations
├── patterns/             # Advanced design patterns
├── persistent_memory/    # Memory and state persistence
├── rag/                  # Retrieval-Augmented Generation examples
├── reducers/             # State reduction and aggregation
├── subgraphs/            # Nested and modular graph structures
├── tool_calling/         # Tool integration and function calling
├── usecase/              # Real-world application examples
└── util/                 # Utility functions and helpers
```

## 🎯 Key Examples

### 1. Tool Calling (`tool_calling/toolnode_auto.py`)

Demonstrates automatic tool selection and execution with restaurant recommendations:

- **Features:** Tool binding, conditional routing, state management
- **Tools:** Restaurant recommendations, table booking
- **Usage:** Shows how agents can automatically select and execute appropriate tools

### 2. Conditional Routing (`conditional_routing/customer_service.py`)

Customer service agent with dynamic routing:

- **Features:** Multi-path workflows, conditional logic
- **Use Case:** Route customer queries to appropriate handlers

### 3. Human-in-the-Loop (`hitl/code_generator.py`)

Interactive code generation with human feedback:

- **Features:** Human approval workflows, iterative refinement
- **Use Case:** Generate and review code with human oversight

### 4. RAG Implementation (`rag/rag_demo.py`)

Retrieval-Augmented Generation examples:

- **Features:** Document retrieval, context injection
- **Use Case:** Answer questions using external knowledge

### 5. Real-world Application (`usecase/`)

Insurance claim processing system:

- **Features:** Complete workflow, Chainlit UI, API endpoints
- **Components:**
  - `claim_processing_agent.py` - Core agent logic
  - `app.py` - Chainlit web interface
  - `claim_processing_api.py` - FastAPI endpoints

## 🔧 Core Technologies

- **LangGraph:** State management and workflow orchestration
- **LangChain:** LLM integration and tool management
- **Azure OpenAI:** Language model provider
- **Chainlit:** Web UI for chat applications
- **FastAPI:** REST API development
- **PostgreSQL:** Persistent storage with psycopg

## 📖 Learning Path

1. **Start with Basics:**

   - `basics/hello_world.py` - Simple LangGraph introduction
   - `basics/hello_world_pydantic.py` - Structured data handling

2. **Tool Integration:**

   - `tool_calling/toolcalling_demo.py` - Manual tool calling
   - `tool_calling/toolnode_auto.py` - Automatic tool selection

3. **Advanced Patterns:**

   - `patterns/reflection_agent.py` - Self-reflection patterns
   - `patterns/parallel_processing.py` - Concurrent execution
   - `patterns/tree_of_thought.py` - Complex reasoning

4. **Production Features:**
   - `persistent_memory/permanent_memory.py` - State persistence
   - `asyncandstreaming/` - Performance optimization
   - `usecase/` - Complete application

## 🚀 Quick Start

1. **Run a basic example:**

   ```powershell
   python basics/hello_world.py
   ```

2. **Try tool calling:**

   ```powershell
   python tool_calling/toolnode_auto.py
   ```

3. **Launch the insurance claim app:**
   ```powershell
   chainlit run usecase/app.py
   ```

## 🔍 Key Concepts Demonstrated

### State Management

- **MessagesState:** Managing conversation history
- **Custom States:** Domain-specific state structures
- **State Persistence:** Long-term memory storage

### Workflow Patterns

- **Linear Workflows:** Sequential processing
- **Conditional Routing:** Dynamic path selection
- **Parallel Processing:** Concurrent task execution
- **Human-in-the-Loop:** Interactive workflows

### Tool Integration

- **Function Calling:** LLM-driven tool selection
- **Custom Tools:** Domain-specific functionality
- **Tool Chaining:** Sequential tool execution

### Advanced Features

- **Async Operations:** Non-blocking execution
- **Streaming:** Real-time response generation
- **Checkpointing:** Workflow state preservation
- **Subgraphs:** Modular workflow composition

## 🔧 Configuration

### Environment Variables

- `AZURE_OPENAI_API_KEY` - Your Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT` - Azure OpenAI service endpoint
- `AZURE_OPENAI_API_VERSION` - API version (e.g., "2024-02-15-preview")
- `AZURE_OPENAI_DEPLOYMENT` - Model deployment name

### Model Configuration

The project uses Azure ChatOpenAI by default. To switch providers, modify the LLM initialization in individual files.

## 🧪 Testing

Run individual examples to test functionality:

```powershell
# Test basic functionality
python basics/hello_world.py

# Test tool calling
python tool_calling/toolnode_auto.py

# Test RAG implementation
python rag/rag_demo.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your examples or improvements
4. Test thoroughly
5. Submit a pull request

## 📚 Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [Azure OpenAI Documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/openai/)

## 📄 License

This project is for educational purposes. Please check individual dependencies for their respective licenses.

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors:**

   - Ensure `.env` file is properly configured
   - Verify API key validity and permissions

2. **Import Errors:**

   - Activate virtual environment: `.\.venv\Scripts\Activate.ps1`
   - Install dependencies: `pip install -r requirements.txt`

3. **Tool Calling Issues:**

   - Check model supports function calling
   - Verify tool definitions are properly formatted

4. **Memory/Persistence Issues:**
   - Ensure PostgreSQL is running (if using postgres checkpoints)
   - Check database connection parameters

### Getting Help

- Check the documentation links above
- Review similar examples in the repository
- Test with simpler examples first

---

**Note:** This is a learning repository with examples ranging from basic concepts to production-ready applications. Start with the basics and gradually work through more complex examples.
