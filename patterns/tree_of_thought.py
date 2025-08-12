from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from typing import Dict, List

# Initialize OpenAI model
llm = ChatOpenAI()


# Define Agent State
class StrategyState(Dict):
    business_type: str
    expansion_options: List[str]
    strategy_analysis: Dict[str, str]
    best_strategy: str


# 🟢 Step 1: Generate Expansion Strategies
def generate_expansion_options(state: StrategyState) -> StrategyState:
    prompt = f"""
    The company specializes in {state['business_type']}. Suggest three possible expansion strategies:

    1. Entering a new geographical market.
    2. Launching a new product line.
    3. Partnering with an existing brand.

    Provide a brief overview of each strategy.
    """

    response = llm.invoke([SystemMessage(content="You are a business strategist."), HumanMessage(content=prompt)])

    state["expansion_options"] = response.content.split("\n")[:3]  # Extract first three options
    return state


# 🟢 Step 2: Analyze Each Strategy (Breaking Down Into ToT Paths)
def analyze_strategy(state: StrategyState) -> StrategyState:
    strategy_analysis = {}

    for strategy in state["expansion_options"]:
        prompt = f"""
        Analyze the following business expansion strategy:

        {strategy}

        Evaluate it based on:
        - Cost implications
        - Risk factors
        - Potential return on investment (ROI)

        Provide a structured breakdown.
        """
        response = llm.invoke([SystemMessage(content="You are a business analyst."), HumanMessage(content=prompt)])
        strategy_analysis[strategy] = response.content

    state["strategy_analysis"] = strategy_analysis
    return state


# 🟢 Step 3: Choose the Best Strategy (Final Decision)
def select_best_strategy(state: StrategyState) -> StrategyState:
    prompt = f"""
    Given the following business expansion strategies and their analysis:

    {state['strategy_analysis']}

    Rank these strategies based on:
    - Highest ROI
    - Lowest risk
    - Overall feasibility

    Select the **best** strategy and explain why it is the optimal choice.
    """

    response = llm.invoke(
        [SystemMessage(content="You are an expert business strategist."), HumanMessage(content=prompt)])

    state["best_strategy"] = response.content
    return state


# 🔵 Build the LangGraph Workflow
workflow = StateGraph(StrategyState)

# Adding Nodes
workflow.add_node("generate_expansion_options", generate_expansion_options)
workflow.add_node("analyze_strategy", analyze_strategy)
workflow.add_node("select_best_strategy", select_best_strategy)

# Define Execution Flow
workflow.set_entry_point("generate_expansion_options")
workflow.add_edge("generate_expansion_options", "analyze_strategy")
workflow.add_edge("analyze_strategy", "select_best_strategy")

# Compile Graph
graph = workflow.compile()

# 🟢 Run Example
input_data = {
    "business_type": "AI-based EdTech Startup"
}

result = graph.invoke(input_data)

print("🚀 AI-Generated Expansion Strategies:\n", result["expansion_options"])
print("\n🔍 Strategy Analysis:\n", result["strategy_analysis"])
print("\n🏆 Best Strategy Selected:\n", result["best_strategy"])