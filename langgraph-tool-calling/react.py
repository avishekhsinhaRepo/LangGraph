from langchain_tavily import TavilySearch

@tool
def triple(num: float) -> float:
    return float(num) * 3


tools= [TavilySearch(max_result=1), triple]
