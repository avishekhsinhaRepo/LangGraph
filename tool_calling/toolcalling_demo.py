from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from langchain.agents.output_parsers import ReActSingleInputOutputParser

load_dotenv()

# Set your Azure OpenAI credentials
subscription_key = os.environ["AZURE_OPENAI_API_KEY"]
endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_version = os.environ["AZURE_OPENAI_API_VERSION"]
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]


@tool
def get_restaurant_recommendations(location: str):
    """Provides a list of top restaurant recommendations for a given location."""
    recommendations = {
        "munich": ["Hofbräuhaus", "Augustiner-Keller", "Tantris"],
        "new york": ["Le Bernardin", "Eleven Madison Park", "Joe's Pizza"],
        "paris": ["Le Meurice", "L'Ambroisie", "Bistrot Paul Bert"],
    }
    return recommendations.get(
        location.lower(), ["No recommendations available for this location."]
    )


# Initialize the Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_deployment=deployment,
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# TODO: Bind the tool to the model
tools = [get_restaurant_recommendations]

llm_with_tools = llm.bind_tools(tools)

messages = [HumanMessage("Recommend some restaurants in Munich.")]


# TODO: Invoke the llm
llm_output = llm_with_tools.invoke(messages)
print("LLM Raw Output:", llm_output)
