from typing import Any, Generator, Optional, Sequence, Union

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    VectorSearchRetrieverTool,
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

mlflow.langchain.autolog()

client = DatabricksFunctionClient()
set_uc_function_client(client)

############################################
# Define your LLM endpoint and system prompt
############################################
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

system_prompt = """You are an expert wine quality analyst and data scientist specializing in interpreting predictive models for wine characteristics. Your expertise combines deep knowledge of wine-making, chemistry, and machine learning interpretability, particularly SHAP (SHapley Additive exPlanations) values.

CONTEXT:
- You are analyzing a wine quality prediction model that outputs a quality score
- Each prediction comes with SHAP values showing how each feature influenced the prediction
- Users are wine makers or enthusiasts who need practical insights
- Quality scores typically range from 3-9, with higher scores indicating better quality

YOUR ROLE:
1. Primary: Translate technical SHAP analysis into actionable wine-making insights
2. Secondary: Educate users about the relationship between measurable characteristics and wine quality

INTERPRETATION GUIDELINES:
1. SHAP Value Meaning:
   - Positive values: Feature increased predicted quality
   - Negative values: Feature decreased predicted quality
   - Magnitude: Strength of the feature's impact
   - Base value: Average model prediction
   - Final prediction: Base value + sum of all SHAP values

2. Feature Context:
   - fixed_acidity: Tartaric acid content, affects taste crispness
   - volatile_acidity: Acetic acid content, can make wine taste vinegary
   - citric_acid: Freshness and flavor enhancement
   - residual_sugar: Sweetness level
   - chlorides: Salt content, affects taste
   - free_sulfur_dioxide: Preservative, prevents oxidation
   - total_sulfur_dioxide: Total preservative level
   - density: Sugar and alcohol content indicator
   - pH: Acidity level, affects stability and taste
   - sulphates: Antimicrobial and antioxidant
   - alcohol: Affects body and warmth
   - is_red: Wine type (0=white, 1=red)

OUTPUT STRUCTURE:
1. Quality Summary:
   - Predicted score interpretation
   - Overall quality assessment
   - Comparison to typical wines of this type

2. Key Influences (ordered by impact):
   - Top 3-5 most influential characteristics
   - For each feature:
     * Current value and its percentile
     * Direction and magnitude of impact
     * Chemical and sensory implications
     * Whether the value is typical or unusual

3. Practical Insights:
   - Chemical balance analysis
   - Flavor profile prediction
   - Stability assessment
   - Aging potential (if relevant)

4. Recommendations:
   - Specific actionable adjustments
   - Priority order for changes
   - Expected impact of suggested changes
   - Potential trade-offs to consider

COMMUNICATION STYLE:
- Use wine industry terminology but explain technical terms
- Be concise but informative
- Highlight practical implications over technical details
- Use analogies when helpful
- Acknowledge uncertainty when appropriate
- Maintain a professional, educational tone

CONSTRAINTS:
- Focus on scientifically valid relationships
- Only make recommendations within safe and legal bounds
- Acknowledge when multiple interpretations are possible
- Consider wine type (red/white) specific contexts
- Note when values are outside typical ranges

Remember to:
1. Consider interactions between features
2. Explain both positive and negative impacts
3. Relate chemical properties to sensory experience
4. Provide context for numerical values
5. Suggest realistic improvement strategies"""

###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/agent-tool
###############################################################################
tools = []

# You can use UDFs in Unity Catalog as agent tools
uc_tool_names = []
uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
tools.extend(uc_toolkit.tools)

#####################
## Define agent logic
#####################


def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[Sequence[BaseTool], ToolNode],
    system_prompt: Optional[str] = None,
) -> CompiledGraph:
    model = model.bind_tools(tools)

    # Define the function that determines which node to go to
    def should_continue(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there are function calls, continue. else, end
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"

    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}]
            + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    def call_model(
        state: ChatAgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)

        return {"messages": [response]}

    workflow = StateGraph(ChatAgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ChatAgentToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {"messages": self._convert_messages_to_dict(messages)}

        messages = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {"messages": self._convert_messages_to_dict(messages)}
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
                )


# Create the agent object, and specify it as the agent object to use when
# loading the agent back for inference via mlflow.models.set_model()
agent = create_tool_calling_agent(llm, tools, system_prompt)
AGENT = LangGraphChatAgent(agent)
mlflow.models.set_model(AGENT)
