# Databricks notebook source
# MAGIC %md
# MAGIC #Tool-calling Agent
# MAGIC
# MAGIC This is an auto-generated notebook created by an AI Playground export.
# MAGIC
# MAGIC This notebook uses [Mosaic AI Agent Framework](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/build-genai-apps) to recreate your agent from the AI Playground. It  demonstrates how to develop, manually test, evaluate, log, and deploy a tool-calling agent in LangGraph.
# MAGIC
# MAGIC The agent code implements [MLflow's ChatAgent](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ChatAgent) interface, a Databricks-recommended open-source standard that simplifies authoring multi-turn conversational agents, and is fully compatible with Mosaic AI agent framework functionality.
# MAGIC
# MAGIC  **_NOTE:_**  This notebook uses LangChain, but AI Agent Framework is compatible with any agent authoring framework, including LlamaIndex or pure Python agents written with the OpenAI SDK.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "")
dbutils.widgets.text("schema_name", "")

# COMMAND ----------

# Training Wine Quality Model with SHAP Explanations

catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")

# Create schema if it doesn't exist
query = f"""
CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}
"""
spark.sql(query)

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow langchain langgraph==0.3.4 databricks-langchain pydantic databricks-agents unitycatalog-langchain[databricks] uv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Define the agent in code
# MAGIC Below we define our agent code in a single cell, enabling us to easily write it to a local Python file for subsequent logging and deployment using the `%%writefile` magic command.
# MAGIC
# MAGIC For more examples of tools to add to your agent, see [docs](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/agent-tool).

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC from typing import Any, Generator, Optional, Sequence, Union
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks_langchain import (
# MAGIC     ChatDatabricks,
# MAGIC     VectorSearchRetrieverTool,
# MAGIC     DatabricksFunctionClient,
# MAGIC     UCFunctionToolkit,
# MAGIC     set_uc_function_client,
# MAGIC )
# MAGIC from langchain_core.language_models import LanguageModelLike
# MAGIC from langchain_core.runnables import RunnableConfig, RunnableLambda
# MAGIC from langchain_core.tools import BaseTool
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.graph import CompiledGraph
# MAGIC from langgraph.graph.state import CompiledStateGraph
# MAGIC from langgraph.prebuilt.tool_node import ToolNode
# MAGIC from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
# MAGIC from mlflow.pyfunc import ChatAgent
# MAGIC from mlflow.types.agent import (
# MAGIC     ChatAgentChunk,
# MAGIC     ChatAgentMessage,
# MAGIC     ChatAgentResponse,
# MAGIC     ChatContext,
# MAGIC )
# MAGIC
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC client = DatabricksFunctionClient()
# MAGIC set_uc_function_client(client)
# MAGIC
# MAGIC ############################################
# MAGIC # Define your LLM endpoint and system prompt
# MAGIC ############################################
# MAGIC LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC system_prompt = """You are an expert wine quality analyst and data scientist specializing in interpreting predictive models for wine characteristics. Your expertise combines deep knowledge of wine-making, chemistry, and machine learning interpretability, particularly SHAP (SHapley Additive exPlanations) values.
# MAGIC
# MAGIC CONTEXT:
# MAGIC - You are analyzing a wine quality prediction model that outputs a quality score
# MAGIC - Each prediction comes with SHAP values showing how each feature influenced the prediction
# MAGIC - Users are wine makers or enthusiasts who need practical insights
# MAGIC - Quality scores typically range from 3-9, with higher scores indicating better quality
# MAGIC
# MAGIC YOUR ROLE:
# MAGIC 1. Primary: Translate technical SHAP analysis into actionable wine-making insights
# MAGIC 2. Secondary: Educate users about the relationship between measurable characteristics and wine quality
# MAGIC
# MAGIC INTERPRETATION GUIDELINES:
# MAGIC 1. SHAP Value Meaning:
# MAGIC    - Positive values: Feature increased predicted quality
# MAGIC    - Negative values: Feature decreased predicted quality
# MAGIC    - Magnitude: Strength of the feature's impact
# MAGIC    - Base value: Average model prediction
# MAGIC    - Final prediction: Base value + sum of all SHAP values
# MAGIC
# MAGIC 2. Feature Context:
# MAGIC    - fixed_acidity: Tartaric acid content, affects taste crispness
# MAGIC    - volatile_acidity: Acetic acid content, can make wine taste vinegary
# MAGIC    - citric_acid: Freshness and flavor enhancement
# MAGIC    - residual_sugar: Sweetness level
# MAGIC    - chlorides: Salt content, affects taste
# MAGIC    - free_sulfur_dioxide: Preservative, prevents oxidation
# MAGIC    - total_sulfur_dioxide: Total preservative level
# MAGIC    - density: Sugar and alcohol content indicator
# MAGIC    - pH: Acidity level, affects stability and taste
# MAGIC    - sulphates: Antimicrobial and antioxidant
# MAGIC    - alcohol: Affects body and warmth
# MAGIC    - is_red: Wine type (0=white, 1=red)
# MAGIC
# MAGIC OUTPUT STRUCTURE:
# MAGIC 1. Quality Summary:
# MAGIC    - Predicted score interpretation
# MAGIC    - Overall quality assessment
# MAGIC    - Comparison to typical wines of this type
# MAGIC
# MAGIC 2. Key Influences (ordered by impact):
# MAGIC    - Top 3-5 most influential characteristics
# MAGIC    - For each feature:
# MAGIC      * Current value and its percentile
# MAGIC      * Direction and magnitude of impact
# MAGIC      * Chemical and sensory implications
# MAGIC      * Whether the value is typical or unusual
# MAGIC
# MAGIC 3. Practical Insights:
# MAGIC    - Chemical balance analysis
# MAGIC    - Flavor profile prediction
# MAGIC    - Stability assessment
# MAGIC    - Aging potential (if relevant)
# MAGIC
# MAGIC 4. Recommendations:
# MAGIC    - Specific actionable adjustments
# MAGIC    - Priority order for changes
# MAGIC    - Expected impact of suggested changes
# MAGIC    - Potential trade-offs to consider
# MAGIC
# MAGIC COMMUNICATION STYLE:
# MAGIC - Use wine industry terminology but explain technical terms
# MAGIC - Be concise but informative
# MAGIC - Highlight practical implications over technical details
# MAGIC - Use analogies when helpful
# MAGIC - Acknowledge uncertainty when appropriate
# MAGIC - Maintain a professional, educational tone
# MAGIC
# MAGIC CONSTRAINTS:
# MAGIC - Focus on scientifically valid relationships
# MAGIC - Only make recommendations within safe and legal bounds
# MAGIC - Acknowledge when multiple interpretations are possible
# MAGIC - Consider wine type (red/white) specific contexts
# MAGIC - Note when values are outside typical ranges
# MAGIC
# MAGIC Remember to:
# MAGIC 1. Consider interactions between features
# MAGIC 2. Explain both positive and negative impacts
# MAGIC 3. Relate chemical properties to sensory experience
# MAGIC 4. Provide context for numerical values
# MAGIC 5. Suggest realistic improvement strategies"""
# MAGIC
# MAGIC ###############################################################################
# MAGIC ## Define tools for your agent, enabling it to retrieve data or take actions
# MAGIC ## beyond text generation
# MAGIC ## To create and see usage examples of more tools, see
# MAGIC ## https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/agent-tool
# MAGIC ###############################################################################
# MAGIC tools = []
# MAGIC
# MAGIC # You can use UDFs in Unity Catalog as agent tools
# MAGIC uc_tool_names = []
# MAGIC uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
# MAGIC tools.extend(uc_toolkit.tools)
# MAGIC
# MAGIC #####################
# MAGIC ## Define agent logic
# MAGIC #####################
# MAGIC
# MAGIC
# MAGIC def create_tool_calling_agent(
# MAGIC     model: LanguageModelLike,
# MAGIC     tools: Union[Sequence[BaseTool], ToolNode],
# MAGIC     system_prompt: Optional[str] = None,
# MAGIC ) -> CompiledGraph:
# MAGIC     model = model.bind_tools(tools)
# MAGIC
# MAGIC     # Define the function that determines which node to go to
# MAGIC     def should_continue(state: ChatAgentState):
# MAGIC         messages = state["messages"]
# MAGIC         last_message = messages[-1]
# MAGIC         # If there are function calls, continue. else, end
# MAGIC         if last_message.get("tool_calls"):
# MAGIC             return "continue"
# MAGIC         else:
# MAGIC             return "end"
# MAGIC
# MAGIC     if system_prompt:
# MAGIC         preprocessor = RunnableLambda(
# MAGIC             lambda state: [{"role": "system", "content": system_prompt}]
# MAGIC             + state["messages"]
# MAGIC         )
# MAGIC     else:
# MAGIC         preprocessor = RunnableLambda(lambda state: state["messages"])
# MAGIC     model_runnable = preprocessor | model
# MAGIC
# MAGIC     def call_model(
# MAGIC         state: ChatAgentState,
# MAGIC         config: RunnableConfig,
# MAGIC     ):
# MAGIC         response = model_runnable.invoke(state, config)
# MAGIC
# MAGIC         return {"messages": [response]}
# MAGIC
# MAGIC     workflow = StateGraph(ChatAgentState)
# MAGIC
# MAGIC     workflow.add_node("agent", RunnableLambda(call_model))
# MAGIC     workflow.add_node("tools", ChatAgentToolNode(tools))
# MAGIC
# MAGIC     workflow.set_entry_point("agent")
# MAGIC     workflow.add_conditional_edges(
# MAGIC         "agent",
# MAGIC         should_continue,
# MAGIC         {
# MAGIC             "continue": "tools",
# MAGIC             "end": END,
# MAGIC         },
# MAGIC     )
# MAGIC     workflow.add_edge("tools", "agent")
# MAGIC
# MAGIC     return workflow.compile()
# MAGIC
# MAGIC
# MAGIC class LangGraphChatAgent(ChatAgent):
# MAGIC     def __init__(self, agent: CompiledStateGraph):
# MAGIC         self.agent = agent
# MAGIC
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC
# MAGIC         messages = []
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 messages.extend(
# MAGIC                     ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
# MAGIC                 )
# MAGIC         return ChatAgentResponse(messages=messages)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> Generator[ChatAgentChunk, None, None]:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 yield from (
# MAGIC                     ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
# MAGIC                 )
# MAGIC
# MAGIC
# MAGIC # Create the agent object, and specify it as the agent object to use when
# MAGIC # loading the agent back for inference via mlflow.models.set_model()
# MAGIC agent = create_tool_calling_agent(llm, tools, system_prompt)
# MAGIC AGENT = LangGraphChatAgent(agent)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the `agent` as an MLflow model
# MAGIC Determine Databricks resources to specify for automatic auth passthrough at deployment time
# MAGIC - **TODO**: If your Unity Catalog tool queries a [vector search index](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/unstructured-retrieval-tools) or leverages [external functions](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/external-connection-tools), you need to include the dependent vector search index and UC connection objects, respectively, as resources. See [docs](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/log-agent#specify-resources-for-automatic-authentication-passthrough) for more details.
# MAGIC
# MAGIC Log the agent as code from the `agent.py` file. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).

# COMMAND ----------

# Determine Databricks resources to specify for automatic auth passthrough at deployment time
import mlflow
from agent import tools, LLM_ENDPOINT_NAME
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool

resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]
for tool in tools:
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "Analyze this wine quality prediction:\n\nPredicted Quality Score: 6.8\n\nInput Features and Their Impacts (ordered by importance):\n- alcohol: value = 11.8, SHAP value = 0.842 (increased the prediction)\n- volatile_acidity: value = 0.28, SHAP value = -0.523 (decreased the prediction)\n- sulphates: value = 0.65, SHAP value = 0.412 (increased the prediction)\n- total_sulfur_dioxide: value = 145.0, SHAP value = -0.315 (decreased the prediction)\n- chlorides: value = 0.087, SHAP value = 0.298 (increased the prediction)\n- free_sulfur_dioxide: value = 32.0, SHAP value = 0.245 (increased the prediction)\n- density: value = 0.9956, SHAP value = -0.187 (decreased the prediction)\n- pH: value = 3.42, SHAP value = 0.156 (increased the prediction)\n- citric_acid: value = 0.31, SHAP value = 0.142 (increased the prediction)\n- fixed_acidity: value = 7.2, SHAP value = 0.089 (increased the prediction)\n- residual_sugar: value = 2.8, SHAP value = 0.076 (increased the prediction)\n- is_red: value = 1, SHAP value = 0.065 (increased the prediction)\n\nPlease provide a detailed interpretation following the format specified in your instructions."
        }
    ]
}

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model="agent.py",
        input_example=input_example,
        resources=resources,
        extra_pip_requirements=[
            "databricks-connect"
        ]
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform pre-deployment validation of the agent
# MAGIC Before registering and deploying the agent, we perform pre-deployment checks via the [mlflow.models.predict()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict) API. See [documentation](https://learn.microsoft.com/azure/databricks/machine-learning/model-serving/model-serving-debug#validate-inputs) for details

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"messages": [{"role": "user", "content": "Hello!"}]},
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = catalog_name
schema = schema_name
model_name = "wine_quality_interpreter"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents
from databricks.sdk.service.serving import ServedModelInputWorkloadSize

deployment = agents.deploy(
    model_name=UC_MODEL_NAME,
    model_version=uc_registered_model_info.version,
    scale_to_zero=True,
    workload_size=ServedModelInputWorkloadSize.SMALL,
    tags={
        "endpointSource": "playground",
        "environment": "dev",
        "purpose": "wine-quality-interpretation"
    }
)
