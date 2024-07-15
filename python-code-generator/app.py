import os
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_community.llms import HuggingFaceEndpoint
from langchain_experimental.tools.python.tool import PythonREPLTool

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_XmnfHMNCDP...YxUbIFKNpma"
llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2",temperature=0.01)

agent = create_python_agent(
    llm=llm,
    tool=PythonREPLTool(),
    verbose=True,
)

ans = agent.invoke({"input":"Write and return a python function that implements the quadratic formula"})
print(ans)
print(ans["output"])
