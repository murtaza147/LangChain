import os
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.llms import HuggingFaceEndpoint
from langchain_experimental.tools import PythonREPLTool
from langchain.agents.output_parsers.react_json_single_input import ReActJsonSingleInputOutputParser

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_XmnfHMNC...sYxUbIFKNpma"
llm = HuggingFaceEndpoint(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",temperature=0.01)
#llm = llm.bind(stop = ["Observation:"])
tools = [PythonREPLTool()]

base_prompt = hub.pull("hwchase17/react-json")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=base_prompt,
    output_parser=ReActJsonSingleInputOutputParser()
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=False,
    handle_parsing_errors=True
)

ans = agent_executor.invoke({"input":"write a python function that implements the pythagorean theorem, only output the function as your final answer."})

if "```python" in ans['output']:
    prfx = ans['output'].find("`") + 10
else:
    prfx = ans['output'].find("`") + 4

sufx = ans['output'].find("<") - 3

filt = ans['output'][prfx:sufx]

with open("output.py", "w") as py_file:
    py_file.write(filt)

# custom tools
# custom parsers
