from dotenv import load_dotenv
import pypdfium2 as pdfium
from langchain import hub
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.llms import HuggingFaceEndpoint

# get raw text
def get_text(pdf):
    text = ""
    pdf_reader = pdfium.PdfDocument(pdf)
    for i in range(len(pdf_reader)):
        page = pdf_reader.get_page(i)
        textpage = page.get_textpage()
        text += textpage.get_text_bounded() + "\n"
    return text

text = get_text("LawsOfChess.pdf")

load_dotenv()

google_tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=GoogleSearchAPIWrapper().run
)

llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2",temperature=0.01)
tools = [google_tool]
instructions = f"""You are an agent designed to answer questions related to this text {text}. If it seems like the text does not contain the answer to the question,
you have access to a google search tool, which you can use to answer the question.
"""

base_prompt = hub.pull("langchain-ai/react-agent-template")
prompt = base_prompt.partial(instructions=instructions)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

query = input("Search PDF for: ")
print(agent_executor.invoke({"input": query}))