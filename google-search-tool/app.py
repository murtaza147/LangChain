import os
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool

os.environ["GOOGLE_CSE_ID"] = "863ea...fd54c45"
os.environ["GOOGLE_API_KEY"] = "AIzaSyAZTjV...JnuB7D0gR7R-des"

tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=GoogleSearchAPIWrapper(k=1).run,
)

query = input("Search for: ")
print(tool.run(query))
