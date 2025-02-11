#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import TavilySearchResults
import dotenv
import logging
import requests
import os
import time
import mlflow
import datetime


# mlflow_uri = "http://192.168.30.21:8080/"
# mlflow.set_tracking_uri(mlflow_uri)
date = datetime.datetime.now().strftime("%Y-%m-%d")
mlflow.set_experiment(f"{date}")

# This will enable autologging for all the components in the LLM chain
mlflow.langchain.autolog()


log = logging.getLogger("uvicorn")
logging.basicConfig(level=logging.INFO)
log_oai = logging.getLogger("openai")
log_oai.setLevel(logging.DEBUG)
dotenv.load_dotenv()

# Initialize FastAPI
app = FastAPI()


# Initialize LangChain agent
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)  # Replace with your API setup


# FastAPI request and response models
class Query(BaseModel):
    user_query: str


class WeatherQuery(BaseModel):
    latitude: float
    longitude: float


class Response(BaseModel):
    response: dict | str


# Dummy document loader and vector database setup
def create_vector_store():
    loader = TextLoader("data.txt")  # Replace with your document or dataset
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store


vector_store = create_vector_store()


@tool
async def math_agent(query: str):
    "Useful for solving math problems."
    result = await model.ainvoke(query)
    return result


web_search_tool = TavilySearchResults(
    max_results=3,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    description="Search the web for the user query when needed to provide more information. Use only as a last resort.",
)


@tool
async def weather_agent(query: WeatherQuery):
    """Provides weather information based on geolocation, requires latitude and longitude. Only use this tool when exact location is well known."""

    query = f"https://api.openweathermap.org/data/2.5/forecast?lat={query.latitude}&lon={query.longitude}&appid={os.getenv('OPEN_WEATHER_API_KEY')}"
    html = requests.get(query).text
    return html


@tool
async def knowledge_agent(query: str):
    """Always use this tool first. Usage: to get information from the knowledge base.
    It is helpful when a user asks for specific knowledge, facts, or insights."""

    messages = [
        (
            "system",
            "Provide keywords to be searched in knowledge base to extract \
                unknown information from knowledge base necessary to answer the user question.",
        ),
        ("user", f"Question: {query}"),
    ]

    what_to_search = await model.ainvoke(messages)
    log.info(f"Searching for: {what_to_search.content}")
    response = vector_store.similarity_search(what_to_search.content, k=3)
    log.info(f"Found: {response}")
    return response


tools = [knowledge_agent, math_agent, weather_agent, web_search_tool]
# agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful RAG assistant, always use knowledge base tool first."),
        ("human", "{input}"),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{agent_scratchpad}"),
    ]
)


agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=9,
)


@app.get("/")
async def home():
    return RedirectResponse(url="/docs")


@app.post("/chat", response_model=Response)
async def chat(query: Query):
    with mlflow.start_run():
        try:
            tstart = time.time()
            response = await agent_executor.ainvoke({"input": query})
            # response = agent_executor.invoke({"input": query})
            tend = time.time()
            log.info(f"Took: {tend-tstart} seconds")
            return Response(response=response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# Running the application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
