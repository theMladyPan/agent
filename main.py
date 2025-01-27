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
import dotenv
import logging
import requests
import os

log = logging.getLogger("uvicorn")
logging.basicConfig(level=logging.INFO)
dotenv.load_dotenv()

# Initialize FastAPI
app = FastAPI()


# Initialize LangChain agent
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)  # Replace with your API setup


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
    loader = TextLoader("dummy_data.txt")  # Replace with your document or dataset
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store


vector_store = create_vector_store()


@tool
def math_agent(query: str):
    "Useful for solving math problems."
    log.info(f"### Math Bot: {query}")
    result = model.invoke(query)
    return result


@tool
def web_agent(query):
    """Searches the web for relevant information based on input if not found in knowledge base."""
    log.info(f"### Web Bot: {query} on web")
    query = f"https://www.google.com/search?q={query}"
    html = requests.get(query).text
    return html


@tool
def weather_agent(query: WeatherQuery):
    """Provides weather information based on geolocation, requires latitude and longitude."""

    query = f"https://api.openweathermap.org/data/2.5/forecast?lat={query.latitude}&lon={query.longitude}&appid={os.getenv('OPEN_WEATHER_API_KEY')}"
    log.info(f"### Weather Bot: Searching for: {query}")
    html = requests.get(query).text
    return html


@tool
def knowledge_agent(query: str):
    """Use this tool to search for information from the knowledge base.
    It is helpful when a user asks for specific knowledge, facts, or insights."""

    log.info(f"### Knowledge bot: {query}")
    response = vector_store.similarity_search(query, k=1)
    log.info(f"Found: {response}")
    return response


tools = [knowledge_agent, math_agent, weather_agent, web_agent]
# agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful RAG assistant, always look into knowledge base first."),
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
    try:
        # response = await agent_executor.ainvoke({"input": query})
        response = agent_executor.invoke({"input": query})
        return Response(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Running the application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
