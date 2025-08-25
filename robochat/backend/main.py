import os
import re
import json
from dotenv import load_dotenv

import requests
from serpapi.google_search import GoogleSearch
from bs4 import BeautifulSoup

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import chromadb

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain.agents import tool
from langchain_tavily import TavilySearch

load_dotenv()

# -----------------------------
# FastAPI app setup
# -----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for dev: allow all. Restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Hugging Face model via LangChain
# -----------------------------
model = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv('HF_TOKEN'),
    model_name="openai/gpt-oss-120b:fireworks-ai",
)

# -----------------------------
# Chroma Cloud connection
# -----------------------------

client = chromadb.HttpClient(
    ssl=True,
    host="api.trychroma.com",
    tenant=os.getenv('TENANT'),
    database=os.getenv('DATABASE'),
    headers={"x-chroma-token": os.getenv('CHROMA')},
)

# -----------------------------
# Embeddings
# -----------------------------
embedding_fn = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)

# -----------------------------
# Vectorstore
# -----------------------------
vectorstore = Chroma(
    client=client,
    collection_name="Rag-docs",
    embedding_function=embedding_fn,
)

# -----------------------------
# Agent Tools
# -----------------------------
tavily_search_tool = TavilySearch(
    max_results=3,
    topic="general",
)

@tool
def retrieval_tool(query: str) -> str:
  """Fetch data from VectorDB."""
  retriever = vectorstore.as_retriever(search_kwargs={"k": 5, "score_threshold": 0.3}, search_type="similarity_score_threshold")
  return retriever.invoke(query)

@tool
def get_location_tool() -> dict:
    """Fetch the user's location using IP address."""
    providers = [
        "https://ipwho.is/",
        "https://ipapi.co/json/",
    ]

    for url in providers:
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()

            # normalize response keys
            if "city" in data and "country_name" in data:
                return {
                    "city": data.get("city"),
                    "country": data.get("country_name"),
                    "lat": data.get("latitude"),
                    "lon": data.get("longitude")
                }
            elif "city" in data and "country" in data:
                return {
                    "city": data.get("city"),
                    "country": data.get("country"),
                    "lat": data.get("latitude") or data.get("lat"),
                    "lon": data.get("longitude") or data.get("lon")
                }
            elif "location" in data and isinstance(data["location"], dict):
                loc = data["location"]
                return {
                    "city": loc.get("city"),
                    "country": loc.get("country"),
                    "lat": loc.get("latitude"),
                    "lon": loc.get("longitude")
                }

        except Exception as e:
            print(f"Provider {url} failed: {e}")
            continue

    return {"error": "Could not detect location from any free provider"}

def search_stores_nearby(component: str, location: str, max_results: int = 10):
    """ Search for nearby stores selling a component using SerpAPI (Google Search). """
    
    query = f"Buy {component} in {location['city']}, {location['country']}"
    
    params = {
        "engine": "google",
        "q": query,
        "hl": "en",
        "gl": "us",
        "api_key": os.getenv('SERPAPI_KEY'),  # make sure your key is set in env
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()

    stores = []
    
    if "organic_results" in results:
        for item in results["organic_results"][:max_results]:
            stores.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
                "source": item.get("source")
            })
    
    return stores


def clean_text(value: str) -> str:
    """Remove scripts, boilerplate, and weird junk from scraped text."""
    if not value:
        return None
    value = re.sub(r"\s+", " ", value)  # collapse whitespace
    value = re.sub(r"(BEGIN app block|Google Tag Manager.*|if lt IE.*)", "", value, flags=re.I)
    return value.strip() or None

def scrape_product_details(url: str):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=12)
    except Exception as e:
        return {"error": str(e), "url": url}

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts, styles, comments
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    product = {
        "title": None,
        "price": None,
        "availability": None,
        "description": None
    }

    # 1. JSON-LD (best source)
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            if isinstance(data, dict) and data.get("@type") == "Product":
                product["title"] = product["title"] or clean_text(data.get("name"))
                product["description"] = product["description"] or clean_text(data.get("description"))
                if isinstance(data.get("offers"), dict):
                    product["price"] = product["price"] or clean_text(data["offers"].get("price"))
                    product["availability"] = product["availability"] or clean_text(data["offers"].get("availability"))
        except Exception:
            continue

    # 2. Itemprop microdata
    if not product["price"]:
        price_tag = soup.find(attrs={"itemprop": "price"})
        if price_tag:
            product["price"] = clean_text(price_tag.get("content") or price_tag.get_text())

    if not product["availability"]:
        avail_tag = soup.find(attrs={"itemprop": "availability"})
        if avail_tag:
            product["availability"] = clean_text(avail_tag.get("content") or avail_tag.get_text())

    # 3. Fallback selectors
    if not product["title"]:
        title_tag = soup.find("h1") or soup.find("title")
        if title_tag:
            product["title"] = clean_text(title_tag.get_text())

    if not product["description"]:
        desc = soup.find("meta", attrs={"name": "description"})
        if desc and desc.get("content"):
            product["description"] = clean_text(desc["content"])

    if not product["price"]:
        price_match = soup.find(string=re.compile(r"(\$|USD|EUR|EGP|LE)\s?\d+(\.\d{1,2})?"))
        if price_match:
            product["price"] = clean_text(price_match)

    if not product["availability"]:
        avail_match = soup.find(string=re.compile(r"(In Stock|Out of Stock|Available|Unavailable)", re.I))
        if avail_match:
            product["availability"] = clean_text(avail_match)


    return {k: clean_text(v) if isinstance(v, str) else v for k, v in product.items()}

@tool
def find_product_nearby_tool(component: str, location: dict) -> str:
    """Find the required product in stores near the user's location"""
    stores = search_stores_nearby(component, location)
    details = []
    for store in stores:
        details.append(scrape_product_details(store["link"]))
    return(details)
# -----------------------------
# Agent
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    
class Agent:

    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}
    

# -----------------------------
# Prompt + Initiallization
# -----------------------------
prompt = """You are a smart engineer assistant specialized in robotics.

Follow these steps IN ORDER for EVERY user question:

STEP 1: Use retrieval_tool to search for information about the user's question.

STEP 2: If retrieval_tool doesn't provide sufficient information, use tavily_search_tool.

STEP 3: ALWAYS use get_location_tool to find the user's location.

STEP 4: ALWAYS use find_product_nearby_tool to find components in nearby stores.

STEP 5: ALWAYS add the product details from stores for the use to buy.

You MUST complete ALL FOUR STEPS. Do not skip steps 3 and 4.

When processing results from find_product_nearby_tool:
- Extract complete product details
- Return multiple offers if they have complete information
- Include store location, URL, price, and availability

Remember: You MUST use get_location_tool and find_product_nearby_tool for EVERY query."""

bot = Agent(model, [get_location_tool, find_product_nearby_tool, retrieval_tool, tavily_search_tool], system=prompt)


# -----------------------------
# API Schemas
# -----------------------------
class Query(BaseModel):
    question: str
    top_k: int = 3

# -----------------------------
# Routes
# -----------------------------

@app.post("/ask")
def ask(query: Query):
    result = bot.graph.invoke({"messages": [HumanMessage(content=query.question)]})
    
    # Extract answer
    answer = result['messages'][-1].content
    
    # Find the ToolMessage(s) and extract documents/sources
    sources = []
    
    for message in result['messages']:
        if isinstance(message, ToolMessage):
            # Check which tool was used
            if message.name == 'retrieval_tool':
                # Parse RAG documents using regex
                content = message.content
                
                # Extract all page_content values using regex
                pattern = r"page_content=['\"]([^'\"]*)['\"]"
                matches = re.findall(pattern, content)
                
                for match in matches:
                    # Unescape any escaped characters
                    page_content = match.replace('\\n', '\n').replace('\\\'', "'").replace('\\"', '"')
                    sources.append(page_content)
                    
            elif message.name == 'tavily_search':
                # Parse web search results
                try:
                    # Parse the JSON string
                    search_data = json.loads(message.content)
                    
                    # Extract results if they exist
                    if 'results' in search_data:
                        for result in search_data['results']:
                            # Create a formatted string with the web search result
                            source_text = f"{result.get('title', '')}\n{result.get('content', '')}\nSource: {result.get('url', '')}"
                            sources.append(source_text)
                except json.JSONDecodeError:
                    # If JSON parsing fails, add raw content
                    sources.append(message.content)

    return {"answer": answer, "context": sources}