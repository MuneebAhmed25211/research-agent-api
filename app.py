# All imports should be at the TOP of the file
from dotenv import load_dotenv
load_dotenv()

import os
from typing import Annotated
from typing_extensions import TypedDict
from serpapi.google_search import GoogleSearch
from fastapi import FastAPI          # ← move here
from pydantic import BaseModel       # ← move here
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from fastapi.middleware.cors import CORSMiddleware
class State(TypedDict):
    messages: Annotated[list, add_messages]
    research_plan: str
    search_results: str
    final_report: str


@tool
def planner_tool(query: str) -> str:
    """Break a research topic into sub-questions."""
    llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")
    response = llm.invoke([
        SystemMessage(content="You are a research planner. Break the given topic into 3-5 focused sub-questions. Return them as a numbered list."),
        HumanMessage(content=f"Topic: {query}")
    ])
    return response.content


def planner_node(state: State) -> dict:
    messages = state.get("messages", [])
    query = messages[-1].content if messages else ""
    plan = planner_tool.invoke({"query": query})
    return {"research_plan": plan}


def search_node(state: State) -> dict:
    research_plan = state.get("research_plan", "")
    lines = [line.strip() for line in research_plan.split("\n")
             if line.strip() and line.strip()[0].isdigit()]
    
    all_results = []
    for question in lines:
        try:
            search = GoogleSearch({
                "q": question,
                "api_key": os.environ["SERPAPI_API_KEY"],
                "num": 3
            })
            response = search.get_dict()
            for item in response.get("organic_results", []):
                all_results.append({
                    "question": question,
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet", "")
                })
        except Exception as e:
            all_results.append({"question": question, "error": str(e)})
    
    return {"search_results": str(all_results)}


def analyst_node(state: State) -> dict:
    llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")
    response = llm.invoke([
        SystemMessage(content="You are a research analyst. Synthesize the search results into clear findings."),
        HumanMessage(content=f"Research plan:\n{state.get('research_plan', '')}\n\nSearch results:\n{state.get('search_results', '')}")
    ])
    return {"final_report": response.content}

def writer_node(state: State) -> dict:
    llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")
    response = llm.invoke([
        SystemMessage(content="""You are a professional research writer. 
        Take the analyst's findings and write a polished, well-structured report with:
        - An executive summary (2-3 sentences)
        - Clear sections with headings
        - Key statistics and facts highlighted
        - Cited sources where available
        - A conclusion with key takeaways
        Use markdown formatting."""),
        HumanMessage(content=f"Analyst findings:\n{state.get('final_report', '')}\n\nSearch results (for citations):\n{state.get('search_results', '')}")
        ])
    return {"final_report": response.content}


# ✅ What it should be
builder = StateGraph(State)
builder.add_node("planner", planner_node)
builder.add_node("search", search_node)
builder.add_node("analyst", analyst_node)
builder.add_node("writer", writer_node)
builder.add_edge(START, "planner")
builder.add_edge("planner", "search")
builder.add_edge("search", "analyst")
builder.add_edge("analyst", "writer")
builder.add_edge("writer", END)
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)



app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js runs here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResearchRequest(BaseModel):
    topic: str

class ResearchResponse(BaseModel):
    research_plan: str
    final_report: str

@app.get("/")
def root():
    return {"status": "Research Agent API is running"}

@app.post("/research", response_model=ResearchResponse)
def run_research(request: ResearchRequest):
    config = {"configurable": {"thread_id": f"research-{request.topic[:20]}"}}
    
    initial_state = {
        "messages": [HumanMessage(content=request.topic)],
        "research_plan": "",
        "search_results": "",
        "final_report": ""
    }
    
    result = graph.invoke(initial_state, config=config)
    
    return ResearchResponse(
        research_plan=result["research_plan"],
        final_report=result["final_report"]
    )