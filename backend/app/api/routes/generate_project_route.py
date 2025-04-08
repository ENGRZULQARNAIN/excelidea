# backend/app/api/routes/generate_project_route.py

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, TypedDict, Annotated
import logging
import uuid
import os
from dotenv import load_dotenv

# Langchain & LangGraph specific imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_community.tools import DuckDuckGoSearchResults # Alternative search tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI # Or any other ChatModel like ChatGoogleGenerativeAI

# --- Environment Setup ---
load_dotenv()
# Ensure necessary API keys are set in your .env file (e.g., OPENAI_API_KEY, TAVILY_API_KEY)

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- API Router Setup ---
router = APIRouter(tags=["Project Proposal Generation"])

# --- Pydantic Models for API Request/Response ---
class ProjectInput(BaseModel):
    field: str = Field(..., description="The general field of the project (e.g., 'Artificial Intelligence', 'Renewable Energy').")
    domain: str = Field(..., description="The specific domain within the field (e.g., 'Natural Language Processing', 'Solar Panel Efficiency').")
    idea: Optional[str] = Field(None, description="An initial idea or specific focus for the project (optional).")
    # Add config options if needed, e.g., target audience, specific constraints

class ProposalResponse(BaseModel):
    proposal_id: str
    status: str
    message: str
    result: Optional[Dict[str, Any]] = None

# --- LangGraph State Definition ---
class ProposalGraphState(TypedDict):
    """
    Represents the state of the project proposal generation graph.
    """
    # Inputs
    field: str
    domain: str
    idea: Optional[str]

    # Node outputs / Intermediate results
    problem_statement: Optional[str]
    key_challenges: Optional[List[str]]
    search_queries: Optional[List[str]]
    search_results: Optional[List[Dict[str, str]]] # Store title, snippet, url
    literature_summary: Optional[str]
    research_gaps: Optional[List[str]]
    methodology_approach: Optional[str]
    methodology_tools: Optional[List[str]]
    validation_method: Optional[str]
    expected_metrics: Optional[List[str]]
    expected_impact: Optional[str]
    references: Optional[List[Dict[str, str]]] # Store citation-like info

    # Final Output
    formatted_proposal: Optional[str]

    # Control Flow & Error Handling
    error: Optional[str] = None
    node_history: List[str] = Field(default_factory=list) # Track node execution

# --- Tool Definition ---
# Using Tavily for search as it's often preferred in LangChain examples
# Requires TAVILY_API_KEY environment variable
search_tool = TavilySearchResults(max_results=5)
# Alternatively, use DuckDuckGo:
# search_tool = DuckDuckGoSearchResults()

# --- LLM Definition ---
# Ensure OPENAI_API_KEY is set in environment
# You can swap this with other models like ChatGoogleGenerativeAI, ChatAnthropic, etc.
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

# --- Node Functions ---

async def identify_problem(state: ProposalGraphState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Node to identify and articulate the core problem based on field, domain, and optional idea.
    """
    logger.info("--- Executing Node: identify_problem ---")
    field = state["field"]
    domain = state["domain"]
    idea = state.get("idea")
    state["node_history"].append("identify_problem")

    prompt_text = """You are an expert research analyst specializing in identifying impactful problems.
Given the field '{field}' and domain '{domain}', please perform the following:
1.  **Problem Statement:** Clearly articulate a significant and specific problem within this area. Make it concise but comprehensive (target 50-100 words).
2.  **Key Challenges:** List 3-5 distinct key challenges associated with addressing this problem. These should be technical, practical, or theoretical hurdles.

Consider this initial user idea if provided: '{idea}'

Format your response as a JSON object with keys 'problem_statement' (string) and 'key_challenges' (list of strings).
Example JSON:
{{
  "problem_statement": "Existing methods for X in Y suffer from low accuracy when dealing with Z, leading to significant inefficiencies...",
  "key_challenges": ["Challenge 1 description", "Challenge 2 description", "Challenge 3 description"]
}}"""

    if not idea:
        prompt_text = prompt_text.replace("Consider this initial user idea if provided: '{idea}'", "")
    else:
        prompt_text = prompt_text.format(field=field, domain=domain, idea=idea)

    prompt = ChatPromptTemplate.from_messages([("system", prompt_text)])
    chain = prompt | llm | JsonOutputParser()

    try:
        response = await chain.ainvoke({}, config=config)
        if not isinstance(response, dict) or "problem_statement" not in response or "key_challenges" not in response:
             raise ValueError("LLM response did not contain the expected JSON structure.")
        logger.info(f"Problem Identified: {response['problem_statement']}")
        return {
            "problem_statement": response["problem_statement"],
            "key_challenges": response["key_challenges"],
            "error": None
        }
    except Exception as e:
        logger.error(f"Error in identify_problem: {str(e)}")
        return {"error": f"Failed to identify problem: {str(e)}"}

async def conduct_literature_review(state: ProposalGraphState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Node to perform literature search, summarize findings, and identify research gaps.
    """
    logger.info("--- Executing Node: conduct_literature_review ---")
    problem = state.get("problem_statement")
    challenges = state.get("key_challenges", [])
    field = state["field"]
    domain = state["domain"]
    state["node_history"].append("conduct_literature_review")


    if not problem:
        return {"error": "Problem statement is missing for literature review."}

    # 1. Generate Search Queries
    query_prompt = ChatPromptTemplate.from_template(
        "Generate 3 diverse and effective search queries for a literature review on the problem: '{problem}' within the field '{field}' and domain '{domain}'. Focus on recent work, solutions, and challenges: {challenges}. Output as a JSON list of strings."
    )
    query_chain = query_prompt | llm | JsonOutputParser()
    try:
        search_queries = await query_chain.ainvoke({"problem": problem, "field": field, "domain": domain, "challenges": ", ".join(challenges)}, config=config)
        if not isinstance(search_queries, list):
            search_queries = [f"{field} {domain} problem solutions", f"{problem} recent research"]
            logger.warning("Failed to generate queries via LLM, using default queries.")
        logger.info(f"Generated Search Queries: {search_queries}")
    except Exception as e:
        logger.error(f"Error generating search queries: {e}. Using default queries.")
        search_queries = [f"{field} {domain} problem solutions", f"{problem} recent research"]

    # 2. Execute Search
    all_results = []
    try:
        for query in search_queries[:3]: # Limit queries
            logger.info(f"Running search for: {query}")
            # Tavily Tool expects a list of dicts, each with 'content' key
            # DuckDuckGo expects a string query
            if isinstance(search_tool, TavilySearchResults):
                 # Tavily sometimes returns string, sometimes list. Standardize.
                raw_search_output = await search_tool.ainvoke(query, config=config)
                if isinstance(raw_search_output, str): # Handle plain string error/message
                     logger.warning(f"Tavily returned string for '{query}': {raw_search_output}")
                     continue
                if isinstance(raw_search_output, list):
                    all_results.extend(raw_search_output)
                else:
                     logger.warning(f"Unexpected Tavily output type for '{query}': {type(raw_search_output)}")

            # elif isinstance(search_tool, DuckDuckGoSearchResults):
            #     # DDG returns a string, need to parse it or use it directly
            #     ddg_results_str = await search_tool.arun(query, config=config)
            #     # Basic parsing example (you might need more robust parsing)
            #     items = ddg_results_str.split("}, {")
            #     for item in items[:3]: # Limit results per query
            #         try:
            #             title_match = re.search(r"title': '(.*?)'", item)
            #             snippet_match = re.search(r"snippet': '(.*?)'", item)
            #             link_match = re.search(r"link': '(.*?)'", item)
            #             if title_match and snippet_match and link_match:
            #                 all_results.append({"title": title_match.group(1), "content": snippet_match.group(1), "url": link_match.group(1)})
            #         except Exception:
            #             logger.warning(f"Could not parse DDG result item: {item}")
            else:
                 raise NotImplementedError(f"Search tool type {type(search_tool)} not fully supported in this node.")

        # Clean up results (simple example, could involve more sophisticated filtering/ranking)
        # Tavily often gives 'url' and 'content'
        cleaned_results = [{"title": r.get('title', 'N/A'), "content": r.get('content', ''), "url": r.get('url', '')} for r in all_results if r.get('content')]
        logger.info(f"Found {len(cleaned_results)} relevant search results.")
        if not cleaned_results:
             logger.warning("No relevant search results found.")
             # Decide how to proceed: error out, or try to generate gaps without results?
             # For now, let's proceed but flag it.
             # return {"error": "No relevant search results found during literature review."}


    except Exception as e:
        logger.error(f"Error during search execution: {str(e)}")
        return {"error": f"Failed during literature search: {str(e)}"}

    # 3. Summarize and Identify Gaps using LLM
    gap_prompt_text = """You are a research synthesis expert following IEEE standards.
Based on the identified problem statement, key challenges, and the following search results, identify 3-5 key research gaps.

**Problem Statement:**
{problem}

**Key Challenges:**
{challenges}

**Search Results (Summaries/Snippets):**
{search_snippets}

**Task:**
1.  Briefly summarize the current state-of-the-art or existing approaches based ONLY on the provided search results.
2.  Clearly list 3-5 specific research gaps that remain, considering the problem and challenges. These gaps should represent areas where further investigation is needed.

Format your response as a JSON object with keys 'literature_summary' (string) and 'research_gaps' (list of strings).
Example JSON:
{{
  "literature_summary": "Current research focuses on A and B, with promising results in C. However, limitations exist in handling scenario Z...",
  "research_gaps": ["Gap 1: Need for improved methodology for X.", "Gap 2: Lack of data for Y.", "Gap 3: Scalability issues of approach Z."]
}}"""

    search_snippets_str = "\n\n".join([f"Title: {r['title']}\nSnippet: {r['content'][:300]}...\nURL: {r['url']}" for r in cleaned_results[:10]]) # Limit context for LLM

    gap_prompt = ChatPromptTemplate.from_messages([("system", gap_prompt_text)])
    gap_chain = gap_prompt | llm | JsonOutputParser()

    try:
        gap_response = await gap_chain.ainvoke({
            "problem": problem,
            "challenges": ", ".join(challenges),
            "search_snippets": search_snippets_str if cleaned_results else "No search results were found."
        }, config=config)

        if not isinstance(gap_response, dict) or "literature_summary" not in gap_response or "research_gaps" not in gap_response:
             raise ValueError("LLM response for gaps did not contain the expected JSON structure.")

        logger.info(f"Identified Research Gaps: {gap_response['research_gaps']}")

        # Prepare references from search results
        references = [{"title": r.get('title', 'N/A'), "url": r.get('url', '')} for r in cleaned_results[:5]] # Take top 5 for references

        return {
            "search_queries": search_queries,
            "search_results": cleaned_results, # Store the raw results
            "literature_summary": gap_response["literature_summary"],
            "research_gaps": gap_response["research_gaps"],
            "references": references,
            "error": None
        }
    except Exception as e:
        logger.error(f"Error identifying research gaps: {str(e)}")
        return {"error": f"Failed to identify research gaps: {str(e)}"}

async def design_methodology(state: ProposalGraphState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Node to propose a methodology addressing the identified gaps.
    """
    logger.info("--- Executing Node: design_methodology ---")
    problem = state.get("problem_statement")
    gaps = state.get("research_gaps")
    state["node_history"].append("design_methodology")


    if not problem or not gaps:
        return {"error": "Problem statement or research gaps missing for methodology design."}

    prompt_text = """You are a senior researcher designing a project methodology.
Based on the problem statement and identified research gaps, propose a methodology.

**Problem Statement:**
{problem}

**Identified Research Gaps:**
{gaps}

**Task:**
Propose a clear methodology that directly addresses at least one major research gap. Specify:
1.  **Approach:** Describe the overall research approach (e.g., experimental, simulation-based, theoretical, design-based).
2.  **Key Steps/Techniques:** Outline the main steps or specific techniques to be used.
3.  **Tools/Technologies:** List key software, hardware, datasets, or libraries required.
4.  **Validation Method:** Briefly describe how the results or proposed solution will be validated (e.g., comparison with baseline, user studies, statistical analysis).

Format your response as a JSON object with keys 'approach' (string), 'techniques' (list of strings), 'tools' (list of strings), and 'validation_method' (string).
Example JSON:
{{
  "approach": "An experimental approach comparing algorithm A and a novel algorithm B.",
  "techniques": ["Develop algorithm B based on theory X", "Implement both algorithms", "Conduct experiments on benchmark dataset Y", "Analyze performance metrics"],
  "tools": ["Python", "PyTorch", "Benchmark Dataset Y", "Specific Hardware Z"],
  "validation_method": "Statistical comparison of performance metrics (accuracy, speed) against baseline algorithm A and state-of-the-art results."
}}"""

    prompt = ChatPromptTemplate.from_messages([("system", prompt_text)])
    # Using StrOutputParser first to see raw output, then trying JSON
    # chain = prompt | llm | StrOutputParser() # For debugging
    chain = prompt | llm | JsonOutputParser()

    try:
        # raw_response = await chain.ainvoke({"problem": problem, "gaps": "\n- ".join(gaps)}, config=config) # Debugging
        # print("RAW METHODOLOGY:", raw_response) # Debugging
        response = await chain.ainvoke({"problem": problem, "gaps": "\n- ".join(gaps)}, config=config)

        if not isinstance(response, dict) or "approach" not in response or "tools" not in response or "validation_method" not in response:
             raise ValueError("LLM response for methodology did not contain the expected JSON structure.")


        logger.info(f"Designed Methodology Approach: {response['approach']}")
        return {
            "methodology_approach": response["approach"],
            "methodology_tools": response.get("tools", []), # Use .get for safety
            "validation_method": response["validation_method"],
            # Store techniques if needed, depends on formatting stage
            "error": None
        }
    except Exception as e:
        logger.error(f"Error in design_methodology: {str(e)}")
        return {"error": f"Failed to design methodology: {str(e)}"}

async def project_results_and_impact(state: ProposalGraphState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Node to outline expected results and their potential impact.
    """
    logger.info("--- Executing Node: project_results_and_impact ---")
    methodology = state.get("methodology_approach")
    problem = state.get("problem_statement")
    gaps = state.get("research_gaps", [])
    state["node_history"].append("project_results_and_impact")


    if not methodology or not problem:
        return {"error": "Methodology or problem statement missing for projecting results."}

    prompt_text = """You are projecting the outcomes of a research project.
Based on the problem, the gaps being addressed, and the proposed methodology, outline the expected results and impact.

**Problem Statement:**
{problem}

**Research Gaps Addressed:**
{gaps}

**Proposed Methodology:**
{methodology}

**Task:**
1.  **Expected Metrics:** List 3-5 specific, measurable metrics that will be used to evaluate the success of the project based on the methodology and validation plan.
2.  **Expected Impact:** Describe the potential impact of achieving the expected results. Consider scientific contributions, practical applications, or broader societal benefits.

Format your response as a JSON object with keys 'expected_metrics' (list of strings) and 'expected_impact' (string).
Example JSON:
{{
  "expected_metrics": ["Metric 1 (e.g., Accuracy improvement > 10%)", "Metric 2 (e.g., Reduction in processing time by 20%)", "Metric 3 (User satisfaction score > 4.0/5.0)"],
  "expected_impact": "Successful completion will advance the field of Y by providing a more robust method for Z. It could lead to practical applications in industry Q, potentially saving costs or improving efficiency..."
}}"""

    prompt = ChatPromptTemplate.from_messages([("system", prompt_text)])
    chain = prompt | llm | JsonOutputParser()

    try:
        response = await chain.ainvoke({
            "problem": problem,
            "gaps": "\n- ".join(gaps),
            "methodology": methodology
        }, config=config)

        if not isinstance(response, dict) or "expected_metrics" not in response or "expected_impact" not in response:
             raise ValueError("LLM response for results/impact did not contain the expected JSON structure.")

        logger.info(f"Projected Impact: {response['expected_impact']}")
        return {
            "expected_metrics": response["expected_metrics"],
            "expected_impact": response["expected_impact"],
            "error": None
        }
    except Exception as e:
        logger.error(f"Error in project_results_and_impact: {str(e)}")
        return {"error": f"Failed to project results and impact: {str(e)}"}

async def format_proposal_ieee(state: ProposalGraphState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Node to compile all generated information into an IEEE-style proposal document.
    """
    logger.info("--- Executing Node: format_proposal_ieee ---")
    state["node_history"].append("format_proposal_ieee")

    # Check for necessary components
    required_keys = [
        "problem_statement", "research_gaps", "methodology_approach",
        "methodology_tools", "validation_method", "expected_metrics",
        "expected_impact", "references"
    ]
    missing_keys = [key for key in required_keys if not state.get(key)]
    if missing_keys:
        logger.error(f"Cannot format proposal, missing data for: {', '.join(missing_keys)}")
        return {"error": f"Cannot format proposal, missing data for: {', '.join(missing_keys)}"}

    # Prepare content for the LLM formatter
    content_to_format = f"""
    **1. Introduction / Problem Statement:**
    {state['problem_statement']}

    **2. Literature Review & Research Gaps:**
    Summary: {state.get('literature_summary', 'N/A')}
    Key Identified Gaps:
    - {"\n- ".join(state['research_gaps'])}

    **3. Proposed Methodology:**
    Overall Approach: {state['methodology_approach']}
    Key Tools/Technologies: {", ".join(state['methodology_tools'])}
    Validation Strategy: {state['validation_method']}

    **4. Expected Results and Impact:**
    Evaluation Metrics: {", ".join(state['expected_metrics'])}
    Potential Impact: {state['expected_impact']}

    **5. Preliminary References:**
    {chr(10).join([f"[{i+1}] {ref.get('title', 'N/A')} ({ref.get('url', 'No URL')})" for i, ref in enumerate(state.get('references', []))])}
    """ # Using chr(10) for newline within f-string for references

    prompt_text = f"""You are an expert technical writer specializing in formatting research proposals according to IEEE standards.
Compile the following draft sections into a coherent and professionally formatted project proposal document. Ensure clear headings, logical flow, and academic tone suitable for an IEEE submission.

**DRAFT CONTENT:**
{content_to_format}

**TASK:**
Format the above content into a single, well-structured proposal document. Use standard IEEE section titles (e.g., I. INTRODUCTION, II. RELATED WORK / LITERATURE REVIEW, III. METHODOLOGY, IV. EXPECTED RESULTS AND DISCUSSION, V. REFERENCES). Elaborate slightly where needed for flow, but stick closely to the provided content. Ensure the references section is formatted correctly. Output only the formatted proposal text, starting with 'IEEE PROJECT PROPOSAL'.
"""

    prompt = ChatPromptTemplate.from_messages([("system", prompt_text)])
    chain = prompt | llm | StrOutputParser()

    try:
        formatted_proposal = await chain.ainvoke({}, config=config)
        logger.info("Proposal formatted successfully.")
        return {
            "formatted_proposal": formatted_proposal,
            "error": None
        }
    except Exception as e:
        logger.error(f"Error in format_proposal_ieee: {str(e)}")
        return {"error": f"Failed to format proposal: {str(e)}"}


# --- Graph Conditional Logic ---

def check_for_errors(state: ProposalGraphState) -> str:
    """
    Checks if an error occurred in the previous node.
    """
    if state.get("error"):
        logger.error(f"Graph execution stopped due to error: {state['error']}")
        return "error_occurred"
    else:
        # Determine the last executed node to route correctly
        last_node = state.get("node_history", [])[-1] if state.get("node_history") else None
        if last_node == "identify_problem":
            return "continue_to_literature"
        elif last_node == "conduct_literature_review":
            return "continue_to_methodology"
        elif last_node == "design_methodology":
            return "continue_to_results"
        elif last_node == "project_results_and_impact":
            return "continue_to_formatting"
        else:
            # Default or if something unexpected happens
            logger.warning(f"Unexpected state in check_for_errors after node: {last_node}")
            return "end_processing" # Or route to error

# --- Workflow Construction ---

workflow = StateGraph(ProposalGraphState)

# Add nodes
workflow.add_node("identify_problem", identify_problem)
workflow.add_node("conduct_literature_review", conduct_literature_review)
workflow.add_node("design_methodology", design_methodology)
workflow.add_node("project_results_and_impact", project_results_and_impact)
workflow.add_node("format_proposal_ieee", format_proposal_ieee)

# Define edges and entry point
workflow.set_entry_point("identify_problem")

# Conditional edges after each main step
workflow.add_conditional_edges(
    "identify_problem",
    check_for_errors,
    {
        "continue_to_literature": "conduct_literature_review",
        "error_occurred": END
    }
)
workflow.add_conditional_edges(
    "conduct_literature_review",
    check_for_errors,
    {
        "continue_to_methodology": "design_methodology",
        "error_occurred": END
    }
)
workflow.add_conditional_edges(
    "design_methodology",
    check_for_errors,
    {
        "continue_to_results": "project_results_and_impact",
        "error_occurred": END
    }
)
workflow.add_conditional_edges(
    "project_results_and_impact",
    check_for_errors,
    {
        "continue_to_formatting": "format_proposal_ieee",
        "error_occurred": END
    }
)

# Final edge
workflow.add_edge("format_proposal_ieee", END)

# Compile the graph
proposal_generator_app = workflow.compile()

# --- API Endpoint Implementation ---

# In-memory storage for POC (Replace with DB/Cache for production)
proposal_jobs: Dict[str, Dict[str, Any]] = {}

async def run_proposal_graph(job_id: str, initial_state: ProposalGraphState):
    """Runs the graph asynchronously and updates the job status."""
    proposal_jobs[job_id]["status"] = "running"
    try:
        # Configuration for the run, can include thread_id for persistence if needed
        config = {"configurable": {"thread_id": f"proposal-thread-{job_id}"}}
        # stream events for potential real-time updates (though we await the final here)
        final_state = await proposal_generator_app.ainvoke(initial_state, config=config)

        if final_state.get("error"):
             proposal_jobs[job_id]["status"] = "failed"
             proposal_jobs[job_id]["message"] = f"Proposal generation failed: {final_state['error']}"
             proposal_jobs[job_id]["result"] = final_state # Include error details
        else:
            proposal_jobs[job_id]["status"] = "completed"
            proposal_jobs[job_id]["message"] = "Proposal generated successfully."
            # Only include relevant final parts in the result if needed
            proposal_jobs[job_id]["result"] = {
                "formatted_proposal": final_state.get("formatted_proposal"),
                "problem_statement": final_state.get("problem_statement"),
                "research_gaps": final_state.get("research_gaps"),
                 "node_history": final_state.get("node_history", [])
                # Add other parts if desired in the final status check
            }
            logger.info(f"Job {job_id} completed successfully.")

    except Exception as e:
        logger.error(f"Unhandled exception during graph execution for job {job_id}: {e}", exc_info=True)
        proposal_jobs[job_id]["status"] = "failed"
        proposal_jobs[job_id]["message"] = f"An unexpected error occurred: {str(e)}"
        proposal_jobs[job_id]["result"] = {"error": str(e)}


@router.post("/generate_proposal", status_code=202, response_model=ProposalResponse)
async def create_proposal_job(
    background_tasks: BackgroundTasks,
    body: ProjectInput
):
    """
    Asynchronously starts the project proposal generation process.
    Returns a job ID to check the status later.
    """
    job_id = str(uuid.uuid4())
    logger.info(f"Received proposal request for field='{body.field}', domain='{body.domain}'. Job ID: {job_id}")

    # Initialize state dictionary from input
    initial_state = ProposalGraphState(
        field=body.field,
        domain=body.domain,
        idea=body.idea,
        # Initialize other fields to None or empty lists
        problem_statement=None,
        key_challenges=[],
        search_queries=[],
        search_results=[],
        literature_summary=None,
        research_gaps=[],
        methodology_approach=None,
        methodology_tools=[],
        validation_method=None,
        expected_metrics=[],
        expected_impact=None,
        references=[],
        formatted_proposal=None,
        error=None,
        node_history=[]
    )

    # Store initial job info
    proposal_jobs[job_id] = {
        "proposal_id": job_id,
        "status": "pending",
        "message": "Proposal generation initiated.",
        "result": None
    }

    # Run the graph in the background
    background_tasks.add_task(run_proposal_graph, job_id, initial_state)

    return ProposalResponse(
        proposal_id=job_id,
        status="pending",
        message="Proposal generation started. Check status using the job ID."
    )

@router.get("/generate_proposal/status/{job_id}", response_model=ProposalResponse)
async def get_proposal_status(job_id: str):
    """
    Checks the status and result of a proposal generation job.
    """
    job_info = proposal_jobs.get(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail="Job ID not found.")

    return ProposalResponse(**job_info)
