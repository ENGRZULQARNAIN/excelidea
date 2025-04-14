import yaml
import os
import json

# Import relevant functionality
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# Import things that are needed generically
from pydantic import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.prompts import PromptTemplate
load_dotenv()
import uuid
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

prompt = """
You are an AI assistant designed to help undergraduate students in technology fields, such as Software Engineering and Computer Science, find ideas for their final year projects by providing tailored problem statements. Your task is to engage in a conversation with the student, collect their inputs, use a websearch tool to gather relevant information, and iteratively refine a problem statement that suits their profile.

  instructions:
    - |
      Collect student inputs: The student will provide details such as areas of interest, skills and expertise, preferred project type, real-world problems or inspirations, project duration, available resources, and collaboration context. Use these to customize the problem statement.
    - |
      Use the websearch tool to enhance the problem statement by finding recent trends, researching applications, discovering tools, and drawing inspiration from existing projects. For example, search for recent machine learning applications or current challenges in online learning.
    - |
      Engage in conversation: Ask follow-up questions to clarify preferences, provide suggestions based on websearch and inputs, and iteratively refine the problem statement.
    - |
      Generate the problem statement: Present a clear and concise statement including the problem, technologies, expected outcomes, and constraints. Ensure it is original and feasible for an undergraduate project.
    - |
      Be helpful and encouraging: Maintain a friendly tone and suggest trending topics if the student lacks ideas.

  example_interaction:
    - role: student
      message: "I'm interested in web development and have skills in JavaScript and React. I have 6 months for my project."
    - role: agent
      message: "Awesome! Web development offers tons of possibilities. Let me search for some trends… Progressive web apps (PWAs) are hot right now. How about building a PWA for real-time collaboration, like a shared task manager? You could use React for the frontend and WebSockets for real-time features. Sound good?"
    - role: student
      message: "Yes, but I'd like it to relate to education."
    - role: agent
      message: "Got it! Let me refine that… How about a PWA for students to collaborate on study notes in real-time? You could add features like version control for notes or AI-driven study suggestions. What do you think?"

  guidelines:
    - Use the websearch tool judiciously to provide current and relevant information.
    - Tailor suggestions to the student's undergraduate skill level, ensuring projects are challenging yet achievable.
"""


def load_prompt_string(file_path='prompts.yaml'):
    """
    Load the prompt string from a YAML file.
    
    Args:
        file_path (str): Path to the YAML file relative to the current directory
        
    Returns:
        str: The prompt string from the YAML file, or None if not found
    """
    try:
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the full path to the YAML file
        full_path = os.path.join(current_dir, file_path)
        
        # Check if the file exists
        if not os.path.exists(full_path):
            error_msg = f"Prompts file not found at: {full_path}"
            raise FileNotFoundError(error_msg)
        
        # Load the YAML file
        with open(full_path, 'r', encoding='utf-8') as file:
            yaml_content = yaml.safe_load(file)
            
        # Extract the prompt string from the task section
        if yaml_content and 'task' in yaml_content and 'prompt' in yaml_content['task']:
            return yaml_content['task']['prompt']
        else:
            print("Prompt string not found in the YAML file")
            return None
            
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None
    except Exception as e:
        print(f"Error loading prompt: {e}")
        return None

from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
model_1 = ChatOpenAI(model="gpt-4o-mini",temperature=0.2)

class PropertyData(BaseModel):
    """Order form data model."""
    pb_id: str = Field(description="The unique id of the contents of the problem statement")
    content: str = Field(description="The contents of the problem statement")



parser = PydanticOutputParser(pydantic_object=PropertyData)
fixing_parser = OutputFixingParser.from_llm(llm=model_1, parser=parser)
extraction_prompt = PromptTemplate(
    template="Extract structured property data from the given input which is a problem statement and the id of the problem statement but note that sometime the id is not given you must leave it empty:\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions":parser.get_format_instructions()},
)
prompt_and_model = extraction_prompt | model_1
        
@tool(return_direct=True)
def store_results(new_problem_statement: str):
    """
    it will call when the user want to store the problem statement in the json file or want to confirm the problem statement
    """
    print("Calling store_results")
    file_path = "./users_ids.json"
    pb_id = str(uuid.uuid4())[:8]
    new_problem_statement = {pb_id: new_problem_statement}
    # Create the file with default structure if it doesn't exist
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump({"problem_statements": []}, file, indent=4)
    
    # Load the JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Add the new problem statement to the list
    if 'problem_statements' in data:
        data['problem_statements'].append(new_problem_statement)
    else:
        data['problem_statements'] = [new_problem_statement]
    
    # Save the updated data back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)
    
    print(f"Successfully added: {new_problem_statement}")
    return "Problem statement stored successfully with id: "+pb_id


model = ChatOpenAI(model_name="gpt-4o-mini")
search = TavilySearchResults(max_results=2)
tools = [search, store_results]
agent_executor = create_react_agent(model, tools)

def serialized_history(history):
    history_list = []

    for message in history:
        if isinstance(message, dict):
            # Handle dictionary format
            if message["role"] == "human":
                history_list.append(HumanMessage(content=message["content"]))
            elif message["role"] == "ai":
                history_list.append(AIMessage(content=message["content"]["content"]))
        else:
            # Handle Pydantic Message objects
            if message.role == "human":
                history_list.append(HumanMessage(content=message.content))
            elif message.role == "ai":
                history_list.append(AIMessage(content=message.content["content"]))
    
    return history_list




def get_problem_statement(user_id: str):
    """
    Get the most recent problem statement
    """
    file_path = "./users_ids.json"
    
    # Check if file exists, if not return a default message
    if not os.path.exists(file_path):
        return "No problem statements available yet."
    
    # Load the JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Check if there are any problem statements
    if not data.get('problem_statements'):
        return "No problem statements available yet."
    
    # Get the most recent problem statement
    problem_statements = data['problem_statements'][-1]
    return problem_statements


def problem_conversation(history: list):
    history_list = serialized_history(history)
    history_list.insert(0, SystemMessage(content=prompt))
    response = agent_executor.invoke({"messages": history_list})
    
    output = prompt_and_model.invoke({"query": response["messages"][-1].content})
    output = fixing_parser.invoke(output)
    try:
        output = json.loads(output)
    except:
        output = output
    print("output", output)
    history.append({"role": "ai", "content": output})
    return history
    
    
# response = agent_executor.invoke({"messages": [HumanMessage(content="whats the weather in Islamabad? and")]})

