# TODOs
# Add More crews
# Add crew selection based on prompt
# Allow for crew model selection
# Feed the crew feedback to master agent
# Make the master agent define the crew input needs (pydantic)

import os
import re
import sys
import json
import click
import ollama
import openai
from crewai import Crew, LLM
from typing import Dict, List
from dotenv import load_dotenv
from .crews.CustomerServiceCrew.crew import CustomerServiceCrew

#Define Available Models
local_models = ["llama3.2", "deepseek-r1"]
openai_models = ["gpt-4-turbo", "gpt-4o"]
############################################################################################################
######################################### Ollama and OpenAI API Keys #######################################
############################################################################################################
# Set up Ollama accelerator
os.environ["OLLAMA_ACCELERATE"] = "true"

# Load OpenAI API Key
load_dotenv() 
openai.api_key = os.getenv('API_KEY')

############################################################################################################
######################################### Chat Response Functions ##########################################
############################################################################################################
def parse_message(user_message, model="llama3.2"):
    match = re.search(r"<crew>(.*?)</crew>", user_message)
    if match:
        crew_name = match.group(1).strip()
        cleaned_prompt = re.sub(r"<crew>.*?</crew>", "", user_message).strip()  # Remove the tag from the prompt
        llm = LLM(model="gpt-4-turbo", api_key=os.getenv('API_KEY'), temperature=0)
        input = {"input" : user_message}
        return run_crew_tool(crew=CustomerServiceCrew(llm=llm).crew(), input= input)
    else:
        return respond(user_message, model=model)  # No crew tag found, return response

def respond(user_message, model="llama3.2"):
    # Generate response
    bot_reply = "Failed to generate response" + model
    if model in openai_models:
        # Use OpenAI API
        client = openai.OpenAI(api_key= os.getenv('API_KEY'))
        response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_message}]
            )
        bot_reply = response.choices[0].message.content
    elif model in local_models:
        # Use Ollama (local LLM)
        response = ollama.chat(model=model, messages=[{"role": "user", "content": user_message}])
        bot_reply = response["message"]["content"]
    return bot_reply
############################################################################################################
def run_crew_tool(crew: Crew, input: str, **kwargs):
    """
    Runs the crew using crew.kickoff(inputs=kwargs) and returns the output.

    Args:
        crew (Crew): The crew instance to run.
        messages (List[Dict[str, str]]): The chat messages up to this point.
        **kwargs: The inputs collected from the user.

    Returns:
        str: The output from the crew's execution.

    Raises:
        SystemExit: Exits the chat if an error occurs during crew execution.
    """
    try:
        # Run the crew with the provided inputs
        crew_output = crew.kickoff(inputs=input)

        # Convert CrewOutput to a string to send back to the user
        result = str(crew_output)

        return result
    except Exception as e:
        # Exit the chat and show the error message
        click.secho("An error occurred while running the crew:", fg="red")
        click.secho(str(e), fg="red")
        sys.exit(1)
