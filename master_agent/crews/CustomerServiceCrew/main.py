#!/usr/bin/env python
import sys
import warnings

from crew import CustomerServiceCrew
from crewai import LLM
import os

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

####################################################################################################
##################################### Load Model Parameters ########################################
####################################################################################################
# importing necessary functions from dotenv library
from dotenv import load_dotenv 
# loading variables from .env file
load_dotenv() 

# accessing and printing value
model = os.getenv('MODEL')
api_key = os.getenv('API_KEY')

# Define Key as Environment Variable
os.environ["OPENAI_API_KEY"] = api_key

# Load LLM Model
llm = LLM(model=model, api_key=api_key, temperature=0)

####################################################################################################
# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run(inputs=None):
    """
    Run the crew.
    """  
    try:
        CustomerServiceCrew(llm=llm).crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train(inputs = None, n_iterations = 1, filename = "training.pkl"):
    """
    Train the crew for a given number of iterations.
    """
    try:
        CustomerServiceCrew(llm=llm).crew().train(n_iterations=n_iterations, filename=filename, inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay(task_id = None):
    """
    Replay the crew execution from a specific task.
    """
    try:
        CustomerServiceCrew(llm=llm).crew().replay(task_id=task_id)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test(inputs = None, n_iterations = 1, openai_model_name = "gpt-4o"):
    """
    Test the crew execution and returns the results.
    """
    try:
        CustomerServiceCrew(llm=llm).crew().test(n_iterations=n_iterations, openai_model_name=openai_model_name, inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


if __name__ == "__main__":
    mode = input("Please select mode [run, train, replay, test]: ")

    inputs = {
        "programming_language": "Python",
        "requirements": "I want a To-Do List App. A user can create a to-do list where tasks can be added, viewed, completed, and deleted.Tasks should have:A description (e.g., Buy groceries).A status (e.g., Incomplete or Complete).Interaction should happen through a command-line interface.The program saves tasks to a file so they persist between sessions."
    }
    
    if mode == "run":
        run(inputs)
    elif mode == "train":
        train(inputs, n_iterations=1, filename="training.pkl")
    elif mode == "replay":
        replay(inputs)
    elif mode == "test":
        test(inputs, n_iterations=1, openai_model_name="gpt-4o")
    else:
        raise Exception("Invalid mode. Please select from [run, train, replay, test].")