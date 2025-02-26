import os
import time
import json
import litellm
import requests
import importlib
import subprocess
from dotenv import load_dotenv
from master_agent.Memory import Memory

############################################################################################################
######################################### Ollama and OpenAI API Keys #######################################
############################################################################################################

# Set up Ollama accelerator
os.environ["OLLAMA_ACCELERATE"] = "true"

# Load OpenAI API Key from environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('API_KEY')
############################################################################################################

class LLM():
    def __init__(self,
                model="gpt-4o",
                model_provider="openai",
                max_memory_size=30,
                summary_trigger=10,
                preserve_last_n_context=4,
                role="assistant",
                description="You are a helpful AI assistant.",
                logger=None):
        """
        Initialize the LLM class with the given parameters.
        
        :param model: The model to use for generating responses.
        :param max_memory_context_buffer: The maximum number of messages to keep in history.
        :param role: The role of the assistant.
        :param description: The initial description of the assistant.
        """
        self.model = model
        self.model_provider = model_provider
        self.role = role
        self.description = description
        self.tools = self.__load_tools()  # Load tools dynamically
        self.ollama_process = None
        self.logger = logger
        self.logger_name = "LLM"

        # Add Short Term Memory
        self.memory = Memory(system_message={"role": "system", "content": self.description}, max_buffer_size=max_memory_size, summary_trigger=summary_trigger, preserve_last_n=preserve_last_n_context, logger=logger)
        
        # Start or stop the Ollama server based on the model provider
        self.__handle_ollama_server()

    def chat(self, user_input):
        """
        Handle a chat interaction with the user.
        
        :param user_input: The input message from the user.
        :return: The response from the model.
        """
        # Add the user input to the Short Term Memory
        self.memory.add_interaction("user", user_input)

        # if self.logger is not None:
        #     message = f"Prompt : {self.memory.get_context()}"
        #     self.logger.info(f"[{self.logger_name}] {message}")

        # Invoke the model to get a response
        response = self.__invoke(self.memory.get_context())

        # Extract the content of the response
        filtered_response = response["choices"][0]["message"]["content"]

        # Add the response to the Short Term Memory
        self.memory.add_interaction(self.role, filtered_response)

        return filtered_response

    def reset(self):
        """
        Reset the conversation history.
        """
        self.history = [{"role": self.role, "content": self.description}]

    def set_model(self, model, model_provider):
        """
        Set the model to use for generating responses.
        
        :param model: The model to use.
        """
        self.model = model
        self.model_provider = model_provider

        # Start or stop the Ollama server based on the model provider
        self.__handle_ollama_server()
    
    def end_chat(self):
        """
        End the chat session and save the conversation memory.
        """
        self.memory.end_chat()

    def __load_tools(self):
        """Load tools dynamically from `tools_config.json`."""
        with open("master_agent/tools/tools_config.json", "r") as file:
            config = json.load(file)

        tool_instances = {}
        for tool in config["tools"]:
            try:
                module = importlib.import_module(tool["module"])
                tool_class = getattr(module, tool["class"])
                tool_instances[tool["name"]] = tool_class()
            except ModuleNotFoundError as e:
                if self.logger is not None:
                    message = f"Error loading {tool['name']}: {e}"
                    self.logger.info(f"[{self.logger_name}] {message}")
                print(message)
        
        return tool_instances
    
    def __invoke(self, prompt):
        """
        Invoke the model to generate a response based on the given prompt.
        
        :param prompt: The prompt to send to the model.
        :return: The response from the model.
        """
        response = "Failed to generate response, model not found"
        if "gpt" in self.model:
            # if self.logger is not None:
            #     message = [tool.as_dict() for tool in self.tools.values()]
            #     self.logger.info(f"[{self.logger_name}] {message}")
            response = litellm.completion(
                        model= self.model,
                        messages=prompt,
                        tools=[tool.as_dict() for tool in self.tools.values()]
                    )
            '''
            if self.logger is not None:
                message = response
                self.logger.info(f"[{self.logger_name}] {message}")
            '''
        elif "llama" in self.model:
            response = litellm.completion(
                        model= "ollama/" + str(self.model),
                        messages=prompt,
                        api_base="http://localhost:11434"
                    )
        
        elif "deepseek" in self.model:
            response = litellm.completion(
                        model= "ollama/" + str(self.model),
                        messages=prompt,
                        api_base="http://localhost:11434"
                    )
        '''
        if self.logger is not None:
            message = "gpt" in self.model
            self.logger.info(f"[{self.logger_name}] {message}")
            message = "choices" in response
            self.logger.info(f"[{self.logger_name}] {message}") 
            message = response["choices"][0].message.tool_calls is not None
            self.logger.info(f"[{self.logger_name}] {message}") 
        '''

        # Check if the model requested a tool call
        current_prompt = prompt
        if "gpt" in self.model:
            while("choices" in response and response["choices"][0].message.tool_calls is not None):
                tool_calls = response["choices"][0].message.tool_calls  # ✅ Access `tool_calls` properly
                '''
                if self.logger is not None:
                    message = tool_calls
                    self.logger.info(f"[{self.logger_name}] {message}")
                '''
                for tool_call in tool_calls:
                    function_name = tool_call.function.name.lower()
                    arguments = json.loads(tool_call.function.arguments)
                    '''
                    if self.logger is not None:
                        message = str(function_name) + " " + str(arguments) + " " + str(self.tools)
                        self.logger.info(f"[{self.logger_name}] {message}")
                    '''
                    if function_name in {name.lower(): tool for name, tool in self.tools.items()}:
                        tool = self.tools[[key for key in self.tools.keys() if key.lower() == function_name][0]]
                        result = tool.run(arguments)
                        '''
                        if self.logger is not None:
                            message = result
                            self.logger.info(f"[{self.logger_name}] {message}")
                        '''
                        if not isinstance(result, str):
                            result = json.dumps(result)  # ✅ Convert result to string if needed


                        # Ensure OpenAI gets a properly formatted tool response
                        current_prompt += [
                            {"role": "function", "name": function_name, "content": result}
                        ]

                        '''
                        if self.logger is not None:
                            message = current_prompt
                            self.logger.info(f"[{self.logger_name}] {message}")
                        '''
                        response = litellm.completion(
                        model=self.model,
                        messages=current_prompt,
                        tools=[tool.as_dict() for tool in self.tools.values()])
                        '''
                        if self.logger is not None:
                            message = response
                            self.logger.info(f"[{self.logger_name}] {message}")
                        '''
        return response

    # Function to check if ollama is running
    def __is_ollama_running(self):
        try:
            response = requests.get("http://localhost:11434")
            return response.status_code == 200
        except requests.ConnectionError:
            return False
        
    # Function to Handle ollama server run or stop
    def __handle_ollama_server(self):
        if self.model_provider == "ollama":
            if not self.__is_ollama_running():
                self.ollama_process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                time.sleep(5)  # Give some time for the server to start
        
        # Shut down Ollama server if provider is not "ollama"
        elif self.__is_ollama_running() and self.ollama_process is not None:
            self.ollama_process.terminate()

"""
# Unit Test
if __name__ == "__main__":
    # Create an instance of the LLM class with specific parameters
    llm = LLM(keep_history=True, max_history=5, summarize_history=True)
    
    # Run an interactive chat loop
    counter = 0
    while True:
        user_input = input("You: ")
        print("Bot: ", llm.chat(user_input))
        

        counter += 1
        print("Counter: ", counter)
        if(counter == 2):
            llm.set_model(model="llama3.2", model_provider="ollama")
"""