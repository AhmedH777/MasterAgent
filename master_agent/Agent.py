import json
import importlib
from master_agent.LLM import LLM

class Agent:
    def __init__(self,
                 model="gpt-4o",
                 max_memory_size=30,
                 summary_trigger=10,
                 preserve_last_n_context=4,
                 role = "assistant",
                 description="You are a helpful AI assistant.",
                 logger=None):
        """
        Initialize the Agent class with the given parameters.
        
        :param model: The model to use for generating responses.
        """
        self.logger = logger
        self.logger_name = "AGENT"
        self.model = model
        self.llm = LLM(model=model,
                       model_provider=self.__find_model_provider(),
                       max_memory_size=max_memory_size,
                       summary_trigger=summary_trigger,
                       preserve_last_n_context=preserve_last_n_context,
                       role=role,                        
                       description=self.__generate_tool_description(description),
                       logger=logger)
        if self.logger:
            text = f"Agent initialized with description:{self.llm.description}"
            self.logger.info(f"[{self.logger_name}] {text}")

    def chat(self, user_input):
        """
        Handle a chat interaction with the user.
        
        :param user_input: The input message from the user.
        :return: The response from the model.
        """
        return self.llm.chat(user_input)
    
    def reset(self):
        """
        Reset the conversation history.
        """
        self.llm.reset()
        self.STM.reset()
    
    def set_model(self, model):
        """
        Set the model to use for generating responses.
        
        :param model: The model to use.
        """
        self.model = model
        self.llm.set_model(model, self.__find_model_provider())

    def get_model(self):
        """
        Get the current model being used.
        
        :return: The current model.
        """
        return self.model
    
    def end_chat(self):
        """
        End the chat session and save the conversation memory.
        """
        self.llm.end_chat()

    def __generate_tool_description(self, description):
        """Dynamically loads tool descriptions from tools_config.json and tool classes."""
        tools_description = []
        tools_config_path = "master_agent/tools/tools_config.json"  # Adjust path as needed

        try:
            # Load tool configurations from JSON
            with open(tools_config_path, "r") as file:
                config = json.load(file)

            for tool in config.get("tools", []):
                module_name = tool["module"]  # e.g., "master_agent.tools.PythonExecutionTool"
                class_name = tool["class"]  # e.g., "PythonExecutionTool"

                # Dynamically import the tool class
                module = importlib.import_module(module_name)
                tool_class = getattr(module, class_name)

                tools_description.append(f"🔹 **{class_name}** - {tool_class.description}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading tool descriptions: {e}")

        return (
                description +  # Base AI assistant description
                ".You have access to the following tools: " +  
                "\t".join(tools_description) +  # Dynamically generated tool list
                "Use these tools when necessary to assist the user efficiently."
                )

    def __find_model_provider(self):
        """
        Find the model provider based on the model name.
        
        :return: The model provider.
        """
        if "gpt" in self.model:
            return "openai"
        elif "llama" in self.model or "deepseek" in self.model:
            return "ollama"
        else:
            return None

"""
# Unit Test
if __name__ == "__main__":
    agent = Agent(model="gpt-4o",
              max_memory_context_buffer=2,            
              role="assistant",
              description="You are a helpful AI assistant.")
    
    counter = 0
    while True:
        user_input = input("You: ")
        response = agent.chat(user_input)
        print("Bot:", response)
        counter += 1
        if counter == 3:
            counter = 0
            agent.llm.STM.display_memory()
"""