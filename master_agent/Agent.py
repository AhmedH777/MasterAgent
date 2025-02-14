from master_agent.LLM import LLM

class Agent:
    def __init__(self,
                 model="gpt-4-turbo",
                 max_memory_context_buffer=10,
                 role="assistant",
                 description="You are a helpful AI assistant."):
        """
        Initialize the Agent class with the given parameters.
        
        :param model: The model to use for generating responses.
        """
        self.model = model
        self.llm = LLM(model=model,
                       model_provider=self.__find_model_provider(),
                       max_memory_context_buffer=max_memory_context_buffer,
                       role=role,
                       description=description)
      
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
    agent = Agent(model="gpt-4-turbo",
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