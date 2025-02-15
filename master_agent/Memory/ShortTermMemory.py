import re
from master_agent.Memory.GenericMemory import GenericMemory

categories = [
        "Artificial Intelligence & Machine Learning",
        "Healthcare & Medical Technology",
        "Big Data & Cloud Computing",
        "Data Privacy & AI Ethics",
        "Blockchain & Decentralized Finance (DeFi)",
        "Cryptocurrency & Financial Regulations",
        "Cybersecurity & AI Fraud Detection",
        "Climate Change & Renewable Energy",
        "Electric Vehicles & Battery Technology",
        "Space Exploration & Autonomous AI",
        "Satellite Technology & Global Connectivity",
        "Geospatial Analysis & Environmental Monitoring",
        "Education & AI-Powered Learning",
        "Mental Health & Digital Well-being",
        "Automation & Job Displacement",
        "AI Ethics & Bias in Decision-Making",
        "Deepfake Technology & Misinformation"
    ]

class ShortTermMemory(GenericMemory):
    def __init__(self, context_buffer_counter=10, presistent_memory="", logger=None):
        super(ShortTermMemory, self).__init__(categories = categories, similarity_threshold=0.6, merge_threshold=0.6, min_cluster_words = 15, logger=logger)

        # Logger
        self.logger = logger
        self.logger_name = "MEMORY"

        # Working Memory
        self.context_buffer_counter = context_buffer_counter
        self.context_buffer_count = 0
        self.context_buffer = []

        # Presistent Memory
        self.presistent_memory = presistent_memory

    def add(self, memory_input):
        # Split memory input into sentences
        self.context_buffer.append(memory_input)
        self.context_buffer_count += 1

        # If the context buffer is full, Process the buffer in Memory
        if self.context_buffer_count >= self.context_buffer_counter:
            self.context_buffer_count = 0
            
            # Split the context buffer into sentences``
            sentences = []
            # pop last context from the buffer to add it back to the context buffer
            last_context = self.context_buffer.pop()
            # Add all the other contexts to the sentences
            for context in self.context_buffer:
                content = context["content"]
                if(self.__is_markdown(content)):
                    content = self.__markdown_to_text(content)
                sentences.extend(content.split("."))

            # Add sentences to Memory
            rejected_sentences = self.add_sentences(sentences)
            # Add rejected sentences back to the context buffer
            self.context_buffer.clear()
            for sentence in rejected_sentences:
                # only add sentences that are not empty or with valid length
                if sentence and len(sentence) > 5:
                    self.context_buffer.append({"role": "system", "content": sentence})
            
            # Add the last context back to the context buffer
            self.context_buffer.append(last_context)

    def get_prompt(self):
        prompt = []
        # Always add presistent memory to the prompt
        prompt.append(self.presistent_memory)

        # Add Summary of Memory Blocks to the prompt
        memory_summary = self.get_memblocks_summary()
        for idea in memory_summary:
            prompt.append({"role": "system", "content": idea})

        # Add Context Buffer to the prompt
        for context in self.context_buffer:
            prompt.append(context)
        
        return prompt
    
    def reset(self):
        self.context_buffer_count = 0
        self.context_buffer.clear()
        self.reset()

    def __is_markdown(self,text: str) -> bool:
        # Patterns indicating common markdown syntax
        patterns = [
            r"^#{1,6}\s",                 # Headers
            r"(\*\*|__).*?\1",            # Bold
            r"(\*|_).*?\1",               # Italics
            r"!\[.*?\]\(.*?\)",           # Images
            r"\[.*?\]\(.*?\)",            # Links
            r"```[\s\S]*?```",            # Code blocks
            r"`[^`]*`",                   # Inline code
            r"^(\*|\-|\+)\s",             # Unordered list
            r"^\d+\.\s",                  # Ordered list
            r"> .+",                      # Blockquotes
        ]

        # Check if any of the patterns match the text
        return any(re.search(pattern, text, re.M) for pattern in patterns)

    def __markdown_to_text(self,md_string: str) -> str:
        # Remove code blocks
        md_string = re.sub(r"```.*?```", "", md_string, flags=re.S)
        # Remove inline code
        md_string = re.sub(r"`(.*?)`", r"\1", md_string)
        # Remove links but keep the text
        md_string = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", md_string)
        # Remove images but keep alt text
        md_string = re.sub(r"!\[([^\]]+)\]\([^\)]+\)", r"\1", md_string)
        # Remove headings
        md_string = re.sub(r"^#{1,6}\s*", "", md_string, flags=re.M)
        # Remove emphasis (bold, italic, etc.)
        md_string = re.sub(r"(\*|_){1,2}(.*?)\1", r"\2", md_string)
        # Remove blockquotes
        md_string = re.sub(r"^>\s*", "", md_string, flags=re.M)
        # Remove lists
        md_string = re.sub(r"^(\s*[-+*]|\d+\.)\s+", "", md_string, flags=re.M)
        # Remove extra spaces
        md_string = re.sub(r"\n{2,}", "\n\n", md_string)
        # Strip leading/trailing whitespace
        return md_string.strip()

