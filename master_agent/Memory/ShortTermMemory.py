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
    def __init__(self, context_buffer_counter=10, presistent_memory=""):
        super(ShortTermMemory, self).__init__(categories = categories, similarity_threshold=0.6, merge_threshold=0.6, min_cluster_words = 15)

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
            for context in self.context_buffer:
                sentences.extend(context["content"].split("."))

            # Add sentences to Memory
            rejected_sentences = self.add_sentences(sentences)
            print("Rejected Sentences: ", rejected_sentences)
            # Add rejected sentences back to the context buffer
            self.context_buffer.clear()
            for sentence in rejected_sentences:
                # only add sentences that are not empty or with valid length
                if sentence and len(sentence) > 5:
                    self.context_buffer.append({"role": "system", "content": sentence})

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