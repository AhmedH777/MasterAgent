import json
import hdbscan
import numpy as np
import networkx as nx
from typing import List, Dict
from pyvis.network import Network
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from master_agent.Memory.MemoryBlock import MemoryBlock

########################################################################################################################
################################################### Memory class #######################################################
########################################################################################################################
class GenericMemory:
    """Agent's memory that dynamically clusters, merges, and links knowledge."""
    
    def __init__(self, categories: List[str], similarity_threshold=0.6, merge_threshold=0.6, min_cluster_words = 15, logger=None):
        """
        - categories: List of category labels to classify Memory Blocks
        - similarity_threshold: Minimum similarity to consider two sentences related
        - merge_threshold: Minimum similarity to merge two Memory Blocks
        """
        self.logger = logger
        self.logger_name = "MEMORY"
        self.categories = categories
        self.similarity_threshold = similarity_threshold
        self.merge_threshold = merge_threshold
        self.min_cluster_words = min_cluster_words
        self.memory = []  # Stores active Memory Blocks objects
        self.links = {}  # Stores relationships between Memory Blocks
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')  # Model for embeddings

    def add_sentences(self, sentences: List[str]):
        """Clusters new sentences and adds them to memory as new or existing Memory Blocks."""
        # Placeholder for rejected sentences (too short)
        rejected_sentences = []

        # Cluster sentences using HDBSCAN
        clustered_sentences = self.__cluster_sentences(sentences)

        # Convert clusters to Cluster objects
        for cluster_id, cluster_sentences in clustered_sentences.items():
            # Create clusters with more than min_cluster_words
            if self.__is_valid_cluster(cluster_sentences):
                new_mem_block = MemoryBlock(cluster_sentences, self.categories, logger=self.logger)

                # Check if this cluster is similar to an existing one
                is_merged = self.__check_and_merge_new_mem_blocks(new_mem_block)

                if(not is_merged):
                    # Add the cluster if it's new
                    self.memory.append(new_mem_block)
                    if self.logger is not None:
                        message = f"✅ Added New Memory Block: {new_mem_block}"
                        self.logger.info(f"[{self.logger_name}] {message}")
            else:
                rejected_sentences.extend(cluster_sentences)

        return rejected_sentences

    def query_memory(self, query: str, top_k=3):
        """Finds the most relevant mem_blocks for a given query."""
        query_embedding = self.embedding_model.encode([query])[0]
        
        scores = []
        for mem_block in self.memory:
            similarity = cosine_similarity([query_embedding], [mem_block.embedding])[0][0]
            scores.append((mem_block, similarity))
        
        # Sort by highest similarity
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]
    
    def get_memblocks_summary(self):
        """Returns a summary of all memblocks in memory."""
        return [mem_block.summary for mem_block in self.memory]

    def optimize_memory(self):
        """ Optimizing memory by merging highly similar mem_blocks."""
        for i, mem_block_a in enumerate(self.memory):
            for j, mem_block_b in enumerate(self.memory):
                if i != j:
                    similarity = cosine_similarity([mem_block_a.embedding], [mem_block_b.embedding])[0][0]
                    if similarity >= self.merge_threshold:
                        mem_block_a.update_summary(mem_block_b.summary)
                        self.memory.remove(mem_block_b)
                        if self.logger is not None:
                            message = f"🔄 Merging Memory Block: {mem_block_a} with Memory Block: {mem_block_b} (Similarity: {similarity:.2f})"
                            self.logger.info(f"[{self.logger_name}] {message}")

        if self.logger is not None:
            message = f"GenericMem : 🧹 Memory optimized."
            self.logger.info(f"[{self.logger_name}] {message}")
    
    def save_memory(self):
        """Saves memory mem_blocks to a JSON file."""
        with open(self.memory_file, "w") as f:
            json.dump([{"summary": c.summary, "category": c.category, "confidence": c.confidence} for c in self.memory], f)
            print("💾 Memory saved.")

    def load_memory(self):
        """Loads memory clusters from a JSON file."""
        try:
            with open(self.memory_file, "r") as f:
                data = json.load(f)
                self.memory = [MemoryBlock([d["summary"]], self.categories, logger=self.logger) for d in data]
                print("📂 Memory loaded.")
        except FileNotFoundError:
                print("⚠ No previous memory found.")

    def reset(self):
        """Resets memory by clearing all mem_blocks and links."""
        self.memory.clear()
        self.links.clear()
        if self.logger is not None:
            message = f"GenericMem : 🗑 Memory reset."
            self.logger.info(f"[{self.logger_name}] {message}")

    def create_links(self):
        """Finds and stores relationships between mem_blocks based on similarity."""
        self.links.clear()
        for i, mem_block_a in enumerate(self.memory):
            for j, mem_block_b in enumerate(self.memory):
                if i != j:
                    similarity = cosine_similarity([mem_block_a.embedding], [mem_block_b.embedding])[0][0]
                    
                    if similarity >= self.similarity_threshold:
                        if mem_block_a not in self.links:
                            self.links[mem_block_a] = []
                        self.links[mem_block_a].append((mem_block_b, similarity))
        if self.logger is not None:
            message = f"GenericMem : 🔗 Links between mem_blocks have been updated."
            self.logger.info(f"[{self.logger_name}] {message}")

    def display_memory_logger(self):
        """Displays the current mem_blocks in memory."""
        if self.logger is not None:
            message = f"GenericMem : 📌 Current Memory:"
            self.logger.info(f"[{self.logger_name}] {message}")
            for mem_block in self.memory:
                self.logger.info(f"[{self.logger_name}] {mem_block}")

    def display_memory(self):
        """Displays the current mem_blocks in memory."""
        print("\n📌 Current Memory:")
        for mem_block in self.memory:
            print(mem_block)

    def display_links(self):
        """Displays relationships between mem_blocks."""
        print("\n🔗 Relationships between Memory Blocks:")
        for mem_block, related_mem_blocks in self.links.items():
            print(f"\n{mem_block.summary} is linked to:")
            for related, sim in related_mem_blocks:
                print(f"  ➡ {related.summary} (Similarity: {sim:.2f})")


    def visualize_links(self, output_file="knowledge_graph.html"):
        """Visualizes knowledge links using NetworkX and Pyvis."""
        if not self.links:
            if self.logger is not None:
                message = f"GenericMem : ⚠ No links to visualize."
                self.logger.info(f"[{self.logger_name}] {message}")
            return
        
        G = nx.Graph()

        # Add nodes
        for mem_block in self.memory:
            G.add_node(mem_block.category)  # Use shortened summary as node label

        # Add edges (relationships)
        for mem_block, related_mem_blocks in self.links.items():
            for related, sim in related_mem_blocks:
                G.add_edge(
                    mem_block.category, 
                    related.category, 
                    weight=float(sim) * 5,  # ✅ Ensure similarity is a native Python float
                    title=f"Similarity: {sim:.2f}")

        # Create Pyvis network visualization
        net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
        net.from_nx(G)

        # Save and open the graph
        net.write_html(output_file)  
        if self.logger is not None:
            message = f"GenericMem : 📌 Knowledge Graph saved as {output_file}"
            self.logger.info(f"[{self.logger_name}] {message}")

        # Open the graph in the browser
        import webbrowser
        webbrowser.open(output_file)


    def __is_valid_cluster(self, cluster_sentences: List[str]) -> bool:
        """Checks if a cluster is valid based on the number of words."""
        return len(" ".join(cluster_sentences).split()) >= self.min_cluster_words
    
    def __cluster_sentences(self, sentences: List[str]) -> Dict[int, List[str]]:
        """Clusters sentences using HDBSCAN with cosine similarity."""
        embeddings = self.embedding_model.encode(sentences)
        cosine_distances = (1 - cosine_similarity(embeddings)).astype(np.float64)

        clusterer = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=2, min_samples=1, cluster_selection_method="eom")
        labels = clusterer.fit_predict(cosine_distances)

        clusters_dict = {}
        for i, label in enumerate(labels):
            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append(sentences[i])

        return clusters_dict

    def __check_and_merge_new_mem_blocks(self, new_memory_block: MemoryBlock) -> bool:
        """Merges new clusters if they are similar to existing memory block."""
        for memory_block in self.memory:
            similarity = cosine_similarity([memory_block.embedding], [new_memory_block.embedding])[0][0]
            if similarity >= self.merge_threshold:
                memory_block.update_summary(new_memory_block.summary)
                if self.logger is not None:
                    message = f"GenericMem : 🔄 Merging MemBlock: {memory_block} with New MemBlock (Similarity: {similarity:.2f})"
                    self.logger.info(f"[{self.logger_name}] {message}")
                return True
        return False


'''
# Test the Generic Memory
if __name__ == "__main__":
    # Define paragraph for testing
    paragraph = "As artificial intelligence (AI) continues to reshape industries, its role in healthcare is becoming increasingly vital, with AI-powered diagnostics and robotic-assisted surgeries enhancing medical precision. This progress is deeply linked to big data and cloud computing, which store vast amounts of patient records, helping doctors make data-driven decisions. Meanwhile, technological advancements are also influencing the financial sector, with blockchain providing secure transaction methods and AI-driven analytics optimizing investment strategies. The rise of cryptocurrency, a blockchain innovation, is challenging traditional banking, creating debates on regulations and financial stability.At the same time, climate change remains a critical global issue, with nations seeking innovative solutions to reduce carbon footprints. AI is playing a role here as well, analyzing climate patterns and optimizing renewable energy sources like solar and wind power. Governments and corporations are investing in sustainable energy, with electric vehicles (EVs) gaining traction as an eco-friendly alternative. However, the production of lithium-ion batteries for EVs raises concerns about mining practices and resource depletion, linking environmental sustainability to ethical and economic discussions.Space exploration is another field witnessing rapid advancements, with NASA and private companies such as SpaceX and Blue Origin developing reusable rockets for deep-space missions. AI and robotics are essential in these projects, assisting in autonomous navigation and data analysis. Space missions also drive technological innovation on Earth, from satellite-based climate monitoring to communication networks that enhance global internet coverage, further connecting AI, environmental monitoring, and financial markets.The transformation in education is another crucial development, as online learning platforms powered by AI-driven algorithms personalize learning experiences. This shift is crucial for remote areas where internet access, supported by satellite technology, is bridging the digital divide. However, excessive screen time and reliance on technology raise mental health concerns, as studies indicate increased rates of anxiety and depression due to social media exposure and digital overuse.With these rapid changes, ethical debates surrounding data privacy, AI biases, and automation-driven job displacement are becoming more prevalent. Governments and institutions are working on policies to balance innovation with societal well-being, ensuring that technological progress benefits humanity without exacerbating inequalities.As artificial intelligence (AI) continues to revolutionize industries, its impact on healthcare is profound, with AI-powered diagnostics, robotic surgeries, and personalized treatment plans improving patient outcomes. These advancements are closely tied to big data and cloud computing, which enable real-time access to patient histories and predictive analytics, enhancing medical precision. However, concerns about data privacy and the ethical implications of AI-driven decisions are growing, prompting governments to establish regulatory frameworks to protect sensitive health information.Meanwhile, blockchain technology, initially developed for cryptocurrency, is now being integrated into healthcare systems to create immutable patient records, ensuring data security and reducing fraud. The financial sector is also undergoing massive changes, with AI-driven trading algorithms and blockchain-based decentralized finance (DeFi) disrupting traditional banking structures. The rise of cryptocurrencies challenges regulatory institutions, leading to ongoing debates about financial stability, digital asset taxation, and the risks of market manipulation. As fintech evolves, so does the intersection of AI and cybersecurity, as AI-driven fraud detection systems aim to prevent financial crimes but also raise concerns about potential biases in automated decision-making.At the same time, climate change remains a pressing issue, with AI being deployed to optimize renewable energy sources like solar, wind, and hydroelectric power. Advanced climate models powered by machine learning are helping scientists predict extreme weather events more accurately, providing crucial data for disaster preparedness. The global shift toward sustainable energy is also accelerating the adoption of electric vehicles (EVs), with companies investing in battery innovations to extend lifespan and reduce reliance on scarce resources like lithium and cobalt. However, the mining practices associated with these materials have sparked debates on environmental responsibility and ethical sourcing.Space exploration is another frontier witnessing AI integration, as NASA, SpaceX, and Blue Origin leverage machine learning for autonomous spacecraft navigation, mission planning, and extraterrestrial resource exploration. Satellite technology, originally developed for space missions, now plays a critical role in climate monitoring, global internet connectivity, and military defense strategies. AI-powered geospatial analysis is being used to track deforestation, analyze urbanization patterns, and enhance agricultural productivity through precision farming techniques.The education sector is also undergoing a transformation, with AI-driven adaptive learning systems personalizing curriculum plans based on individual student needs. Online education platforms, accelerated by the pandemic, are leveraging AI to automate grading, provide real-time feedback, and enhance accessibility for students in remote areas. However, the increased screen time and dependence on digital learning tools raise concerns about mental health, particularly regarding social isolation and cognitive overload.With the rise of automation and robotics, discussions about job displacement have intensified. AI-powered systems are streamlining processes in manufacturing, logistics, and customer service, but this progress fuels debates about the future of work, the need for reskilling programs, and how societies should adapt to an AI-driven economy. Ethical concerns regarding bias in AI models, responsible AI development, and the impact of deepfake technology on misinformation are gaining global attention, prompting calls for stricter AI governance.As AI continues to push the boundaries of innovation, policymakers, businesses, and researchers must work together to strike a balance between technological advancement and ethical responsibility, ensuring that AI serves humanity without deepening social inequalities or compromising fundamental rights."
    # Split paragraph into sentences
    sentences = paragraph.split(". ")

    # Define categories for classification
    topics = [
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


    # Initialize GenericMemory
    memory = GenericMemory(topics, similarity_threshold=0.45, merge_threshold=0.8, min_cluster_words=12, debug=True)

    # Add sentences to memory
    rejected_sentences = memory.add_sentences(sentences)
    print("\nRejected Sentences:", rejected_sentences)

    # Optimize memory by merging similar clusters
    memory.optimize_memory()

    # Create relationships between clusters
    memory.create_links()

    # Display the working memory (active clusters)
    memory.display_memory()

    # Display linked clusters
    memory.display_links()

    # Visualize linked clusters
    memory.visualize_links()

    ################################################################################
    # Define categories for classification
    categories = ["Deep Learning", "Machine Learning", "AI", "Stocks", "Finance", "Technology"]

    # Initialize GenericMemory
    memory = GenericMemory(categories, similarity_threshold=0.6, merge_threshold=0.8, min_cluster_words=12, debug=True)

    # Sentences to be clustered
    sentences = [
        "Machine learning is transforming industries.",
        "Artificial intelligence improves decision-making.",
        "Deep learning helps in image recognition.",
        "AI is changing the world of business and automation.",
        "Stock market trends are unpredictable.",
        "Self-driving cars rely on machine learning algorithms.",
        "Recent government policies impact international trade."
    ]

    # Create New Test Sentences with a mix of existing and new information
    new_sentences = [
        "AI is revolutionizing the business world.",
        "The stock market is influenced by government policies.",
        "Machine learning algorithms are used in self-driving cars.",
        "Finance and technology are interconnected.",
        "Deep learning models are used in image recognition systems.",
        "Government regulations affect international trade.",
        "Stock market trends are influenced by machine learning."
    ]

    # Create New sentences with a mix of existing and new information
    new_sentences2 = [
        "Stocks are influenced by government policies.",
        "Deep learning models are used in image recognition systems.",
        "Computer vision is a subset of machine learning.",
        "Finance and technology are interconnected.",
        "Goverment support is crucial for AI research."]
    
    ###################### Iteration 1 ######################
    # Add sentences to memory
    rejected_sentences = memory.add_sentences(sentences)
    print("\nRejected Sentences:", rejected_sentences)
    ###################### Iteration 2 ######################
    # Combine New Sentences with rejected sentences
    new_sentences.extend(rejected_sentences)

    # Add new sentences to memory
    rejected_sentences = memory.add_sentences(new_sentences)
    print("\nRejected Sentences:", rejected_sentences)
    ####################### Iteration 3 ######################
    # Combine New Sentences with rejected sentences
    new_sentences2.extend(rejected_sentences)

    # Add new sentences to memory
    rejected_sentences = memory.add_sentences(new_sentences2)
    print("\nRejected Sentences:", rejected_sentences)
    print("##################################################")

    # Optimize memory by merging similar clusters
    memory.optimize_memory()

    # Create relationships between clusters
    memory.create_links()

    # Display the working memory (active clusters)
    memory.display_memory()

    # Display linked clusters
    #memory.display_links()

    # Visualize linked clusters
    memory.visualize_links()

    # Query the memory with a new sentence
    #query = "How does AI impact business operations?"
    #scores = memory.query_memory(query, top_k=3)
    '''