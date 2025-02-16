import os
import re
import nltk
import openai
import numpy as np
from dotenv import load_dotenv
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load OpenAI API Key
load_dotenv() 
openai.api_key = os.getenv('API_KEY')
############################################################################################################################
################################################## Summarizer Class ########################################################
############################################################################################################################
# The Summarizer class is used to generate summaries of text data. It can be used to generate one-sentence summaries, long-form summaries, titles, and classify text into categories.
class Summarizer:
    def __init__(self, use_openai=True, openai_model="gpt-4o", local_model="t5-large"):
        self.use_openai = use_openai
        self.openai_model = openai_model
        self.local_summarizer = pipeline("text2text-generation", model=local_model) if not use_openai else None

        if self.use_openai:
            self.client = openai.OpenAI(api_key=openai.api_key)  # Create OpenAI client
    # 1️⃣ Generate a one-sentence summary
    def summarize_one_sentence(self, text, user_input, max_length=100):
        prompt = f"The text was generated given User input: {user_input}\nSummarize the following text in one single sentence only as plain string:\n{text}"
        return self._generate_response(prompt, max_length)

    # 2️⃣ Generate a long-form summary
    def summarize_long(self, text, user_input, max_length=300):
        prompt = f"The text was generated given User input: {user_input}\nSummarize the following text concisely and with enough detail to capture all the information only as plain string:\n{text}"
        return self._generate_response(prompt, max_length)

    # 3️⃣ Generate a title for the text
    def generate_title(self, text, user_input, max_length=60):
        prompt = f"The text was generated given User input: {user_input}\nGenerate a title for the following text only as plain string:\n{text}"
        return self._generate_response(prompt, max_length)

    # 4️⃣ Classify the text into categories
    def classify_category(self, text, user_input, max_length=20):
        prompt = (f"The text was generated given User input: {user_input}\n"
                  f"Categorize the following text into a generic category only as plain string:\n{text}")
        return self._generate_response(prompt, max_length)

    # 🔧 Internal method to handle both OpenAI and local models
    def _generate_response(self, prompt, max_length):
        if self.use_openai:
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "system", "content": prompt}],
                max_tokens=max_length
            )
            return response.choices[0].message.content.strip()
        else:
            return self.local_summarizer(f"summarize: {prompt}", max_length=max_length)[0]['generated_text']

############################################################################################################################
################################################## Embedder Class ##########################################################
############################################################################################################################
# The Embedder class is used to generate embeddings for text data and calculate similarity between texts.
class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Initialize the embedding model
        self.model = SentenceTransformer(model_name)

    # Generate an embedding for a given text
    def embed_chunk(self, text):
        return self.model.encode(text)

    def embed_chunk_sentences(self, text):
        sentences = text.split(".")
        # Encode each sentence
        embeddings = []
        for sentence in sentences:
            embeddings.append(self.model.encode(sentence))
        # Average the sentence embeddings
        mean_embedding = np.mean(embeddings, axis=0)
        return mean_embedding
    
    # Calculate similarity between two texts
    def similarity(self, text1, text2):
        emb1 = self.embed_chunk(text1)
        emb2 = self.embed_chunk(text2)
        return cosine_similarity([emb1], [emb2])[0][0]

    def similarity_sentences(self, text1, text2):
        emb1 = self.embed_chunk_sentences(text1)
        emb2 = self.embed_chunk_sentences(text2)
        return cosine_similarity([emb1], [emb2])[0][0]
############################################################################################################################
################################################## TextProcessor Class ######################################################
############################################################################################################################
# The TextProcessor class is used to process text data by splitting it into sentences and cleaning up Markdown formatting.
class TextProcessor:
    def __init__(self):
        pass

    # 1️⃣ Split text into sentences using NLTK
    def split_sentences(self, text):
        return nltk.sent_tokenize(text)

    # 2️⃣ Clean Markdown formatting
    def clean_markdown(self, text):
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)   # Bold -> Plain
        text = re.sub(r'\*(.*?)\*', r'\1', text)       # Italics -> Plain
        text = re.sub(r'`(.*?)`', r'\1', text)         # Inline code -> Plain
        text = re.sub(r'```[\s\S]*?```', '', text)     # Code blocks -> Remove
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)    # Images -> Remove
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)     # Links -> Remove
        text = re.sub(r'###\s(.*?)\n', r'\1\n', text)  # Headings -> Sentences
        text = re.sub(r'\n+', ' ', text)               # Newlines -> Spaces
        text = re.sub(r'\s{2,}', ' ', text)            # Remove extra spaces
        text = re.sub(r'[-*•]\s+(.*?)\n', r'\1. ', text) # Convert lists to sentences
        return text.strip()

    # 3️⃣ Process text: clean Markdown + split sentences
    def process_input(self, text):
        cleaned = self.clean_markdown(text)
        return self.split_sentences(cleaned)

'''
# Unit Test
if __name__ == "__main__":

    # Input List of Text
    input_list =  [{'role': 'system', 'content': 'You are a helpful AI assistant.'},
                   {'role': 'user', 'content': 'tell me about sensor fusion'},
                   {'role': 'assistant', 'content': "**Sensor fusion** is the process of combining sensory data from multiple sources in order to make more accurate and comprehensive assessments of the environment than could be achieved by using any single source alone. The principle behind sensor fusion is that different sensors have complementary strengths and limitations; by intelligently merging their data, the strengths of one can offset the weaknesses of others.\n\n### Applications of Sensor Fusion\nSensor fusion is used in a wide range of applications, including:\n\n1. **Autonomous Vehicles**: Combining data from LIDAR, radar, cameras, and ultrasonic sensors to detect obstacles, navigate, and understand the vehicle’s surroundings.\n2. **Robotics**: Robots use sensor fusion to map environments, avoid obstacles, and interact safely with humans and other objects.\n3. **Smartphones**: Modern devices use sensor fusion for applications such as orientation and motion detection, combining accelerometers, gyroscopes, and magnetometers.\n4. **Healthcare**: In medical monitoring devices, data from various sensors monitoring different physiological parameters can be fused to gain a comprehensive understanding of a patient's state.\n5. **Aerospace and Defense**: Sensor fusion enhances the capabilities of systems in aircraft, drones, and satellites, integrating data for navigation, target tracking, and system health monitoring.\n\n### Techniques\nThe techniques used in sensor fusion can be broadly classified into several levels:\n\n1. **Data Level Fusion**: Directly combines raw data from different sources to produce new raw data which is presumably more accurate.\n2. **Feature Level Fusion**: Involves extracting features from multiple data sources (like shapes or textures in images, or key frequencies in sound) and then combining these features.\n3. **Decision Level Fusion**: At this level, each sensor system makes an independent decision, and these are then combined into a final decision using voting systems, weighted decisions, etc.\n\n### Methods and Algorithms\nVarious mathematical and computational methods are employed in sensor fusion, including:\n\n- **Kalman Filters**: Used extensively in applications like tracking and navigation, especially where noise and uncertainty are involved.\n- **Bayesian Networks**: Useful for reasoning under uncertainty, particularly in decision level fusion.\n- **Neural Networks**: Particularly deep learning methods have become prominent for feature-level fusion by learning to combine features in an end-to-end manner.\n- **Fuzzy Logic**: Helps in dealing with imprecision and reasoning about uncertain or vague data.\n\n### Challenges \nSome of the challenges in sensor fusion include:\n\n- **Data Association**: Correctly associating data from different sensors to the same object or event in the environment.\n- **Synchronization**: Ensuring data from different sources is temporally aligned.\n- **Scalability**: Managing the increased computational load as the number of sensors increases.\n- **Conflict Resolution**: Handling contradictory information from different sensors.\n\n### Conclusion\nSensor fusion is a crucial component in modern technology systems, enabling smarter, more reliable, and efficient operations. Its development involves not just advancements in sensor technology but also significant innovations in areas like artificial intelligence and machine learning to process and integrate complex and diverse data streams. As technology advances, the use of sensor fusion is likely to become more ubiquitous across industries, further enhancing machine perception and decision-making capabilities."}, {'role': 'user', 'content': 'what about machine learning?'}, {'role': 'assistant', 'content': 'Machine learning is a branch of artificial intelligence (AI) focused on building applications that learn from data and improve their accuracy over time without human intervention. It involves developing algorithms that can process input data and use statistical analysis to predict an output value within an acceptable range. The core idea is to automatically identify patterns in data and make decisions with minimal human direction.\n\n### Key Concepts in Machine Learning\n\n1. **Supervised Learning**: This involves learning from a labeled dataset, allowing the model to predict outcomes for unseen data. Examples are regression and classification tasks. The training data include both the input and the desired output.\n\n2. **Unsupervised Learning**: In unsupervised learning, the data used to train the model are not labeled, meaning that the model tries to find patterns and relationships in the data. Common tasks include clustering and association.\n\n3. **Reinforcement Learning**: This type simulates a scenario where an agent learns to make decisions by performing certain actions and receiving rewards or penalties. It’s widely used in areas such as gaming, navigation, and real-time decisions.\n\n4. **Semi-supervised Learning**: This approach uses both labeled and unlabeled data for training, usually a small amount of labeled data with a large amount of unlabeled data. This can be useful when acquiring labels is expensive or laborious.\n\n5. **Deep Learning**: A subset of machine learning that uses architectures made of layers of nodes, or neurons, inspired by the human brain (neural networks). Deep learning is particularly powerful for tasks like image and speech recognition.\n\n### Common Algorithms\n\n- **Linear Regression**\n- **Logistic Regression**\n- **Decision Trees**\n- **Support Vector Machines (SVM)**\n- **K-nearest Neighbors (K-NN)**\n- **Random Forests**\n- **Gradient Boosting algorithms**: Like XGBoost and LightGBM\n- **Neural Networks**\n\n### Applications of Machine Learning\n\n1. **Healthcare**: From diagnostic applications to drug discovery and personalized medicine.\n2. **Finance**: For algorithmic trading, credit scoring, fraud detection.\n3. **E-commerce**: Recommendation systems that suggest products to users.\n4. **Autonomous Vehicles**: Including navigation systems and driverless cars.\n5. **Marketing**: Customer segmentation, sales forecasting, and sentiment analysis.\n6. **Robotics**: For robotic sensors, processing data inputs, and action decisions.\n7. **Natural Language Processing (NLP)**: For tasks such as translation, sentiment analysis, and information extraction.\n\n### Advantages\n\n- **Efficiency**: Automates decision-making and can process large sets of data quickly.\n- **Predictive Power**: Provides businesses and researchers the ability to predict certain outcomes based on historical data.\n- **Personalization**: Enables personalized services and products which can lead to better user experiences.\n\n### Challenges\n\n- **Data Quality**: Requires a large amount of quality data; poor quality data can lead to misleading results.\n- **Overfitting and Underfitting**: Models might perform well on training data but poorly on unseen data.\n- **Bias**: Machine learning models can inadvertently become biased if biased data is used in training.\n- **Computational Costs**: Some algorithms, particularly in deep learning, require significant computational resources.\n\n### Conclusion\n\nMachine learning is a rapidly evolving field leading to new technologies, business models, and avenues of research. It is a cornerstone of modern AI applications, influencing a multitude of industries and improving automation, decision-making processes, and user interaction across the tech landscape. As machine learning evolves, it pushes the boundaries of what machines can do in ways that emulate and sometimes exceed human capabilities, particularly in processing speed and efficiency.'}, 
                   {'role': 'user', 'content': 'which one is better for the future?'}, 
                   {'role': 'assistant', 'content': 'Choosing which technology—sensor fusion or machine learning—is "better" for the future isn\'t straightforward, as both technologies serve different purposes and are often complementary. The evaluation largely depends on specific use cases, industries, and the type of problems being solved. Here are some considerations for each and how they might be perceived for future applications:\n\n### Sensor Fusion\n**Applicability**: Sensor fusion is crucial in fields where multiple sensor inputs are necessary to create accurate real-time assessments of the environment. This is especially relevant in robotics, autonomous vehicles, and IoT devices.\n\n**Future Outlook**: As the world becomes more connected and devices more autonomous, the role of sensor fusion is set to increase significantly. Technologies requiring high reliability and precision in dynamic environments, such as automated and flying cars, will especially benefit from advanced sensor fusion.\n\n**Limitations and Enhancements**: The main limitations revolve around data handling, processing power, and sensor compatibility. Future advancements in hardware and real-time data processing algorithms can address these issues.\n\n### Machine Learning\n**Applicability**: Machine learning has a broader scope of application compared to sensor fusion, impacting virtually every industry from healthcare to finance, with capabilities ranging from predictive analytics to autonomous decision-making systems.\n\n**Future Outlook**: Machine learning, and particularly deep learning, continues to advance rapidly. Its ability to adapt, learn from vast amounts of data, and improve over time makes it integral to AI development. Its role in data analytics and automation is critical and growing as computational power and data collection capabilities expand.\n\n**Limitations and Enhancements**: Current limitations include issues like data privacy, ethical concerns, and the need for large datasets. Advances in AI safety, privacy-preserving machine learning (such as federated learning), and more efficient algorithms are likely to mitigate these issues.\n\n### Synergistic Use\nIn many cutting-edge applications, the convergence of sensor fusion and machine learning provides exceptional benefits. For instance:\n\n- In autonomous vehicles, sensor fusion integrates data from various sensors to understand the environment accurately, and machine learning algorithms interpret this data to make driving decisions.\n- In robotics, sensor fusion provides accurate environmental data which machine learning algorithms use to guide movements and interactions.\n\n### Conclusion\nConclusively, neither technology is "better" outright; instead, they are both vital and often complementary. The choice of which technology to emphasize depends on specific industry goals and technological needs. \n\nFor industries focused on physical interaction with the environment, such as automotive or robotics, sensor fusion might be more immediately critical. Conversely, for domains emphasizing data interpretation, prediction, and automation, like finance or healthcare, machine learning could be more transformative.\n\nThus, it\'s not just a matter of choosing one over the other but rather understanding how each can best be used or integrated to enhance capabilities and solve problems. The future is likely to lean heavily on both, leveraging their strengths to enable smarter, safer, and more efficient systems.'}, 
                   {'role': 'user', 'content': 'thanks'}]
    
    # Initialize the Summarizer
    summarizer = Summarizer(use_openai=True)

    # Initialize the Embedder
    embedder = Embedder()

    # Initialize the TextProcessor
    processor = TextProcessor()

    # Sample text for processing
    text = input_list[4]['content']

    # Process the text
    processed_text = processor.process_input(text)
    print("Processed Text:")
    for sentence in processed_text:
        print(sentence)

    # Generate a one-sentence summary
    one_sentence_summary = summarizer.summarize_one_sentence(text, "Summarize this text")
    print("\nOne-Sentence Summary:", one_sentence_summary)

    # Generate a long-form summary
    long_summary = summarizer.summarize_long(text, "Summarize this text")
    print("\nLong Summary:", long_summary)

    # Generate a title for the text
    title = summarizer.generate_title(text, "Generate a title for this text")
    print("\nTitle:", title)

    # Classify the text into categories
    category = summarizer.classify_category(text, "Categorize this text")
    print("\nCategory:", category)
    
    # Generate embeddings for the text
    #embeddings = embedder.embed(text)
    #print("\nEmbeddings:", embeddings)

    # Calculate similarity between two texts
    text1 = "This is a sample text for similarity calculation."
    text2 = "This is another sample text for similarity calculation."
    similarity = embedder.similarity(text1, text2)
    print("\nSimilarity:", similarity)  
'''

