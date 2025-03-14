import os
from typing import List, Dict
import json
from model import JenkinsNLPModel
from scraper import JenkinsDocScraper

class JenkinsKnowledgeBase:
    def __init__(self):
        self.docs_path = "backend/data/docs"
        self.raw_path = "backend/data/raw"
        self.model = JenkinsNLPModel()
        self.embeddings = {}
        self.documents = {}
        self.initialize()

    def initialize(self):
        """Initialize the knowledge base and create necessary directories"""
        os.makedirs(self.docs_path, exist_ok=True)
        os.makedirs(self.raw_path, exist_ok=True)
        self.load_documents()
        print("Knowledge base initialized")

    def load_documents(self):
        """Load and process all documents from the raw directory"""
        if not os.path.exists(self.raw_path):
            return

        for filename in os.listdir(self.raw_path):
            if filename.endswith('.json'):
                with open(os.path.join(self.raw_path, filename), 'r') as f:
                    doc = json.load(f)
                    source = filename.replace('.json', '')
                    self.process_documentation(doc['content'], source)

    def process_documentation(self, text: str, source: str):
        """Process and embed documentation text"""
        # Generate embeddings for the text
        embedding = self.model.encode_text(text)
        
        processed = {
            "text": text,
            "embedding": embedding.tolist()
        }
        
        # Store processed document
        self.documents[source] = text
        self.embeddings[source] = embedding
        
        # Save to disk
        with open(f"{self.docs_path}/{source}.json", "w") as f:
            json.dump(processed, f)

    def query(self, question: str) -> str:
        """Query the knowledge base"""
        if not self.embeddings:
            return "No documentation has been loaded yet. Please run the scraper first."

        # Generate embedding for the question
        question_embedding = self.model.encode_text(question)
        
        # Compare with all document embeddings
        similarities = []
        for source, doc_embedding in self.embeddings.items():
            similarity = self.model.compute_similarity(question_embedding, [doc_embedding])[0]
            similarities.append((source, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get the most relevant document
        most_relevant_source = similarities[0][0]
        context = self.documents[most_relevant_source]
        
        # Generate response using the context
        return self.model.generate_response(question, [context])

class JenkinsChatbot:
    def __init__(self):
        self.kb = JenkinsKnowledgeBase()
        
    def handle_query(self, query: str) -> str:
        """Process user query and return response"""
        return self.kb.query(query)

    def initialize_knowledge_base(self):
        """Initialize the knowledge base with fresh documentation"""
        scraper = JenkinsDocScraper()
        scraper.scrape_documentation()
        self.kb.load_documents()

def main():
    # Initialize chatbot
    chatbot = JenkinsChatbot()
    
    # First-time setup
    print("Initializing Jenkins AI Assistant...")
    print("Scraping documentation (this may take a few minutes)...")
    chatbot.initialize_knowledge_base()
    
    print("\nJenkins AI Assistant is ready! Type 'exit' to quit.")
    
    while True:
        query = input("\nYour question: ").strip()
        
        if query.lower() == 'exit':
            break
            
        response = chatbot.handle_query(query)
        print("\nAssistant:", response)

if __name__ == "__main__":
    main()