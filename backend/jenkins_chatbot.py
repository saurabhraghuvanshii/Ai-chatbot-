import os
import re
import json
import requests
from typing import List, Dict, Any, Tuple
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
 
class JenkinsDataRetriever:
    """Component for retrieving Jenkins documentation and plugin information"""
    
    def __init__(self, cache_dir: str = "./jenkins_data"):
        self.cache_dir = cache_dir
        self.plugins_data = {}
        self.documentation_data = {}
        self.forum_data = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def fetch_plugins_data(self, update_cache: bool = False) -> Dict:
        """Fetch Jenkins plugins data from the update center"""
        cache_file = os.path.join(self.cache_dir, "plugins_data.json")
        
        # Use cached data if available and not forced to update
        if os.path.exists(cache_file) and not update_cache:
            with open(cache_file, 'r') as f:
                self.plugins_data = json.load(f)
            return self.plugins_data
        
        # Fetch data from Jenkins update center
        try:
            # Updated URL - use .actual.json instead of .json
            response = requests.get("https://updates.jenkins.io/current/update-center.actual.json")
            
            # If response is not JSON, try alternative URL
            if "application/json" not in response.headers.get("Content-Type", ""):
                response = requests.get("https://updates.jenkins.io/update-center.json")
                
            # Remove )]; prefix if present (sometimes Jenkins wraps JSON in JSONP)
            content = response.text
            if content.startswith("updateCenter.post("):
                content = content.split("updateCenter.post(", 1)[1]
                if content.endswith(");"):
                    content = content[:-2]
                    
            data = json.loads(content)
            
            # Extract plugin information
            self.plugins_data = {
                plugin_id: {
                    "name": plugin_info.get("name", ""),
                    "version": plugin_info.get("version", ""),
                    "description": plugin_info.get("title", plugin_info.get("description", "")),
                    "url": plugin_info.get("url", ""),
                    "wiki": plugin_info.get("wiki", ""),
                    "dependencies": plugin_info.get("dependencies", [])
                }
                for plugin_id, plugin_info in data.get("plugins", {}).items()
            }
            
            # Cache the data
            with open(cache_file, 'w') as f:
                json.dump(self.plugins_data, f)
                
            return self.plugins_data
            
        except Exception as e:
            print(f"Error fetching plugin data: {e}")
            # Create sample data as fallback
            sample_plugins = {
                "git": {
                    "name": "Git",
                    "version": "4.11.5",
                    "description": "Integrates Jenkins with Git version control system",
                    "url": "https://plugins.jenkins.io/git/",
                    "wiki": "https://plugins.jenkins.io/git/",
                    "dependencies": []
                },
                "pipeline": {
                    "name": "Pipeline",
                    "version": "2.7",
                    "description": "A suite of plugins that lets you orchestrate automation, simple or complex",
                    "url": "https://plugins.jenkins.io/workflow-aggregator/",
                    "wiki": "https://plugins.jenkins.io/workflow-aggregator/",
                    "dependencies": []
                }
            }
            self.plugins_data = sample_plugins
            return self.plugins_data

    def fetch_documentation(self, urls: List[str], update_cache: bool = False) -> Dict:
        """Fetch Jenkins documentation from specified URLs"""
        cache_file = os.path.join(self.cache_dir, "documentation_data.json")
        
        # Use cached data if available and not forced to update
        if os.path.exists(cache_file) and not update_cache:
            with open(cache_file, 'r') as f:
                self.documentation_data = json.load(f)
            return self.documentation_data
        
        # If we can't fetch actual documentation, create sample data
        sample_docs = {
            "git_plugin": {
                "title": "Git Plugin - Jenkins",
                "content": "The Git Plugin allows Jenkins to integrate with Git repositories to trigger builds and use source code. To install the Git plugin, go to Manage Jenkins -> Manage Plugins -> Available and search for 'Git'. Select the Git plugin and click 'Install without restart'.",
                "url": "https://plugins.jenkins.io/git/"
            },
            "jenkins_install": {
                "title": "Installing Jenkins",
                "content": "Jenkins can be installed on various platforms including Windows, macOS, and Linux. For Linux, you can use package managers like apt or yum. For Jenkins 2.375, make sure your Java version is compatible (Java 11 or Java 17 recommended).",
                "url": "https://www.jenkins.io/doc/book/installing/"
            },
            "pipeline_tutorial": {
                "title": "Pipeline Tutorial",
                "content": "Jenkins Pipeline provides a suite of plugins that supports implementing and integrating continuous delivery pipelines into Jenkins. Define your pipeline in a Jenkinsfile using either declarative or scripted syntax.",
                "url": "https://www.jenkins.io/doc/book/pipeline/"
            }
        }
        
        try:
            for url in urls:
                # Attempt to fetch real documentation
                response = requests.get(url, timeout=5)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract title and content
                title = soup.title.string if soup.title else url
                content = soup.get_text()
                
                # Store in documentation data
                self.documentation_data[url] = {
                    "title": title,
                    "content": content,
                    "url": url
                }
        except Exception as e:
            print(f"Error fetching documentation: {e}")
            # Use sample data as fallback
            self.documentation_data = sample_docs
            
        # Cache the data
        with open(cache_file, 'w') as f:
            json.dump(self.documentation_data, f)
            
        return self.documentation_data
        
    def fetch_forum_data(self, forum_urls: List[str], update_cache: bool = False) -> Dict:
        """Fetch data from Jenkins community forums"""
        cache_file = os.path.join(self.cache_dir, "forum_data.json")
        
        # Use cached data if available and not forced to update
        if os.path.exists(cache_file) and not update_cache:
            with open(cache_file, 'r') as f:
                self.forum_data = json.load(f)
            return self.forum_data
        
        # Process forum URLs
        try:
            for url in forum_urls:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract threads and posts
                threads = soup.find_all('div', class_='thread')
                for thread in threads:
                    thread_id = thread.get('id', '')
                    title = thread.find('h2').text if thread.find('h2') else ''
                    posts = thread.find_all('div', class_='post')
                    
                    self.forum_data[thread_id] = {
                        "title": title,
                        "url": url,
                        "posts": [post.get_text() for post in posts]
                    }
            
            # Cache the data
            with open(cache_file, 'w') as f:
                json.dump(self.forum_data, f)
                
            return self.forum_data
            
        except Exception as e:
            print(f"Error fetching forum data: {e}")
            return {}


class NLPProcessor:
    """Process natural language queries related to Jenkins"""
    
    def __init__(self):
        # Load pre-trained models
        self.intent_model = self._load_intent_model()
        self.entity_model = self._load_entity_model()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Intent categories
        self.intents = [
            "plugin_info", "installation_help", "configuration_help", 
            "troubleshooting", "usage_example", "compatibility", "general_query"
        ]
        
    def _load_intent_model(self):
        """Load or initialize intent classification model"""
        try:
            # Use transformers pipeline for text classification
            model_name = "distilbert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            return pipeline("text-classification", model=model, tokenizer=tokenizer)
        except Exception as e:
            print(f"Error loading intent model: {e}")
            # Fall back to rule-based intent detection
            return None
    
    def _load_entity_model(self):
        """Load or initialize entity extraction model"""
        try:
            # Use transformers pipeline for NER
            return pipeline("ner")
        except Exception as e:
            print(f"Error loading entity model: {e}")
            # Fall back to rule-based entity extraction
            return None
    
    def extract_intent(self, query: str) -> str:
        """Extract the intent from a user query"""
        if self.intent_model:
            # Use ML model for intent classification
            result = self.intent_model(query)
            intent = result[0]['label']
            return intent
        else:
            # Rule-based intent detection as fallback
            query = query.lower()
            if any(word in query for word in ["install", "download", "setup", "set up"]):
                return "installation_help"
            elif any(word in query for word in ["configure", "setting", "option", "parameter"]):
                return "configuration_help"
            elif any(word in query for word in ["error", "issue", "problem", "fail", "bug"]):
                return "troubleshooting"
            elif any(word in query for word in ["example", "sample", "how to", "tutorial"]):
                return "usage_example"
            elif any(word in query for word in ["compatible", "version", "support", "work with"]):
                return "compatibility"
            elif any(word in query for word in ["plugin", "extension", "add-on"]):
                return "plugin_info"
            else:
                return "general_query"
    
    def extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities from a user query"""
        entities = {
            "plugin_name": [],
            "version": [],
            "jenkins_version": [],
            "action": [],
            "error_message": []
        }
        
        # Rule-based entity extraction
        # Extract plugin names using improved regex
        plugin_pattern = r'(plugin|extension|add-on)\s+(?:called|named)?\s*([a-zA-Z0-9\-]+)'
        plugin_direct_pattern = r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+plugin'
        
        plugin_matches = re.findall(plugin_pattern, query, re.IGNORECASE)
        for match in plugin_matches:
            if match[1].lower() not in ['for', 'in', 'to', 'with', 'and', 'the']:
                entities["plugin_name"].append(match[1])
        
        direct_matches = re.findall(plugin_direct_pattern, query)
        for match in direct_matches:
            entities["plugin_name"].append(match)
        
        # Specific plugin extraction for common plugins
        common_plugins = ["git", "maven", "pipeline", "docker", "kubernetes", "credentials", "ssh"]
        for plugin in common_plugins:
            if re.search(r'\b' + plugin + r'\b', query, re.IGNORECASE):
                entities["plugin_name"].append(plugin)
        
        # Extract versions using regex
        version_pattern = r'(version|v)\s+(\d+(\.\d+)*)'
        jenkins_version_pattern = r'Jenkins\s+(\d+(\.\d+)*)'
        
        version_matches = re.findall(version_pattern, query, re.IGNORECASE)
        for match in version_matches:
            entities["version"].append(match[1])
        
        jenkins_matches = re.findall(jenkins_version_pattern, query, re.IGNORECASE)
        for match in jenkins_matches:
            entities["jenkins_version"].append(match[0])
        
        # Extract error messages using regex
        error_pattern = r'(error|exception):\s+([^\.\n]+)'
        error_matches = re.findall(error_pattern, query, re.IGNORECASE)
        for match in error_matches:
            entities["error_message"].append(match[1])
        
        # Extract actions
        action_words = ["install", "configure", "setup", "update", "remove", "delete", "use"]
        for action in action_words:
            if re.search(r'\b' + action + r'\b', query, re.IGNORECASE):
                entities["action"].append(action)
        
        return entities
    
    def process_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Process a user query and return intent and entities"""
        intent = self.extract_intent(query)
        entities = self.extract_entities(query)
        return intent, entities
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Get vector embedding for a query"""
        return self.embedding_model.encode(query)
    
    def get_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get vector embeddings for a list of texts"""
        return self.embedding_model.encode(texts)
    
    def get_tfidf_matrix(self, texts: List[str]) -> np.ndarray:
        """Get TF-IDF matrix for a list of texts"""
        return self.vectorizer.fit_transform(texts)


class JenkinsKnowledgeBase:
    """Knowledge base for storing and retrieving Jenkins information"""
    
    def __init__(self, nlp_processor: NLPProcessor, data_retriever: JenkinsDataRetriever):
        self.nlp_processor = nlp_processor
        self.data_retriever = data_retriever
        self.plugin_embeddings = {}
        self.doc_embeddings = {}
        self.forum_embeddings = {}
        
    def initialize(self):
        """Initialize the knowledge base with data"""
        # Fetch data
        self.data_retriever.fetch_plugins_data()
        self.data_retriever.fetch_documentation([
            "https://www.jenkins.io/doc/",
            "https://www.jenkins.io/doc/book/installing/",
            "https://www.jenkins.io/doc/book/pipeline/",
            # Add more URLs as needed
        ])
        self.data_retriever.fetch_forum_data([
            "https://community.jenkins.io/",
            # Add more forums as needed
        ])
        
        # Create embeddings for plugins
        plugin_texts = []
        plugin_ids = []
        for plugin_id, plugin_info in self.data_retriever.plugins_data.items():
            text = f"{plugin_info['name']} {plugin_info['description']}"
            plugin_texts.append(text)
            plugin_ids.append(plugin_id)
        
        if plugin_texts:
            self.plugin_embeddings = {
                "ids": plugin_ids,
                "embeddings": self.nlp_processor.get_text_embeddings(plugin_texts)
            }
        
        # Create embeddings for documentation
        doc_texts = []
        doc_ids = []
        for doc_id, doc_info in self.data_retriever.documentation_data.items():
            text = f"{doc_info['title']} {doc_info['content']}"
            doc_texts.append(text)
            doc_ids.append(doc_id)
        
        if doc_texts:
            self.doc_embeddings = {
                "ids": doc_ids,
                "embeddings": self.nlp_processor.get_text_embeddings(doc_texts)
            }
        
        # Create embeddings for forum posts
        forum_texts = []
        forum_ids = []
        for thread_id, thread_info in self.data_retriever.forum_data.items():
            text = f"{thread_info['title']} " + " ".join(thread_info['posts'])
            forum_texts.append(text)
            forum_ids.append(thread_id)
        
        if forum_texts:
            self.forum_embeddings = {
                "ids": forum_ids,
                "embeddings": self.nlp_processor.get_text_embeddings(forum_texts)
            }
    
    def retrieve(self, query: str, intent: str, entities: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        """Retrieve relevant information based on query, intent and entities"""
        query_embedding = self.nlp_processor.get_query_embedding(query)
        results = {
            "plugins": [],
            "documentation": [],
            "forum": []
        }
        
        # Retrieve plugin information
        if intent in ["plugin_info", "installation_help", "configuration_help"] or entities["plugin_name"]:
            if self.plugin_embeddings:
                # Calculate similarities
                similarities = np.dot(self.plugin_embeddings["embeddings"], query_embedding)
                # Get top k results
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                for idx in top_indices:
                    plugin_id = self.plugin_embeddings["ids"][idx]
                    plugin_info = self.data_retriever.plugins_data[plugin_id]
                    results["plugins"].append({
                        "id": plugin_id,
                        "info": plugin_info,
                        "relevance": float(similarities[idx])
                    })
        
        # Retrieve documentation
        if self.doc_embeddings:
            # Calculate similarities
            similarities = np.dot(self.doc_embeddings["embeddings"], query_embedding)
            # Get top k results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            for idx in top_indices:
                doc_id = self.doc_embeddings["ids"][idx]
                doc_info = self.data_retriever.documentation_data[doc_id]
                results["documentation"].append({
                    "id": doc_id,
                    "info": doc_info,
                    "relevance": float(similarities[idx])
                })
        
        # Retrieve forum posts
        # Retrieve forum posts
        if intent in ["troubleshooting", "usage_example"] or entities["error_message"]:
            if self.forum_embeddings:
                # Calculate similarities
                similarities = np.dot(self.forum_embeddings["embeddings"], query_embedding)
                # Get top k results
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                for idx in top_indices:
                    forum_id = self.forum_embeddings["ids"][idx]
                    forum_info = self.data_retriever.forum_data[forum_id]
                    results["forum"].append({
                        "id": forum_id,
                        "info": forum_info,
                        "relevance": float(similarities[idx])
                    })

        return results