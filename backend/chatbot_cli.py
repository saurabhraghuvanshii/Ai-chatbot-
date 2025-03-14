from jenkins_chatbot import JenkinsDataRetriever, NLPProcessor, JenkinsKnowledgeBase
import textwrap
from typing import Dict
import re

class JenkinsChatbotCLI:
    def __init__(self):
        print("Initializing Jenkins AI Assistant...")
        print("Loading Jenkins documentation and plugin data...")
        
        self.data_retriever = JenkinsDataRetriever()
        self.nlp_processor = NLPProcessor()
        self.knowledge_base = JenkinsKnowledgeBase(self.nlp_processor, self.data_retriever)
        
        self.knowledge_base.initialize()
        print("\nJenkins AI Assistant is ready!")
        print("\nYou can ask questions about:")
        print("- Installing and configuring Jenkins")
        print("- Creating and using plugins")
        print("- Working with Jenkins Pipeline")
        print("- Troubleshooting issues")
        print("- And more!")
        
    def generate_response(self, query: str, results: Dict) -> str:
        """Generate a natural language response for Jenkins queries"""
        intent, entities = self.nlp_processor.process_query(query)
        
        response_parts = []
        seen_content = set()  # Track unique content to avoid repetition
        
        # Add context based on intent
        if intent == "plugin_development":
            response_parts.append("Here's information about developing Jenkins plugins:")
            response_parts.append("\n1. Set up your development environment with Java and Maven")
            response_parts.append("2. Use the Jenkins plugin archetype to create a new plugin:")
            response_parts.append("   mvn archetype:generate -Dfilter=io.jenkins.archetypes:plugin")
            response_parts.append("3. Import the project into your IDE")
            response_parts.append("4. Build with: mvn clean verify")
            
        elif intent == "installation":
            response_parts.append("Here's how to install Jenkins:")
            response_parts.append("\n1. Ensure Java 11 or Java 17 is installed")
            response_parts.append("2. Choose your installation method:")
            response_parts.append("   - Docker: docker pull jenkins/jenkins:lts")
            response_parts.append("   - Linux: sudo apt-get install jenkins")
            response_parts.append("   - Windows: Download the installer from jenkins.io")
            
        elif intent == "pipeline":
            response_parts.append("Jenkins Pipeline is a suite of plugins for implementing continuous delivery pipelines.")
            response_parts.append("\nKey concepts:")
            response_parts.append("- Pipeline can be written as code in a Jenkinsfile")
            response_parts.append("- Supports both Declarative and Scripted syntax")
            response_parts.append("- Enables defining the entire build process: build, test, deploy")
            response_parts.append("\nExample Declarative Pipeline:")
            response_parts.append("""
pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                sh 'make build'
            }
        }
        stage('Test') {
            steps {
                sh 'make test'
            }
        }
    }
}""")
            
        # Add documentation references with better content extraction and deduplication
        if results["documentation"]:
            relevant_docs = []
            
            for doc in results["documentation"]:
                title = doc['info']['title']
                content = doc['info']['content']
                
                # Skip if title is too generic
                if title.lower() in ['jenkins user documentation', 'jenkins']:
                    continue
                    
                # Extract relevant section
                relevant_section = self._extract_relevant_section(content, query)
                if relevant_section:
                    # Clean up the section
                    cleaned_section = self._clean_content(relevant_section)
                    
                    # Check for duplicate or similar content
                    content_hash = self._get_content_hash(cleaned_section)
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        relevant_docs.append({
                            'title': title,
                            'content': cleaned_section,
                            'url': doc['info']['url']
                        })
            
            # Add unique relevant documentation
            for doc in relevant_docs[:2]:  # Limit to top 2 unique docs
                response_parts.append(f"\n📚 {doc['title']}")
                response_parts.append(f"\n{doc['content']}")
                response_parts.append(f"\nMore details: {doc['url']}")
                
        # Add plugin information if relevant
        if results["plugins"]:
            response_parts.append("\n🔌 Relevant plugins:")
            for plugin in results["plugins"][:3]:  # Show top 3 plugins
                info = plugin['info']
                response_parts.append(f"\n- {info['name']}")
                if info.get('description'):
                    response_parts.append(f"  Description: {info['description']}")
                if info.get('url'):
                    response_parts.append(f"  Learn more: {info['url']}")
                    
        # Add troubleshooting info if needed
        if intent == "troubleshooting":
            response_parts.append("\n🔧 Troubleshooting tips:")
            response_parts.append("1. Check the Jenkins log files")
            response_parts.append("2. Verify permissions and access rights")
            response_parts.append("3. Ensure all required plugins are installed")
            response_parts.append("4. Review system requirements")
                    
        return "\n".join(response_parts)
        
    def _get_content_hash(self, content: str) -> str:
        """Generate a simple hash of the content for deduplication"""
        # Normalize content by removing common variations
        normalized = content.lower()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Get first 100 chars as a signature
        return normalized[:100]
        
    def _extract_relevant_section(self, content: str, query: str) -> str:
        """Extract most relevant section from documentation content"""
        # Split into paragraphs and filter out boilerplate
        paragraphs = []
        for para in content.split('\n\n'):
            para = para.strip()
            if len(para.split()) >= 10 and not any(boilerplate in para.lower() for boilerplate in [
                'there are a vast array of plugins',
                'theoretically, jenkins can',
                'the procedures in this chapter',
                'was this page helpful',
                'see existing feedback',
                'click here to learn more',
                'for more information, see',
                'table of contents',
                'chapter sub-sections'
            ]):
                paragraphs.append(para)
        
        # Find most relevant paragraph using keyword matching and context
        query_words = set(query.lower().split())
        scored_paragraphs = []
        
        for para in paragraphs:
            para_words = set(para.lower().split())
            
            # Calculate relevance score with improved weighting
            word_match_score = len(query_words.intersection(para_words)) * 3
            
            # Check for key phrases that indicate important content
            key_phrases = ['important', 'note', 'example', 'how to', 'usage', 'definition', 'means']
            phrase_score = sum(2 for phrase in key_phrases if phrase in para.lower())
            
            # Check for technical terms relevance
            tech_terms = ['pipeline', 'jenkins', 'plugin', 'step', 'stage', 'agent', 'node']
            tech_score = sum(1 for term in tech_terms if term in para.lower())
            
            total_score = word_match_score + phrase_score + tech_score
            
            if total_score > 0:
                # Consider paragraph position
                position_score = max(0, 5 - paragraphs.index(para) * 0.5)
                total_score += position_score
                scored_paragraphs.append((total_score, para))
        
        # Sort by score and combine top paragraphs
        if scored_paragraphs:
            scored_paragraphs.sort(reverse=True)
            good_paras = [p[1] for p in scored_paragraphs if p[0] > 3][:2]
            if good_paras:
                return '\n\n'.join(good_paras)
        
        return ""
        
    def _clean_content(self, content: str) -> str:
        """Clean up extracted content for better readability"""
        # Remove extra whitespace
        content = ' '.join(content.split())
        
        # Common boilerplate texts to remove
        boilerplate = [
            r'There are a vast array of plugins available to Jenkins.*?wizard\.',
            r'Theoretically, Jenkins can also be run as a servlet.*?details\.',
            r'The procedures in this chapter are for new installations of Jenkins\.',
            r'Was this page helpful\?.*',
            r'See existing feedback here\.',
            r'Click here to learn more\.',
            r'For more information, see.*?\.',
            r'Note: This is.*?\.',
            r'Important:.*?\.',
            r'Table of Contents.*?\n',
            r'Chapter Sub-Sections.*?\n',
        ]
        
        # Remove boilerplate content
        for pattern in boilerplate:
            content = re.sub(pattern, '', content, flags=re.DOTALL|re.IGNORECASE)
        
        # Remove other noise patterns
        noise_patterns = [
            r'\[[\d\s]+\]',          # Reference numbers
            r'\u00a0',               # Non-breaking spaces
            r'\s{2,}',              # Multiple spaces
            r'â',                   # Special characters
            r'\(.*?\)',             # Parenthetical asides
            r'\b\d+\.\d+\.\d+\b',   # Version numbers
            r'Click here.*?\.',     # Click here instructions
            r'See .*? for more.*?\.', # See X for more... phrases
        ]
        
        for pattern in noise_patterns:
            content = re.sub(pattern, ' ', content)
        
        # Smart truncation with better sentence handling
        if len(content) > 500:
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            truncated = []
            length = 0
            
            for sentence in sentences:
                if length + len(sentence) > 450:
                    break
                if sentence and not any(boilerplate_text in sentence.lower() for boilerplate_text in [
                    'see the documentation',
                    'refer to',
                    'this chapter',
                    'this page',
                    'this documentation',
                    'this guide'
                ]):
                    truncated.append(sentence)
                    length += len(sentence)
            
            content = '. '.join(truncated) + '...'
        
        # Final cleanup
        content = re.sub(r'\s+', ' ', content)  # Remove extra spaces
        content = re.sub(r'\.+', '.', content)  # Remove multiple periods
        content = re.sub(r'\s+\.', '.', content)  # Fix space before period
        
        return content.strip()
        
    def run(self):
        print("\nType your question about Jenkins (or 'exit' to quit)")
        
        while True:
            query = input("\n> ")
            if query.lower() in ['exit', 'quit', 'bye']:
                print("Thank you for using Jenkins AI Assistant. Goodbye!")
                break
                
            # Process query
            intent, entities = self.nlp_processor.process_query(query)
            results = self.knowledge_base.retrieve(query, intent, entities)
            
            # Generate and display response
            response = self.generate_response(query, results)
            print("\n" + "="*60)
            print("JENKINS AI ASSISTANT:")
            print("="*60)
            print(response)
            print("="*60)

if __name__ == "__main__":
    chatbot = JenkinsChatbotCLI()
    chatbot.run()