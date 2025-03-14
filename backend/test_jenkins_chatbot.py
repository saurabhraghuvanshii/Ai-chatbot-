from jenkins_chatbot import JenkinsDataRetriever, NLPProcessor, JenkinsKnowledgeBase

# Initialize components
data_retriever = JenkinsDataRetriever()
nlp_processor = NLPProcessor()
knowledge_base = JenkinsKnowledgeBase(nlp_processor, data_retriever)
 
# Test data retrieval
print("Fetching plugins data...")
plugins = data_retriever.fetch_plugins_data()
print(f"Retrieved {len(plugins)} plugins")

# Test NLP processing
test_query = "How do I install the Git plugin for Jenkins 2.375?"
print(f"\nProcessing query: '{test_query}'")
intent, entities = nlp_processor.process_query(test_query)
print(f"Detected intent: {intent}")
print(f"Extracted entities: {entities}")

# Initialize knowledge base (this will take some time)
print("\nInitializing knowledge base...")
knowledge_base.initialize()

# Test retrieval
print("\nRetrieving information...")
results = knowledge_base.retrieve(test_query, intent, entities)
print(f"Found {len(results['plugins'])} relevant plugins")
print(f"Found {len(results['documentation'])} relevant documentation pages")
print(f"Found {len(results['forum'])} relevant forum posts")

# Print top plugin result if available
if results['plugins']:
    top_plugin = results['plugins'][0]
    print(f"\nTop plugin match: {top_plugin['info']['name']}")
    print(f"Description: {top_plugin['info']['description']}")
    print(f"Relevance score: {top_plugin['relevance']:.4f}")
