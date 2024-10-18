import os
from pinecone import Pinecone, ServerlessSpec
import openai
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
PINECONE_API_KEY = os.getenv("84e92eba-57bc-48b9-b2ed-ee7b4be85608")
OPENAI_API_KEY = os.getenv("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")

# Initialize Pinecone with the API key from environment
pc = Pinecone(api_key="84e92eba-57bc-48b9-b2ed-ee7b4be85608")

# Initialize OpenAI API
openai.api_key = "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"

# Define your index specifications
index_name = 'eco-friendly-chatbot'
dimension = 1536
metric = 'euclidean'

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east1'  # Change to us-east1
        )
    )


# Connect to the index
index = pc.index(index_name)

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Function to generate GPT responses
def chat_with_gpt(prompt):
    try:
        response = openai.Completion.create(
            engine="gpt-3.5-turbo",
            prompt=prompt,
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].text.strip() if response.choices else "No response generated."
    except Exception as e:
        print(f"Error generating GPT response: {e}")
        return "There was an error generating a response."

# Function to query the vector database
def query_vector_db(query):
    try:
        query_emb = embeddings.embed_query(query)  # Correct for queries
        results = index.query(query_emb, top_k=5, include_metadata=True)
        return results['matches']
    except Exception as e:
        print(f"Error querying vector database: {e}")

# Enhanced chatbot response with vector search
def enhanced_chatbot_response(user_input):
    search_results = query_vector_db(user_input)
    if search_results:
        response = "\n".join([result['metadata']['text'] for result in search_results])
    else:
        response = chat_with_gpt(user_input)
    return response

# Streamlit interface
st.title("Biodiversity Protection & Eco-friendly Chatbot")

# User input
user_input = st.text_input("Ask me anything about biodiversity protection or eco-friendly living:")

# Handle user input and display chatbot responses
if user_input:
    response = enhanced_chatbot_response(user_input)
    st.write(response)

# Example code for embedding and inserting documents into Pinecone
documents = [
    {"text": "Protecting biodiversity is essential for the health of the planet."},
    {"text": "Eco-friendly lifestyle includes reducing waste, conserving energy, and using renewable resources."},
]

# Convert documents to embeddings and insert into Pinecone
for i, doc in enumerate(documents):
    try:
        doc_emb = embeddings.embed_document(doc['text'])  # Correct method for documents
        index.upsert([(str(i), doc_emb, {"text": doc['text']})])
    except Exception as e:
        print(f"Error embedding and inserting document {i}: {e}")

available_regions = pc.list_regions()
print(available_regions)  # This will give you a list of available regions for your account

