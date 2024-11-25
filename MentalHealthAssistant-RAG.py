import os
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
from dotenv import load_dotenv
import pickle

def load_and_preprocess_data(file_path):
    # Load dataset
    data = pd.read_csv(file_path)

    # Extract relevant text column; replace 'text_column' with your column name
    if 'statement' not in data.columns:
        raise ValueError("Dataset does not contain a column named 'text_column'")
    
    documents = data['statement'].dropna().tolist()
    return documents

def create_faiss_index(documents, model_name='all-MiniLM-L6-v2'):
    # Initialize sentence transformer model
    model = SentenceTransformer(model_name)

    # Generate embeddings for documents
    embeddings = model.encode(documents)

    # Create and populate FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, model

def retrieve_relevant_docs(query, index, model, documents, top_k=3):
    # Generate query embedding
    query_embedding = model.encode([query])

    # Search index
    distances, indices = index.search(query_embedding, top_k)

    # Return top-k relevant documents
    return [documents[i] for i in indices[0]]

def main():
    try:
        load_dotenv()

        # Azure OpenAI setup
        endpoint = os.getenv("OpenAI_ENDPOINT_URL")
        deployment = os.getenv("AZURE_OAI_DEPLOYMENT_NAME")
        key = os.getenv("OpenAI_KEY")

        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version="2023-09-01-preview"
        )

        # File paths for saved FAISS index and embeddings
        documents_file = "documents.pkl"
        index_file = "faiss_index.bin"

        # Check if FAISS index exists
        if os.path.exists(documents_file) and os.path.exists(index_file):
            # Reload FAISS index and embeddings
            with open(documents_file, "rb") as f:
                documents = pickle.load(f)
            faiss_index = faiss.read_index(index_file)
            print("FAISS index and embeddings loaded from disk.")
        else:
            # Create FAISS index
            print("Creating FAISS index and embeddings...")
            #---------------------------------------------------#
            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))

            # Construct the path to the CSV file in the data folder
            dataset_path = os.path.join(script_dir, "Data", "Combined Data.csv")
            print(dataset_path)

            # Optional: Verify file existence
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"File not found: {dataset_path}")
            #----------------------------------------------------------------#

            documents = load_and_preprocess_data(dataset_path)
            faiss_index, sentence_model = create_faiss_index(documents)
        
            # Save them for future runs
            with open(documents_file, "wb") as f:
                pickle.dump(documents, f)
            faiss.write_index(faiss_index, index_file)
            print("FAISS index and embeddings saved to disk.")
        
        # Initialize conversation
        system_message = "You are a mental health assistant. Use the provided data to answer questions effectively."
        messages_array = [{"role": "system", "content": system_message}]

        while True:
            # Get user input
            user_input = input("\nEnter your question (or type 'quit' to exit):\n")
            if user_input.lower() == "quit":
                break

            # Retrieve relevant documents
            relevant_docs = retrieve_relevant_docs(user_input, faiss_index, sentence_model, documents)
            context = "\n".join(relevant_docs)

            # Add context to prompt
            context_message = f"Here is relevant information from our dataset:\n{context}\n\nUser question: {user_input}"

            # Update messages with user input and context
            messages_array.append({"role": "user", "content": context_message})

            # Call Azure OpenAI
            response = client.chat.completions.create(
                model=deployment,
                messages=messages_array,
                max_tokens=1200,
                temperature=0.7,
                top_p=0.95
            )

            # Extract and print the assistant's response
            assistant_response = response.choices[0].message.content
            print("\nAssistant:", assistant_response)

            # Add response to messages
            messages_array.append({"role": "assistant", "content": assistant_response})

    except Exception as ex:
        print(f"Error: {ex}")

if __name__ == "__main__":
    main()
