import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz
from openai import OpenAI
from constants import API_KEY

# Load your OpenAI API key from the environment or set it directly
api_key = os.getenv("OPENAI_API_KEY", API_KEY)
client = OpenAI(api_key=api_key)

# File paths for storing the extracted corpus
corpus_file_path = "corpus_data.pkl"

# Maximum token threshold to trigger summarization
TOKEN_THRESHOLD = 3000

# Summarize long conversations to compress tokens (only conversation, not relevant context)
def summarize_conversation(conversation_history):
    """
    Summarizes the older parts of the conversation to reduce token count.
    
    Args:
    - conversation_history (list): List of conversation messages.
    
    Returns:
    - list: Updated conversation history with summaries replacing older content.
    """
    # Summarize only conversation history, not context or relevant passages
    conversation_text = "\n".join([entry["content"] for entry in conversation_history if entry["role"] == "user" or entry["role"] == "assistant"])
    
    summary_prompt = [
        {"role": "system", "content": "Summarize the following conversation for brevity while keeping the core meaning."},
        {"role": "user", "content": conversation_text}
    ]

    summary_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=summary_prompt,
        max_tokens=300,
        temperature=0.5
    )

    summary = summary_response.choices[0].message.content.strip()

    # Return a new conversation history that replaces older entries with the summary
    summarized_history = [{"role": "system", "content": f"Conversation summary: {summary}"}]

    # Append the most recent assistant and user messages to the summarized history
    summarized_history += conversation_history[-2:]  # Keep the most recent two exchanges

    return summarized_history

def save_corpus_to_file(text_corpus, table_corpus, image_corpus, file_path):
    """Saves the text, table, and image corpuses to a file using pickle."""
    with open(file_path, "wb") as file:
        pickle.dump((text_corpus, table_corpus, image_corpus), file)

def load_corpus_from_file(file_path):
    """Loads the text, table, and image corpuses from a file using pickle."""
    with open(file_path, "rb") as file:
        return pickle.load(file)

def load_corpus_tables_and_images(pdf_path):
    """
    Extracts both text, tables, and images from a PDF file.
    
    Args:
    - pdf_path (str): The file path to the PDF document.
    
    Returns:
    - text_corpus (list): A list of extracted text per page from the PDF.
    - table_corpus (list): A list of tables extracted from the PDF in structured format (list of rows).
    - image_corpus (list): A list of images extracted as metadata about images on each page.
    """
    pdf_document = fitz.open(pdf_path)
    text_corpus = []
    table_corpus = []
    image_corpus = []

    for page_number in range(pdf_document.page_count):
        page = pdf_document[page_number]
        text = page.get_text("text")  # Extract the page's plain text
        text_corpus.append(text)

        # Extract tables
        tables = page.find_tables(strategy='lines_strict')
        if tables:
            for table in tables:
                # Extract table as text
                extracted_table = table.extract()
                table_corpus.append(extracted_table)

        # Extract images metadata
        images = page.get_images(full=True)
        for image in images:
            xref = image[0]
            image_rect = page.get_image_bbox(image)
            image_info = {
                "xref": xref,
                "image_rect": image_rect,
                "image_metadata": image
            }
            image_corpus.append(image_info)

    pdf_document.close()
    return text_corpus, table_corpus, image_corpus

# Check if the saved corpus exists
if os.path.exists(corpus_file_path):
    print("Loading corpus from saved file...")
    text_corpus, table_corpus, image_corpus = load_corpus_from_file(corpus_file_path)
else:
    print("Extracting corpus from the PDF...")
    text_corpus, table_corpus, image_corpus = load_corpus_tables_and_images("proposal.pdf")
    print("Saving corpus for future use...")
    save_corpus_to_file(text_corpus, table_corpus, image_corpus, corpus_file_path)

# Load a pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode text corpus
text_embeddings = model.encode(text_corpus, convert_to_tensor=True)

# Encode tables by treating each row as a separate entity to allow targeted row-based queries
table_embeddings = []
flattened_table_corpus = []  # Flatten the table corpus for easier retrieval

for table in table_corpus:
    for row in table:
        # Convert any NoneType to an empty string before joining the row
        row_text = " | ".join([str(cell) if cell is not None else "" for cell in row])
        flattened_table_corpus.append(row_text)
        # Encode each row and append it as a 2D tensor (embedding for each row)
        row_embedding = model.encode([row_text], convert_to_tensor=False)
        table_embeddings.append(row_embedding)

# Convert list of embeddings to a 2D numpy array
if len(table_embeddings) > 0:
    table_embeddings = np.vstack(table_embeddings)  # Use vstack to ensure the 2D shape

# Set up FAISS indices for text, tables, and images
embedding_dim = text_embeddings.shape[1]
text_index = faiss.IndexFlatL2(embedding_dim)

if len(table_embeddings) > 0:
    table_index = faiss.IndexFlatL2(embedding_dim)
    table_index.add(table_embeddings)

text_index.add(np.array(text_embeddings))

# Optionally create embeddings for images (using metadata or descriptions of images)
image_metadata = ["Image on page with coordinates " + str(image["image_rect"]) for image in image_corpus]
image_embeddings = model.encode(image_metadata, convert_to_tensor=False)
image_index = faiss.IndexFlatL2(embedding_dim)
image_index.add(image_embeddings)

def retrieve_relevant_passages(query, top_k=3):
    """
    Retrieves the most relevant passages from the corpus based on the input query.
    
    Args:
    - query (str): The user query.
    - top_k (int): Number of top relevant passages to retrieve.
    
    Returns:
    - dict: Contains results from text, tables, and images.
    """
    query_embedding = model.encode([query], convert_to_tensor=True)

    # Search text
    distances_text, indices_text = text_index.search(np.array(query_embedding), top_k)
    relevant_texts = [text_corpus[idx] for idx in indices_text[0]]

    # Search tables
    relevant_tables = []
    if len(table_embeddings) > 0:
        distances_table, indices_table = table_index.search(np.array(query_embedding), top_k)
        relevant_tables = [flattened_table_corpus[idx] for idx in indices_table[0]]

    # Search images
    relevant_images = []
    if len(image_embeddings) > 0:
        distances_image, indices_image = image_index.search(np.array(query_embedding), top_k)
        relevant_images = [image_metadata[idx] for idx in indices_image[0]]

    return {
        "text": relevant_texts,
        "tables": relevant_tables,
        "images": relevant_images
    }

# Initialize conversation history
conversation_history = []

def generate_response(relevant_passages, user_query):
    """
    Generates a response based on the retrieved passages and the user's query, considering conversation history.
    
    Args:
    - relevant_passages (dict): The most relevant passages retrieved from the corpus (text, tables, and images).
    - user_query (str): The user query.
    
    Returns:
    - str: The generated response.
    """
    global conversation_history

    # Combine the relevant passages to provide context
    context_text = "\n".join(relevant_passages["text"])
    context_tables = "\n".join(relevant_passages["tables"])
    context_images = "\n".join(relevant_passages["images"])
    
    context = f"Text:\n{context_text}\n\nTables:\n{context_tables}\n\nImages:\n{context_images}"

    # Append current query and retrieved context to conversation history
    conversation_history.append({"role": "user", "content": user_query})
    conversation_history.append({"role": "system", "content": f"Context:\n{context}"})

    # Check if token count exceeds the threshold (summarize only old conversation, not the current context)
    current_token_count = sum([len(entry["content"].split()) for entry in conversation_history])
    if current_token_count > TOKEN_THRESHOLD:
        conversation_history = summarize_conversation(conversation_history)
    
    # Prepare the messages for GPT (use summarized conversation and current context)
    messages = conversation_history + [
        {"role": "user", "content": f"Context:\n{context}\n\nUser Query: {user_query}"}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1500,  # Adjust if necessary
        temperature=0.7
    )

    # Extract and return the response content
    return response.choices[0].message.content.strip()

# Command-line chatbot loop
print("Type your questions below (Ctrl+C to exit).")
while True:
    try:
        user_query = input("You: ")

        # Retrieve relevant passages from text, tables, and images
        relevant_passages = retrieve_relevant_passages(user_query)

        # Generate and display the response
        response = generate_response(relevant_passages, user_query)
        print(f"Chatbot: {response}\n")
    except KeyboardInterrupt:
        print("\nExiting...")
        break
