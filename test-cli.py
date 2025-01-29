import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
TEXT_FILES_DIR = "text_files"  # Directory containing text files
LLAMA_CPP_PATH = "./llama.cpp/main"  # Path to llama.cpp binary
MODEL_PATH = "./llama.cpp/models/llama-2-7b.bin"  # Path to LLaMA model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Pre-trained embedding model
QUESTION = "What is the main topic of this text?"  # Question to ask
OUTPUT_FILE = "answers.txt"  # Output file for answers

# Load embedding model
embedder = SentenceTransformer(EMBEDDING_MODEL)

def read_text_files(directory):
    """Read all text files in the directory and return their content."""
    texts = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r") as file:
                texts[filename] = file.read()
    return texts

def generate_embeddings(texts):
    """Generate embeddings for a list of texts."""
    return embedder.encode(list(texts.values()))

def find_most_relevant_text(query_embedding, embeddings, texts):
    """Find the most relevant text based on cosine similarity."""
    similarities = cosine_similarity(query_embedding, embeddings)
    most_similar_index = np.argmax(similarities)
    most_similar_filename = list(texts.keys())[most_similar_index]
    most_similar_text = texts[most_similar_filename]
    return most_similar_filename, most_similar_text

def ask_question_with_llama(text, question):
    """Use llama.cpp to ask a question about the text."""
    prompt = f"{text}\n\nQuestion: {question}\nAnswer:"
    result = os.popen(f"{LLAMA_CPP_PATH} -m {MODEL_PATH} -p \"{prompt}\"").read()
    return result.strip()

def main():
    # Step 1: Read text files
    texts = read_text_files(TEXT_FILES_DIR)
    print(f"Read {len(texts)} text files.")

    # Step 2: Generate embeddings for the texts
    embeddings = generate_embeddings(texts)
    print("Generated embeddings for the texts.")

    # Step 3: Generate embedding for the question
    query_embedding = embedder.encode([QUESTION])
    print("Generated embedding for the question.")

    # Step 4: Find the most relevant text
    filename, relevant_text = find_most_relevant_text(query_embedding, embeddings, texts)
    print(f"Most relevant text: {filename}")

    # Step 5: Ask the question using llama.cpp
    answer = ask_question_with_llama(relevant_text, QUESTION)
    print(f"Answer: {answer}")

    # Step 6: Save the answer to a file
    with open(OUTPUT_FILE, "w") as outfile:
        outfile.write(f"Question: {QUESTION}\n")
        outfile.write(f"Relevant File: {filename}\n")
        outfile.write(f"Answer: {answer}\n")
    print(f"Answer saved to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()

    https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/blob/main/nomic-embed-text-v1.5.Q5_K_S.gguf