import hashlib
import json
from pathlib import Path
import sys

from dotenv import load_dotenv
from openai import OpenAI

import re
import numpy as np


RESOURCES_DIR = Path(__file__).parent / "resources"
VECTOR_STORE_DIR = Path(__file__).parent / "vector_store"
VECTOR_STORE_FILE = VECTOR_STORE_DIR / "embeddings.json"


def load_environment() -> None:
    """Load the repository-level environment file."""
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")


def load_documents() -> list[dict]:
    """Load all Markdown documents from the resources directory."""

    documents = []

    for file_path in sorted(RESOURCES_DIR.glob("*.md")):
        documents.append(
            {
                "source": file_path.name,
                "content": file_path.read_text(encoding="utf-8"),
            }
        )

    return documents


## This function takes a list of documents (each represented as a dictionary with 'source' and 'content' keys) and splits the content of each document into chunks
# based on Markdown section headings (specifically, headings that start with '##' or '###').
## Each chunk is then stored in a new dictionary that includes the source of the document, a unique chunk ID, and the chunk content itself. The resulting list of chunk dictionaries is returned.
def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents into chunks based on Markdown section headings."""

    chunks = []

    for document in documents:
        sections = re.split(
            r"(?=^#{2,3} )",
            document["content"],
            flags=re.MULTILINE,
        )

        for index, section in enumerate(sections):
            section = section.strip()

            if not section:
                continue

            chunks.append(
                {
                    "source": document["source"],
                    "chunk_id": f"{document['source']}:{index}",
                    "content": section,
                    "content_hash": create_content_hash(section),
                }
            )

    return chunks


## same old function to create embeddings for each chunk of text using OpenAI's embedding model.
## It takes a client instance and a list of document chunks, generates embeddings for each chunk, and adds the embedding to the chunk dictionary.
def update_embeddings(
    client: OpenAI,
    chunks: list[dict],
    existing_chunks: list[dict],
) -> list[dict]:
    """Reuse unchanged embeddings and create embeddings only for changed chunks."""

    existing_by_id = {chunk["chunk_id"]: chunk for chunk in existing_chunks}

    chunks_to_embed = []

    for chunk in chunks:
        existing_chunk = existing_by_id.get(chunk["chunk_id"])

        if (
            existing_chunk
            and existing_chunk.get("content_hash") == chunk["content_hash"]
        ):
            chunk["embedding"] = existing_chunk["embedding"]
        else:
            chunks_to_embed.append(chunk)

    if chunks_to_embed:
        texts = [chunk["content"] for chunk in chunks_to_embed]

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )

        for chunk, embedding_data in zip(
            chunks_to_embed,
            response.data,
        ):
            chunk["embedding"] = embedding_data.embedding

    print(f"Reused {len(chunks) - len(chunks_to_embed)} embeddings")
    print(f"Created {len(chunks_to_embed)} new embeddings")

    return chunks


## Very basic vector store implementation that persists embeddings to disk as JSON. In a production system, you would likely want to use a more robust solution like FAISS or Milvus.
def save_vector_store(chunks: list[dict]) -> None:
    """Persist document chunks and embeddings to disk."""

    VECTOR_STORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VECTOR_STORE_FILE.write_text(
        json.dumps(chunks, indent=2),
        encoding="utf-8",
    )


## This function loads the previously indexed chunks and embeddings from disk by reading the JSON file and returning it as a list of dictionaries.
def load_vector_store() -> list[dict]:
    """Load previously indexed chunks and embeddings from disk."""

    return json.loads(VECTOR_STORE_FILE.read_text(encoding="utf-8"))


## This function creates an embedding for a user's question using the same embedding model as used for the document chunks.
## It takes a client instance and the question string, generates the embedding, and returns it as a list of floats.
def create_query_embedding(
    client: OpenAI,
    question: str,
) -> list[float]:
    """Create an embedding for the user's question."""

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question,
    )

    return response.data[0].embedding


## This function calculates the cosine similarity between two vectors, which is a measure of how similar they are in terms of direction.
## It takes two lists of floats (vector_a and vector_b), converts them to NumPy arrays, and computes the cosine similarity using the dot product and norms of the vectors.
def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """Calculate cosine similarity between two vectors."""

    a = np.array(vector_a)
    b = np.array(vector_b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


## This function searches the vector store for the chunks most similar to a given query embedding.
def search_vector_store(
    chunks: list[dict],
    query_embedding: list[float],
    top_k: int = 3,
) -> list[dict]:
    """Find the chunks most similar to the query embedding."""

    scored_chunks = []

    for chunk in chunks:
        similarity = cosine_similarity(
            query_embedding,
            chunk["embedding"],
        )

        scored_chunks.append(
            {
                **chunk,
                "similarity": similarity,
            }
        )

    scored_chunks.sort(
        key=lambda chunk: chunk["similarity"],
        reverse=True,
    )

    return scored_chunks[:top_k]


## This function orchestrates the process of indexing documents into a vector store.
def index_documents(client: OpenAI) -> None:
    """Incrementally update and persist the vector index."""

    documents = load_documents()
    chunks = chunk_documents(documents)

    if VECTOR_STORE_FILE.exists():
        existing_chunks = load_vector_store()
    else:
        existing_chunks = []

    chunks = update_embeddings(
        client,
        chunks,
        existing_chunks,
    )

    save_vector_store(chunks)

    print(f"Indexed {len(chunks)} chunks into {VECTOR_STORE_FILE}")


## This function orchestrates the process of querying the vector store with a user's question.
def query_vector_store(
    client: OpenAI,
    question: str,
) -> None:
    """Query the existing vector store."""

    chunks = load_vector_store()

    query_embedding = create_query_embedding(
        client,
        question,
    )

    results = search_vector_store(
        chunks,
        query_embedding,
        top_k=3,
    )

    print(f"\nQuestion: {question.strip()}")

    for chunk in results:
        print("\n" + "=" * 60)
        print(f"Similarity : {chunk['similarity']:.3f}")
        print(f"Source     : {chunk['source']}")
        print(f"Chunk      : {chunk['chunk_id']}")
        print()
        print(chunk["content"])


def create_content_hash(content: str) -> str:
    """Create a stable hash representing the chunk content."""

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


QUESTION = """
Has the customer agreed to the September 30 production date?
"""


## This is the main entry point of the script. It loads the environment variables, initializes
## the argument it takes is "index" to index the documents, otherwise it queries the vector store with a predefined question.
def main() -> None:
    load_environment()

    client = OpenAI()

    if len(sys.argv) > 1 and sys.argv[1] == "index":
        index_documents(client)
        return

    query_vector_store(
        client,
        QUESTION,
    )


if __name__ == "__main__":
    main()
