from pathlib import Path
import re
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

RESOURCES_DIR = Path(__file__).parent / "resources"


def load_documents() -> list[dict]:
    """Load all Markdown documents from the resources directory."""

    documents = []

    for file_path in sorted(RESOURCES_DIR.glob("*.md")):
        content = file_path.read_text(encoding="utf-8")

        documents.append(
            {
                "source": file_path.name,
                "content": content,
            }
        )

    return documents


## This function prepares the documents for RAG by splitting them into chunks based on Markdown section headings.
##.    documents
##        ↓
##     chunking
##        ↓
##  many smaller pieces of text
##        ↓
##  each chunk remembers its source
##
## What we have in our .md docs is quite easy to chunk, as we are using Markdown delimited by headings. In other cases,
## we might want to use a more sophisticated chunking strategy, such as splitting by paragraphs or sentences, or using a library like NLTK or spaCy for natural language processing.
## Chunking isn't merely chopping text into N-character pieces. It's deciding what units of information should be independently retrievable.
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
                }
            )

    return chunks


## Function for creating embeddings for each chunk. This is a crucial step in RAG, as it allows us to represent the text in a vector space, enabling efficient similarity searches later on.
def create_embeddings(
    client: OpenAI,
    chunks: list[dict],
) -> list[dict]:
    """Create an embedding for every document chunk."""

    texts = [chunk["content"] for chunk in chunks]

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )

    for chunk, embedding_data in zip(chunks, response.data):
        chunk["embedding"] = embedding_data.embedding

    return chunks


## Function for calculating cosine similarity between two vectors. This is a common metric used in information retrieval to measure the similarity between two pieces of text represented as vectors.
def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""

    a = np.array(vector_a)
    b = np.array(vector_b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


## Function for creating an embedding for the user's question. This allows us to compare the user's query with the document chunks in the same vector space.
## Mind that it is using the same embedding model as for the document chunks, which is important for consistency in the vector space.
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


## Function for retrieving the most relevant chunks based on the user's query. It calculates the similarity between the query embedding and each chunk's embedding, sorts them, and returns the top K most similar chunks.
## By default, it returns the top 3 most similar chunks, but this can be adjusted with the `top_k` parameter.
def retrieve_chunks(
    chunks: list[dict],
    query_embedding: list[float],
    top_k: int = 3,
) -> list[dict]:
    """Return the chunks most similar to the query."""

    scored_chunks = []

    for chunk in chunks:
        score = cosine_similarity(
            query_embedding,
            chunk["embedding"],
        )

        scored_chunks.append(
            {
                **chunk,
                "similarity": score,
            }
        )

    scored_chunks.sort(
        key=lambda chunk: chunk["similarity"],
        reverse=True,
    )

    return scored_chunks[:top_k]


def generate_answer(
    client: OpenAI,
    question: str,
    retrieved_chunks: list[dict],
) -> str:
    """Generate an answer using only the retrieved context."""

    context = "\n\n---\n\n".join(
        f"Source: {chunk['source']}\n{chunk['content']}" for chunk in retrieved_chunks
    )

    prompt = f"""
Answer the question using only the context provided below.

If the context does not contain enough information to answer the question,
say that you do not have enough information.

QUESTION:
{question}

CONTEXT:
{context}
"""

    response = client.responses.create(
        model="gpt-5.5",
        input=prompt,
    )

    return response.output_text


def load_environment() -> None:
    """Load the repository-level environment file."""

    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")


# USER_QUESTION = """
# Has the customer agreed to the September 30 production date?
# """

USER_QUESTION = """
What is the project's approved budget?
"""


def main() -> None:
    load_environment()

    client = OpenAI()

    documents = load_documents()
    chunks = chunk_documents(documents)
    chunks = create_embeddings(client, chunks)

    query_embedding = create_query_embedding(
        client,
        USER_QUESTION,
    )

    retrieved_chunks = retrieve_chunks(
        chunks,
        query_embedding,
        top_k=3,
    )

    print(f"\nQuestion: {USER_QUESTION.strip()}")
    print("\nTop retrieved chunks:")

    for chunk in retrieved_chunks:
        print("\n" + "=" * 60)
        print(f"Source     : {chunk['source']}")
        print(f"Chunk      : {chunk['chunk_id']}")
        print(f"Similarity : {chunk['similarity']:.3f}")
        print()
        print(chunk["content"])

    answer = generate_answer(
        client,
        USER_QUESTION,
        retrieved_chunks,
    )

    print("\n" + "=" * 60)
    print("Generated answer")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()
