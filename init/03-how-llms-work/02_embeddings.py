from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_MODEL = "text-embedding-3-small"

TEXTS = [
    "The dog is playing in the garden.",
    "A puppy is running after a ball.",
    "The cat is sleeping on the sofa.",
    "A wolf is hunting in the forest.",
    "The electric car needs to be charged.",
    "The vehicle battery has a long range.",
    "The truck transported goods across Germany.",
    "The automobile uses an electric motor.",
    "The software deployment failed in production.",
    "The API returned an authentication error.",
    "The application crashed after the update.",
    "The server experienced a network timeout.",
]

CLASS_LABELS = {
    "animal": "Animals such as dogs, cats, wolves and other living creatures.",
    "automotive": "Cars, vehicles, batteries, trucks and transportation.",
    "software": "Software systems, APIs, applications, servers and technical errors.",
}


def main() -> None:
    load_environment()

    client = OpenAI()

    embeddings = create_embeddings(client, TEXTS)

    print_embedding_information(embeddings)
    demonstrate_similarity(embeddings)
    demonstrate_clustering(embeddings)
    visualize_embeddings(embeddings)
    demonstrate_classification(client)


def load_environment() -> None:
    # Load environment variables from the repository-level .env file.
    repo_root = Path(__file__).resolve().parents[2]
    env_file = repo_root / ".env"
    if not env_file.exists():
        raise FileNotFoundError(f".env file not found at {env_file}")

    load_dotenv(env_file)


def create_embeddings(client: OpenAI, texts: list[str]) -> np.ndarray:
    """Create one embedding vector for every supplied text."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    return np.array([item.embedding for item in response.data])


def print_embedding_information(embeddings: np.ndarray) -> None:
    print("\n========================================")
    print("Embedding Information")
    print("========================================")
    print(f"Number of texts      : {embeddings.shape[0]}")
    print(f"Embedding dimensions : {embeddings.shape[1]}")
    print(f"Matrix shape         : {embeddings.shape}")
    print(f"First five values    : {embeddings[0][:5]}")


def demonstrate_similarity(embeddings: np.ndarray) -> None:
    """Compare every text with the first text using cosine similarity."""
    similarities = cosine_similarity(
        embeddings[0].reshape(1, -1),
        embeddings,
    )[0]

    ranked_indices = np.argsort(similarities)[::-1]

    print("\n========================================")
    print("Semantic Similarity")
    print("========================================")
    print(f'Query: "{TEXTS[0]}"\n')

    for index in ranked_indices:
        print(f"{similarities[index]:.4f} | {TEXTS[index]}")


def demonstrate_clustering(embeddings: np.ndarray) -> np.ndarray:
    """Group texts based only on their embedding vectors."""
    number_of_clusters = 3

    model = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10,
    )

    cluster_ids = model.fit_predict(embeddings)

    print("\n========================================")
    print("Clustering")
    print("========================================")

    for cluster_id in range(number_of_clusters):
        print(f"\nCluster {cluster_id}")

        for text, assigned_cluster in zip(TEXTS, cluster_ids):
            if assigned_cluster == cluster_id:
                print(f"  - {text}")

    return cluster_ids


def visualize_embeddings(embeddings: np.ndarray) -> None:
    """
    Reduce high-dimensional embeddings to two dimensions with PCA
    and display them on a scatter plot.
    """
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(embeddings)

    plt.figure(figsize=(11, 7))
    plt.scatter(coordinates[:, 0], coordinates[:, 1])

    for index, text in enumerate(TEXTS):
        short_label = shorten_text(text)

        plt.annotate(
            short_label,
            (coordinates[index, 0], coordinates[index, 1]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.title("Two-dimensional projection of text embeddings")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.tight_layout()
    plt.show()


def shorten_text(text: str, maximum_length: int = 35) -> str:
    if len(text) <= maximum_length:
        return text

    return text[: maximum_length - 3] + "..."


def demonstrate_classification(client: OpenAI) -> None:
    """
    Perform zero-shot classification by comparing a new text embedding
    with embeddings representing the available class descriptions.
    """
    text_to_classify = "The car cannot start because its battery is empty."

    labels = list(CLASS_LABELS.keys())
    label_descriptions = list(CLASS_LABELS.values())

    label_embeddings = create_embeddings(client, label_descriptions)
    text_embedding = create_embeddings(client, [text_to_classify])

    similarities = cosine_similarity(
        text_embedding,
        label_embeddings,
    )[0]

    predicted_index = int(np.argmax(similarities))
    predicted_label = labels[predicted_index]

    print("\n========================================")
    print("Zero-shot Classification")
    print("========================================")
    print(f"Text             : {text_to_classify}")
    print(f"Predicted class  : {predicted_label}\n")

    for label, score in zip(labels, similarities):
        print(f"{label:<12}: {score:.4f}")


if __name__ == "__main__":
    main()
