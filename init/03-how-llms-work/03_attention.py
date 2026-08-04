import numpy as np


TOKENS = ["The", "animal", "was", "tired"]

# These are deliberately tiny educational embeddings.
# Real LLM embeddings contain hundreds or thousands of dimensions.
EMBEDDINGS = np.array(
    [
        [1.0, 0.0, 1.0, 0.0],  # The
        [0.0, 2.0, 0.0, 1.0],  # animal
        [1.0, 1.0, 0.0, 0.0],  # was
        [0.0, 1.0, 2.0, 1.0],  # tired
    ],
    dtype=float,
)

# Learned projection matrices in a real Transformer.
# Here we define small fixed matrices so we can inspect the calculations.
W_QUERY = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
    ]
)

W_KEY = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.5, 0.5],
    ]
)

W_VALUE = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 0.5],
    ]
)


def softmax(values: np.ndarray) -> np.ndarray:
    """
    Convert arbitrary scores into probabilities that add up to 1.

    Subtracting the maximum value improves numerical stability.
    """
    shifted_values = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted_values)

    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def calculate_attention(
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate scaled dot-product self-attention.

    Returns:
        queries
        keys
        values
        attention_weights
        contextual_embeddings
    """
    queries = embeddings @ W_QUERY
    keys = embeddings @ W_KEY
    values = embeddings @ W_VALUE

    key_dimension = keys.shape[1]

    attention_scores = queries @ keys.T
    scaled_attention_scores = attention_scores / np.sqrt(key_dimension)

    attention_weights = softmax(scaled_attention_scores)

    contextual_embeddings = attention_weights @ values

    return queries, keys, values, attention_weights, contextual_embeddings


def print_matrix(
    title: str,
    matrix: np.ndarray,
    row_labels: list[str] | None = None,
) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)

    for index, row in enumerate(matrix):
        label = f"{row_labels[index]:<10}" if row_labels else ""
        formatted_values = "  ".join(f"{value:8.4f}" for value in row)
        print(f"{label}{formatted_values}")


def print_attention_weights(attention_weights: np.ndarray) -> None:
    print(f"\n{'=' * 60}")
    print("Attention weights")
    print("=" * 60)

    header = f"{'From / To':<12}" + "".join(f"{token:>12}" for token in TOKENS)
    print(header)

    for token, weights in zip(TOKENS, attention_weights):
        formatted_weights = "".join(f"{weight:12.4f}" for weight in weights)
        print(f"{token:<12}{formatted_weights}")

    print(
        "\nEach row shows how much one token attends to every token, including itself."
    )


def print_strongest_connections(attention_weights: np.ndarray) -> None:
    print(f"\n{'=' * 60}")
    print("Strongest attention connection for each token")
    print("=" * 60)

    for source_index, source_token in enumerate(TOKENS):
        target_index = int(np.argmax(attention_weights[source_index]))
        target_token = TOKENS[target_index]
        weight = attention_weights[source_index, target_index]

        print(f"{source_token:<10} -> {target_token:<10} attention={weight:.4f}")


def main() -> None:
    (
        queries,
        keys,
        values,
        attention_weights,
        contextual_embeddings,
    ) = calculate_attention(EMBEDDINGS)

    print_matrix(
        "Initial token embeddings",
        EMBEDDINGS,
        TOKENS,
    )

    print_matrix(
        "Query vectors",
        queries,
        TOKENS,
    )

    print_matrix(
        "Key vectors",
        keys,
        TOKENS,
    )

    print_matrix(
        "Value vectors",
        values,
        TOKENS,
    )

    print_attention_weights(attention_weights)

    print_matrix(
        "Contextual embeddings after attention",
        contextual_embeddings,
        TOKENS,
    )

    print_strongest_connections(attention_weights)


if __name__ == "__main__":
    main()
