# 03 - How LLMs work?

## Objective

Understand how LLMs are working.

## Concepts

# Tokenization

Tokenization is the process of converting raw input text into a sequence of discrete tokens that can be processed by a language model. Depending on the tokenizer, tokens may represent whole words, subwords, individual characters, or byte sequences.

Tokenization is the translation layer between human language and AI. It converts text into numerical token IDs that the model can understand. The tokenizer is deterministic and context-independent—it always produces the same token sequence for the same input.

Key characteristics:
- Converts text into tokens
- Produces integer token IDs
- Context-independent
- Deterministic
- Performed before any neural computation
- Part of the model's architecture (fixed after training)

# Embeddings

An embedding is a dense numerical vector that represents a token or other data item in a continuous vector space, where semantically similar items are located close to one another. Embeddings enable machine learning models to operate on meaningful numerical representations instead of discrete symbols.

Embeddings convert token IDs into high-dimensional numerical vectors that capture semantic relationships. They provide the initial semantic representation used by the Transformer, allowing similar concepts to occupy nearby regions of vector space.

Key characteristics
- Dense floating-point vectors
- Learned during model training
- High-dimensional (hundreds or thousands of values)
- Capture semantic similarity
- Serve as the starting point for the Transformer
- Fixed during inference

Tokenization answers "What symbols make up this text?"
Embeddings answer "What do those symbols initially mean?"
The Transformer answers "What do those symbols mean in this specific context?"

# Attention

Attention is the mechanism that allows a model to determine which tokens in a sequence are most relevant to one another.

After tokenization and the initial embedding step, every token has its own numerical representation. At this stage, however, each token is still represented independently and does not yet include information from the surrounding context.

Attention changes this by allowing every token to compare itself with other tokens in the sequence and assign them different importance weights.

For each token, the model creates three learned representations:

- **Query** — what information the token is looking for
- **Key** — what kind of information the token can be matched against
- **Value** — the information the token contributes if it receives attention

The Query vector of one token is compared with the Key vectors of the other tokens. These comparisons produce attention scores. The scores are normalized using softmax so that they become attention weights whose total is approximately 1.

The token then receives a weighted combination of the Value vectors from the other tokens.

## Overview of the mathematical formula used
Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V
- QKᵀ calculates how relevant the tokens are to one another.
- Division by √dₖ keeps the scores numerically stable.
- softmax converts the scores into normalized attention weights.
- Multiplication by V produces the contextualized representations.

Conceptually, attention follows the following steps:

Initial token embeddings
        ↓
Create Query, Key and Value vectors
        ↓
Compare Queries with Keys
        ↓
Calculate attention weights
        ↓
Combine Value vectors using those weights
        ↓
Contextual token representations

Key Characteristics:
- Attention allows tokens to exchange information with other tokens in the sequence.
- It transforms initial token embeddings into contextual representations.
- The same token can receive a different contextual representation in different sentences.
- Query, Key and Value vectors are derived from the token embeddings using learned projection matrices.
- Queries and Keys determine relevance.
- Values contain the information that is transferred.
- Attention weights are calculated dynamically for every input.
- Attention weights are normalized using softmax.
- Each token can attend to itself as well as other tokens.
- In self-attention, Queries, Keys and Values originate from the same sequence.
- The projection matrices are learned during model training and remain fixed during inference.
- Attention does not update the model during a conversation; it computes new contextual representations using the existing trained weights.
- GPT-style models use causal masking so that a token cannot attend to future tokens during next-token prediction.
- A single attention mechanism captures one pattern of relationships; multi-head attention allows the model to learn several relationship patterns in parallel.

# RAG - Retrival Augmented Generation

The pipeline looks something like:

Question
     │
     ▼
Tokenizer
     │
     ▼
Token IDs
     │
     ▼
Embedding Layer
     │
     ▼
Transformer
     │
     ▼
Generated answer

## Questions I had

### Is tokenization context aware?

No.

Tokenization is deterministic and context-independent.

Context is introduced later by the Transformer.

---

### Are embeddings updated during conversations?

No.

Embeddings are learned during model training.

During inference they are fixed.

Only contextual representations change.

---

### Is the embedding layer the Transformer?

No.

The embedding layer produces the initial vector.

The Transformer refines that vector using context.

## Some valuable references to read about Embeddings:

https://jalammar.github.io/illustrated-word2vec/
https://openai.com/index/introducing-text-and-code-embeddings/
https://arxiv.org/abs/1901.09069