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