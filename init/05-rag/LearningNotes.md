# RAG — Learning Notes

## 1. Retrieval-Augmented Generation (RAG)

### Definition / Explanation

Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with an LLM.

Instead of asking the model to answer only from its training knowledge, the application first retrieves relevant information from an external knowledge base and provides that information to the model as context.

The model then generates an answer grounded in the retrieved information.

A basic RAG pipeline is:

Documents
→ Chunking
→ Embeddings
→ Retrieval
→ Context Augmentation
→ LLM
→ Grounded Answer

### Key Characteristics

- External information is retrieved before the LLM generates an answer.
- The LLM does not need to have seen the information during training.
- The knowledge base can contain private or domain-specific information.
- Retrieved information becomes additional context for the model.
- The quality of the final answer depends heavily on the quality of retrieval.
- RAG can reduce unsupported answers by instructing the model to answer only from retrieved context.


## 2. Document Chunking

### Definition / Explanation

Chunking divides larger documents into smaller units of information that can be independently embedded and retrieved.

A useful chunk should represent a meaningful unit of information rather than simply being below a particular character or token limit.

In this project, Markdown headings are used as semantic boundaries.

### Key Characteristics

- Large chunks preserve more context but may contain several unrelated concepts.
- Small chunks improve retrieval precision but may lose important surrounding context.
- Chunk boundaries can be based on document structure, size, tokens, paragraphs, or semantic meaning.
- Different document types may require different chunking strategies.
- Metadata such as source document and chunk ID should be preserved.


## 3. Embeddings in RAG

### Definition / Explanation

Each document chunk is converted into an embedding: a numerical vector representing the semantic meaning of the text.

The user's question is converted into an embedding using the same embedding model.

This allows the application to compare the semantic meaning of the question with the semantic meaning of each document chunk.

### Key Characteristics

- Document chunks and queries must be embedded into the same vector space.
- Semantically related text tends to produce vectors pointing in similar directions.
- Embeddings are created before retrieval.
- Each stored chunk can be represented as:

  text + metadata + embedding

- A vector database is not required for small datasets; embeddings can be stored and searched directly in memory.


## 4. Semantic Retrieval

### Definition / Explanation

Semantic retrieval finds chunks whose embeddings are most similar to the embedding of the user's question.

In this project, cosine similarity is used to compare vectors.

The chunks are ranked by similarity score and the most relevant chunks are selected.

### Key Characteristics

- Retrieval is based on semantic similarity rather than exact keyword matching.
- Each query embedding is compared against the stored chunk embeddings.
- Higher cosine similarity generally indicates greater semantic relevance.
- `top_k` determines how many chunks are passed to the next stage.
- Retrieval itself does not answer the user's question; it identifies likely sources of relevant information.


## 5. Context Augmentation and Generation

### Definition / Explanation

After retrieval, the selected chunks are added to the prompt as context for the LLM.

The LLM receives both the original question and the retrieved evidence and generates an answer based on that information.

This is the "Augmented Generation" part of RAG.

### Key Characteristics

- Only relevant chunks need to be sent to the LLM.
- The prompt can instruct the model to use only the retrieved context.
- The model can synthesize information from several different chunks or documents.
- The model should be instructed to acknowledge when the retrieved context is insufficient.
- Source metadata can be preserved to make answers traceable to their original documents.


## 6. The RAG Pipeline Built in This Exercise

The implementation in `01_basic_rag.py` follows this flow:

1. Load Markdown documents.
2. Split documents into semantic chunks.
3. Create an embedding for every chunk.
4. Create an embedding for the user's question.
5. Calculate cosine similarity between the question and every chunk.
6. Rank chunks by similarity.
7. Select the top relevant chunks.
8. Add those chunks to the LLM prompt as context.
9. Ask the LLM to answer using only the retrieved information.

The implementation deliberately uses Python, NumPy, and the OpenAI API without a RAG framework or vector database so that each part of the pipeline remains visible.