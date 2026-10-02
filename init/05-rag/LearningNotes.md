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

## 7. Similarity Thresholds

### Definition / Explanation

A similarity threshold can be used to reject retrieved chunks whose
embedding similarity is below a chosen value.

This can prevent obviously unrelated chunks from being passed to the
LLM. However, cosine similarity is not a confidence score, and there is
no universal threshold that separates relevant from irrelevant content.

In this exercise, a threshold of 0.50 successfully rejected irrelevant
results for one query, but also would have rejected highly relevant
information for another differently phrased query.

### Key Characteristics

- Similarity thresholds can remove obviously weak retrieval results.
- Cosine similarity should not be interpreted as a probability or
  confidence percentage.
- A threshold that works for one dataset or query may fail for another.
- Relevant information can have relatively low embedding similarity.
- Thresholds should be evaluated and tuned rather than treated as
  universal constants.


## 8. Reranking

### Definition / Explanation

Reranking is a second retrieval stage that evaluates candidate chunks
more carefully based on how useful they are for answering a specific
question.

Embedding similarity can efficiently identify potentially relevant
chunks, but semantic similarity does not necessarily mean that a chunk
contains the information needed to answer the question.

A reranker evaluates the question together with each candidate chunk
and produces a more precise relevance ranking.

### Key Characteristics

- Vector retrieval is useful for finding candidate chunks efficiently.
- High embedding similarity does not guarantee high answer relevance.
- A relevant chunk may have a relatively low embedding similarity score.
- Initial retrieval should prioritize recall: finding potentially useful information.
- Reranking should prioritize precision: identifying which candidate actually help answer the question.
- Reranking cannot recover a relevant chunk that was not included in the candidate set.
- Multiple candidates can be reranked in a single model request.


## 9. Query Expansion

### Definition / Explanation

Query expansion generates alternative search queries that represent the
same user intent using different terminology or perspectives.

The original question and the expanded queries can each be embedded and
used for retrieval.

This increases the probability that relevant chunks are discovered even
when the user's wording differs significantly from the wording used in
the source documents.

### Key Characteristics

- One user question can be represented by several search queries.
- Expanded queries can introduce terminology likely to appear in the source documents.
- Each query can retrieve a different set of candidate chunks.
- Query expansion improves retrieval recall.
- It is especially useful when the user's language differs from the
  terminology used in the knowledge base.
- Expanded queries are generated dynamically and may differ between executions.
- Query expansion increases retrieval work because multiple searches
  are performed for one user question.


## 10. Candidate Merging

### Definition / Explanation

When multiple queries are used for retrieval, the same document chunk
may be retrieved several times.

Candidate merging combines these retrieval results into a single set of
unique chunks before reranking.

In this exercise, chunks are identified by their `chunk_id`. If the same
chunk is retrieved by several queries, the highest observed similarity
score is retained.

### Key Characteristics

- Duplicate chunks should not be sent repeatedly to the reranker.
- `chunk_id` provides a stable identifier for deduplication.
- A chunk may be discovered by both the original query and several
  expanded queries.
- Keeping the maximum similarity score is a simple merging strategy.
- Other strategies could also consider retrieval rank, average score,
  or how many queries retrieved the chunk.
- Candidate merging reduces unnecessary context and reranking work.


## 11. Recall and Precision

### Definition / Explanation

Recall and precision describe two different goals within the retrieval
pipeline.

Recall is concerned with finding the relevant information in the first
place.

Precision is concerned with ensuring that the retrieved information is
actually useful for answering the question.

The retrieval pipeline can therefore use different stages optimized for
different purposes.

### Key Characteristics

- Candidate retrieval should prioritize recall.
- Query expansion can improve recall by searching from multiple semantic perspectives.
- Reranking improves precision by evaluating candidate usefulness more carefully.
- A reranker cannot evaluate information that candidate retrieval failed to find.
- A strong RAG pipeline therefore needs both good candidate retrieval
  and good candidate selection.


## 12. Improved Retrieval Pipeline

The retrieval pipeline implemented in `02_improved_retrieval.py` is:

User Question
→ Query Expansion
→ Embed Original and Expanded Queries
→ Retrieve Top Candidates for Each Query
→ Merge and Deduplicate Candidates
→ Rerank Candidates
→ Select Best Evidence

The exercise demonstrated that embedding similarity alone is not enough
to determine whether a chunk can answer a question.

A chunk with relatively low similarity to the original question can
still contain the best answer. Query expansion can help retrieve that
chunk, while reranking can identify it as the most useful evidence.