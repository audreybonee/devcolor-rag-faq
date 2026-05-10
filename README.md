# Simple Command Line RAG System

This is a simple Command Line RAG system built to answer questions based on the `/dev/color` FAQ corpus using OpenAI embeddings for semantic search, and GPT-4o-mini for response generation.

---

## How It Works

Each question in `devcolorfaq.txt` is treated as a single entry (chunk). At startup, the system embeds every chunk using OpenAI's `text-embedding-3-small`. When a user submits a query, the system:

1. Embeds the query using the same embedding model
2. Checks a semantic cache — if a near-identical question was already answered (cosine similarity ≥ 0.85), it returns the cached response instantly
3. Performs cosine similarity search across all corpus embeddings and retrieves the top-3 most relevant chunks above a threshold of 0.4
4. Sends those chunks as context to `gpt-4o-mini`, which generates a grounded answer
5. Caches the result for future similar queries

If no chunks pass the similarity threshold, the system responds that it does not have information on that topic rather than hallucinating.

---

## Setup

**Requirements:** Python 3.9+, an OpenAI API key

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create a .env file with your OpenAI API key
OPENAI_API_KEY=sk-your-key-here

# 3. Run the chatbot
python main.py
```

Type `exit`, `quit`, or `q` to end the session.

---

## Project Structure

```
devcolor-rag-faq/
├── main.py           # CLI entry point and interactive loop
├── rag.py            # RAGSystem — orchestrates the full query pipeline
├── retriever.py      # Embeds corpus at startup, in-memory cosine similarity search
├── chunker.py        # Parses devcolorfaq.txt into Q&A chunks
├── cache.py          # In-memory semantic cache for repeated queries
├── devcolorfaq.txt   # FAQ corpus (10 Q&A entries)
└── requirements.txt
```

---

## Sample Output

```
Initializing RAG System... Please wait.
Initializing retriever (embedding corpus)...

==================================================
Welcome to the /dev/color FAQ Assistant.
Ask me anything about /dev/color.
Type 'exit', 'quit', or 'q' to quit.
==================================================

You: How can /dev/color help me develop my career?

Assistant: /dev/color can help you develop your career by providing access to a strong community of Black technologists, offering mentorship opportunities, and providing career development resources. You can engage in programs like the A* Program, which helps you set and achieve career goals through peer support, networking, and skill-building.
(Response time: 2.15s)

You: In which cities is /dev/color located?

Assistant: /dev/color has hosted events in major cities including San Francisco, New York, Atlanta, and Seattle.  However, the context does not specify if these are the only locations or if there are additional locations where /dev/color operates.
(Response time: 1.08s)

You: How can I contribute to devcolor?

Assistant: You can contribute to /dev/color by making financial contributions, participating in mentorship programs, and advocating for diversity in tech.
(Response time: 1.42s)

You: How can I help support /dev/color?

Assistant: You can contribute to /dev/color by making financial contributions, participating in mentorship programs, and advocating for diversity in tech.
[Source: Semantic Cache]
(Response time: 0.17s)

You: What is the capital of France?

Assistant: I don't have information about that in the /dev/color FAQ.
(Response time: 0.85s)

You: quit
Exiting...
```

> **Note:**  Response times are representative of a real run and may vary due to API latency. The corpus is re-embedded on each startup because this time-boxed version uses in-memory retrieval rather than persistent vector storage.
