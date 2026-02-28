# Query Translation & Advanced RAG Techniques — Complete Notes

> These notes cover how to **improve RAG accuracy** beyond the basic Naive RAG pipeline (Rag_1.py). The core idea: the user's query is often the weakest link — fix that, and everything downstream gets better.

---

## Table of Contents

**Part 1 — The 6 RAG Technique Areas:**
1. [RAG Techniques Overview — The Big Picture](#1-rag-techniques-overview--the-big-picture)

**Part 2 — Query Translation (The Core):**
2. [What is Query Translation?](#2-what-is-query-translation)
3. [Multi-Query & RAG Fusion (Fan-Out Retrieval)](#3-multi-query--rag-fusion-fan-out-retrieval)
4. [Reciprocal Rank Fusion (RRF) — Ranking Chunks](#4-reciprocal-rank-fusion-rrf--ranking-chunks)
5. [Query Decomposition — Breaking the Query](#5-query-decomposition--breaking-the-query)
6. [Step-Back Prompting — Going Abstract](#6-step-back-prompting--going-abstract)
7. [Few-Shot Prompting in Query Translation](#7-few-shot-prompting-in-query-translation)
8. [HyDE — Hypothetical Document Embeddings](#8-hyde--hypothetical-document-embeddings)

**Part 3 — The Other RAG Techniques:**
9. [Routing — Sending Queries to the Right Place](#9-routing--sending-queries-to-the-right-place)
10. [Query Construction — Translating to DB Language](#10-query-construction--translating-to-db-language)
11. [Indexing — How You Store Matters](#11-indexing--how-you-store-matters)
12. [Retrieval — Getting the Right Chunks](#12-retrieval--getting-the-right-chunks)
13. [Generation — The Final Answer](#13-generation--the-final-answer)

**Cheat Sheet:**
14. [Quick Revision Cheat Sheet](#14-quick-revision-cheat-sheet)

---

## 1. RAG Techniques Overview — The Big Picture

### The Problem with Naive RAG (What We Built in Rag_1.py)

In our basic RAG:
```
User Query → Embed → Search Vector DB → Top K Chunks → LLM → Answer
```

This works **okay**, but has issues:
- User asks vague/ambiguous questions → BAD retrieval
- Irrelevant chunks get retrieved → Hallucinations
- No ranking of chunks → Noise goes to LLM
- One-size-fits-all pipeline → Doesn't adapt to query type

### The 6 RAG Technique Areas

To improve RAG accuracy, we optimize at **6 different stages** of the pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ADVANCED RAG TECHNIQUES                         │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  1. QUERY TRANSLATION                                        │   │
│  │     Rewrite/improve the user's question before searching     │   │
│  │     → Multi-Query, RAG Fusion, Decomposition, Step-Back,     │   │
│  │       HyDE                                                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                            ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  2. ROUTING                                                  │   │
│  │     Send the query to the RIGHT data source                  │   │
│  │     → Logical routing, Semantic routing                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                            ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  3. QUERY CONSTRUCTION                                       │   │
│  │     Convert natural language → database query language        │   │
│  │     → Text-to-SQL, Text-to-Cypher, Text-to-Metadata          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                            ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  4. INDEXING                                                 │   │
│  │     How you store your data affects retrieval quality         │   │
│  │     → Chunking strategies, Multi-representation, Hierarchy    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                            ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  5. RETRIEVAL                                                │   │
│  │     How you search and rank results                          │   │
│  │     → Re-ranking, Hybrid search, CRAG (Corrective RAG)       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                            ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  6. GENERATION                                               │   │
│  │     How the LLM uses the retrieved context to answer          │   │
│  │     → Self-RAG, Hallucination check, Citation                │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

> **Key insight:** You don't NEED all 6 techniques. Pick what your use case needs. For a simple coding chatbot, basic RAG is fine. For **legal documents or medical records**, you need multiple layers of optimization.

---

## 2. What is Query Translation?

### First — What is "Query" and What is "Translation"?

| Word | Meaning |
|------|---------|
| **Query** | The user's question/prompt — whatever they type into the search/chat box |
| **Translation** | Transforming that query into something **better** — rewriting, expanding, breaking down, or abstracting it |

**In simple terms:**
- **Query** = What the user TYPED
- **Translation** = Converting what they TYPED into what they ACTUALLY MEANT

### The Real-World Analogy — The Song Example 🎵

```
You: *hums a tune* "That song that goes like... da da da daaaa..."
Friend: "Oh! You mean Beethoven's 5th Symphony!"
         ↓
Your INPUT was vague (just a tune)
Your friend TRANSLATED it into the actual thing you wanted
```

That's exactly what Query Translation does — your vague input gets **translated** into precise search queries.

### The Core Problem: Ambiguous Input → Ambiguous Output

```
┌──────────────────────────────────────────────────────┐
│                                                      │
│    Garbage In  →  Garbage Out                        │
│    ──────────     ────────────                       │
│                                                      │
│    Vague query  →  Irrelevant chunks  →  Bad answer  │
│    Good query   →  Relevant chunks    →  Good answer │
│                                                      │
│    The QUERY is the weakest link in RAG!             │
│                                                      │
└──────────────────────────────────────────────────────┘
```

If the user doesn't know what they want → they can't ask the right question → the vector DB returns bad chunks → the LLM generates a bad answer.

> **Hence:** Based on the user's query, we can **optimize RAG** by improving the query BEFORE it hits the vector database.

### The Two Sides of a Query

There are always **two things** happening:

```
┌────────────────┐              ┌────────────────┐
│  What the user │              │  What the user │
│  ASKS          │      ≠       │  ACTUALLY WANTS│
│                │              │                │
│  "auth stuff"  │              │  "How does JWT │
│                │              │   token-based  │
│                │              │   authentication│
│                │              │   work with    │
│                │              │   refresh tokens│
│                │              │   in the API?" │
└────────────────┘              └────────────────┘
```

**Query Translation = Bridge the gap between what user ASKS and what user WANTS.**

### The Abstraction Spectrum

Every user question sits somewhere on a spectrum:

```
More Abstract (Vague)                            Less Abstract (Specific)
←─────────────────────────────────────────────────────────────────────→
"Tell me about                "How does auth    "What is the JWT
 security stuff"               work?"             refresh token
                                                   rotation logic
                                                   in /api/auth
                                                   endpoint?"
     ↑                            ↑                    ↑
  Too vague                   Sweet spot            Too narrow
  → Gets random              → Gets relevant        → Might miss
    chunks                      chunks                 related info
```

**The ideal query is in the MIDDLE** — specific enough to retrieve relevant chunks, but broad enough to not miss important context.

But technically **BOTH ends are necessary:**
- **More abstract** → Gives you the BIG PICTURE, general understanding
- **Less abstract** → Gives you the PRECISE DETAIL you need

> **That's why we use techniques like Step-Back (go abstract) AND Chain-of-Thought decomposition (go specific) — to cover BOTH ends.**

### What We Can Do — The Query Translation Techniques

```
                    User Query
                        │
           ┌────────────┼────────────┐
           ↓            ↓            ↓
     ┌──────────┐ ┌──────────┐ ┌──────────┐
     │ Rewrite  │ │ Decompose│ │ Generate  │
     │          │ │          │ │ Hypothetic│
     │ Multi-   │ │ Break    │ │ al Docs   │
     │ Query /  │ │ into sub-│ │ (HyDE)    │
     │ RAG      │ │ questions│ │           │
     │ Fusion   │ │          │ │           │
     └──────────┘ └──────────┘ └──────────┘
           │            │            │
           ↓            ↓            ↓
     More Abstract  Less Abstract  Hypothetical
     (Step-Back)    (Chain of      Answer
                     Thought)
```

Let's dive into each one.

---

## 3. Multi-Query & RAG Fusion (Fan-Out Retrieval)

### What Is It?

Instead of searching with **one** query, the LLM generates **multiple different versions** of the user's question, and we search for ALL of them **in parallel**.

### The Full Architecture:

```
User Query: "How does authentication work?"
         │
         ↓
┌─────────────────────────────────────┐
│  LLM (with System Prompt)          │
│                                     │
│  System: "Given this user query,    │
│  generate 3 different versions      │
│  of this question that would help   │
│  retrieve relevant documents"       │
│                                     │
│  Output:                            │
│  Q1: "What authentication           │
│       mechanisms are used?"         │
│  Q2: "How does JWT login flow       │
│       work?"                        │
│  Q3: "What is the session           │
│       management system?"           │
└────────┬──────────┬─────────┬───────┘
         │          │         │
         ↓          ↓         ↓         ← This is the "FAN-OUT"
    ┌────────┐ ┌────────┐ ┌────────┐      (queries spread out
    │Embed Q1│ │Embed Q2│ │Embed Q3│       like a fan)
    └───┬────┘ └───┬────┘ └───┬────┘
        ↓          ↓          ↓
    ┌────────┐ ┌────────┐ ┌────────┐
    │Vector  │ │Vector  │ │Vector  │   ← Parallel similarity search
    │DB      │ │DB      │ │DB      │
    │Search  │ │Search  │ │Search  │
    └───┬────┘ └───┬────┘ └───┬────┘
        ↓          ↓          ↓
   Chunks A    Chunks B    Chunks C
    [d1,d2,     [d3,d4,     [d5,d2,
     d3]         d2]         d1]
        │          │          │
        └──────────┼──────────┘
                   ↓
         ┌─────────────────┐
         │ INTERSECTION /  │   ← Filter unique chunks
         │ UNION & RANK    │
         │ (Deduplication) │
         └────────┬────────┘
                  ↓
         Unique relevant chunks
         [d1, d2, d3, d4, d5]
                  ↓
         ┌─────────────────┐
         │ Original User   │
         │ Query + Chunks  │   ← Bring back the ORIGINAL question
         │ → LLM           │
         └────────┬────────┘
                  ↓
         More accurate response!
```

### Why "Fan-Out"?

```
"Fan-Out" = The queries SPREAD OUT like a hand fan:

         Q1
        ╱
User Query ── Q2      ← One input → Multiple outputs → spread out
        ╲
         Q3

Just like a ceiling fan has multiple blades going in different directions,
a Fan-Out sends multiple queries in different directions to cover more ground.

In tech: "Fan-Out" means one request triggers MULTIPLE parallel requests.
```

### The Key Steps Explained:

| Step | What Happens | Why |
|------|-------------|-----|
| 1. User asks question | "How does auth work?" | Starting point |
| 2. LLM rewrites | Generates 3 different versions | Each version covers a different ANGLE |
| 3. Each query → embedded | Convert to vectors | Same as normal RAG |
| 4. Parallel search | Each query hits vector DB independently | Fan-Out — all at once |
| 5. Get chunks | Each query returns its own relevant chunks | More diverse results |
| 6. Filter unique | Remove duplicate chunks (intersection/union) | d2 appeared in Q1 and Q2, keep only once |
| 7. Original query + chunks → LLM | Use the ORIGINAL user question | The rewrites were for SEARCH, not for answering |

### Why Use the Original Query at the End?

```
The rewritten queries were tools to SEARCH better.
But the answer should match what the USER actually asked.

Rewritten Q2: "How does JWT login flow work?"
But user asked: "How does auth work?" (broader)

→ So we give LLM the ORIGINAL query + ALL the diverse chunks
→ LLM answers the ORIGINAL question with BETTER context
```

### Code Example:

```python
import google.generativeai as genai

# Step 1: User query
user_query = "How does authentication work?"

# Step 2: LLM generates multiple queries
system_prompt = """You are a helpful assistant. Given the user's question, 
generate 3 different versions of this question that would help retrieve 
relevant documents from a vector database. Each version should approach 
the topic from a different angle.

Return only the 3 questions, one per line."""

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content(f"{system_prompt}\n\nUser question: {user_query}")

# Step 3: Parse generated queries
queries = response.text.strip().split("\n")
# ['What authentication mechanisms are used?', 
#  'How does JWT login flow work?', 
#  'What is the session management system?']

# Step 4: Fan-Out — search vector DB with each query
all_chunks = []
for q in queries:
    chunks = retriever.similarity_search(query=q, k=3)
    all_chunks.extend(chunks)

# Step 5: Filter unique (deduplicate by content)
unique_chunks = list({chunk.page_content: chunk for chunk in all_chunks}.values())

# Step 6: Original query + unique chunks → LLM
context = "\n\n".join([c.page_content for c in unique_chunks])
final_prompt = f"""Based on this context, answer the user's question.

Context:
{context}

Question: {user_query}"""

answer = model.generate_content(final_prompt)
```

---

## 4. Reciprocal Rank Fusion (RRF) — Ranking Chunks

### The Problem: We Have Chunks, But Which Ones Matter Most?

After Multi-Query retrieval, we have chunks from multiple queries. But some chunks are **more relevant** than others.

**The hallucination problem:**
```
We could give the ENTIRE PDF to the LLM, but that's wasteful.
We could give ALL retrieved chunks — but some are irrelevant noise.
  → Irrelevant context = LLM hallucinates or gives unfocused answers.

For example, out of 9 retrieved chunks, maybe only 3 are truly relevant.
The "blue file" (most relevant) should be ranked HIGHER.

So we must RANK the chunks by relevance, not just blindly pass them all.
```

### What is RRF?

**Reciprocal Rank Fusion** is a **ranking algorithm** that combines results from multiple ranked lists into a single, better-ranked list.

It works by looking at:
- **How often** a chunk appears across different query results (repetition)
- **Where** it appears in each list (position/rank)
- Giving **weighted scores** based on these factors

### The RRF Formula:

```
                        1
RRF_score(doc) =  Σ  ─────────
                     k + rank

Where:
  Σ     = Sum across ALL ranked lists where this document appears
  k     = Constant (usually 60) — dampens effect of very high ranks
  rank  = Position of the document in that list (1 = first, 2 = second, etc.)
```

### Worked Example:

Let's say Multi-Query gave us 3 sets of chunks:

```
chunk1 (from Q1) = [d1, d2, d3]    ← d1 is rank 1, d2 is rank 2, d3 is rank 3
chunk2 (from Q2) = [d3, d4, d2]    ← d3 is rank 1, d4 is rank 2, d2 is rank 3
chunk3 (from Q3) = [d5, d2, d1]    ← d5 is rank 1, d2 is rank 2, d1 is rank 3
```

**Calculate RRF score for each document (k=60):**

```
d1:
  In chunk1 → rank 1  →  1/(60+1) = 0.01639
  In chunk2 → not found → 0
  In chunk3 → rank 3  →  1/(60+3) = 0.01587
  ─────────────────────────────────
  RRF(d1) = 0.01639 + 0 + 0.01587 = 0.03226

d2:
  In chunk1 → rank 2  →  1/(60+2) = 0.01613
  In chunk2 → rank 3  →  1/(60+3) = 0.01587
  In chunk3 → rank 2  →  1/(60+2) = 0.01613
  ─────────────────────────────────
  RRF(d2) = 0.01613 + 0.01587 + 0.01613 = 0.04813  ← HIGHEST! 🏆

d3:
  In chunk1 → rank 3  →  1/(60+3) = 0.01587
  In chunk2 → rank 1  →  1/(60+1) = 0.01639
  In chunk3 → not found → 0
  ─────────────────────────────────
  RRF(d3) = 0.01587 + 0.01639 + 0 = 0.03226

d4:
  In chunk2 → rank 2  →  1/(60+2) = 0.01613
  ─────────────────────────────────
  RRF(d4) = 0.01613

d5:
  In chunk3 → rank 1  →  1/(60+1) = 0.01639
  ─────────────────────────────────
  RRF(d5) = 0.01639
```

**Final Ranking:**
```
1st: d2 (0.04813) ← Appears in ALL 3 lists! Most broadly relevant
2nd: d1 (0.03226) ← Appears in 2 lists
2nd: d3 (0.03226) ← Appears in 2 lists (tie with d1)
4th: d5 (0.01639) ← Only in 1 list but at rank 1
5th: d4 (0.01613) ← Only in 1 list at rank 2
```

> **Key Insight:** d2 wins because it's relevant across ALL three different query angles — this means it's **broadly important**, not just relevant to one specific phrasing.

### The Python Code — Line by Line Explanation:

```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    """
    Performs Reciprocal Rank Fusion (RRF) on a list of ranked lists.

    Args:
        ranked_lists (list of lists): Each inner list is a ranked list 
                                       of document IDs (strings/integers).
        k (int): A constant to dampen the effect of very high ranks.
                 (prevents rank 1 from dominating too much)

    Returns:
        list: A list of (document_id, rrf_score) tuples, sorted by 
              RRF score (descending).
    """
    fused_scores = {}  # Dictionary to accumulate scores: {doc_id: total_score}
    
    for rank_list in ranked_lists:
        # Iterate over each ranked list (e.g., results from Q1, Q2, Q3)
        
        for rank, doc_id in enumerate(rank_list, start=1):
            # enumerate gives: (1, 'doc_A'), (2, 'doc_B'), (3, 'doc_C')
            # start=1 because rank is 1-indexed (first = rank 1, not rank 0)
            
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
                # First time seeing this doc → initialize score to 0
            
            # THE CORE FORMULA: Add reciprocal rank score
            fused_scores[doc_id] += 1.0 / (k + rank)
            # If doc_A is rank 1 in this list: 1/(60+1) = 0.01639
            # If doc_A is rank 3 in this list: 1/(60+3) = 0.01587
            # Score ACCUMULATES across all lists where this doc appears
            
    # Sort all documents by their total fused score (highest first)
    sorted_results = sorted(
        fused_scores.items(),        # [('doc_A', 0.032), ('doc_B', 0.048), ...]
        key=lambda item: item[1],    # Sort by the score (item[1])
        reverse=True                 # Descending — highest score first
    )
    return sorted_results
```

### Running the Code:

```python
# Two search methods returned these ranked results:
ranked_list_1 = ['doc_A', 'doc_B', 'doc_C', 'doc_D']
ranked_list_2 = ['doc_B', 'doc_D', 'doc_E', 'doc_A']

fused_results = reciprocal_rank_fusion([ranked_list_1, ranked_list_2], k=60)

for doc_id, score in fused_results:
    print(f"Document ID: {doc_id}, RRF Score: {score:.4f}")
```

### Step-by-Step Trace:

```
Processing ranked_list_1: ['doc_A', 'doc_B', 'doc_C', 'doc_D']
  doc_A: rank=1 → score += 1/(60+1) = 0.0164  → fused_scores = {'doc_A': 0.0164}
  doc_B: rank=2 → score += 1/(60+2) = 0.0161  → fused_scores = {'doc_A': 0.0164, 'doc_B': 0.0161}
  doc_C: rank=3 → score += 1/(60+3) = 0.0159  → fused_scores = {... 'doc_C': 0.0159}
  doc_D: rank=4 → score += 1/(60+4) = 0.0156  → fused_scores = {... 'doc_D': 0.0156}

Processing ranked_list_2: ['doc_B', 'doc_D', 'doc_E', 'doc_A']
  doc_B: rank=1 → score += 1/(60+1) = 0.0164  → fused_scores = {... 'doc_B': 0.0161+0.0164 = 0.0325}
  doc_D: rank=2 → score += 1/(60+2) = 0.0161  → fused_scores = {... 'doc_D': 0.0156+0.0161 = 0.0317}
  doc_E: rank=3 → score += 1/(60+3) = 0.0159  → fused_scores = {... 'doc_E': 0.0159}
  doc_A: rank=4 → score += 1/(60+4) = 0.0156  → fused_scores = {... 'doc_A': 0.0164+0.0156 = 0.0320}

Final sorted:
  doc_B: 0.0325  ← Winner! (rank 2 in list 1 + rank 1 in list 2)
  doc_A: 0.0320  ← (rank 1 in list 1 + rank 4 in list 2)
  doc_D: 0.0317  ← (rank 4 in list 1 + rank 2 in list 2)
  doc_C: 0.0159  ← Only in list 1
  doc_E: 0.0159  ← Only in list 2
```

### Why k=60?

```
Without k (or k=0):
  rank 1: 1/(0+1) = 1.000
  rank 2: 1/(0+2) = 0.500  ← rank 1 is 2x better than rank 2!

With k=60:
  rank 1: 1/(60+1) = 0.01639
  rank 2: 1/(60+2) = 0.01613  ← rank 1 is only slightly better than rank 2

k=60 DAMPENS the difference between ranks.
Without it, rank 1 would dominate too much.
We want documents that appear across MANY lists to win,
not just one that happened to be rank 1 in one list.
```

### Complete Flow — Multi-Query + RRF Together:

```
User Query: "How does auth work?"
       │
       ↓
LLM generates 3 queries (Fan-Out)
       │
       ├── Q1 → Vector DB → [d1, d2, d3]
       ├── Q2 → Vector DB → [d3, d4, d2]
       └── Q3 → Vector DB → [d5, d2, d1]
                    │
                    ↓
            RRF Ranking Algorithm
                    │
                    ↓
            d2 > d1 = d3 > d5 > d4
                    │
                    ↓
         Top 3: [d2, d1, d3]  ← Most relevant, ranked chunks
                    │
                    ↓
         Original Query + Top Chunks → LLM → Answer
```

---

## 5. Query Decomposition — Breaking the Query

### What Is It?

Instead of asking the question as-is, **break it into smaller sub-questions**, answer each independently, then combine everything for a final answer.

### The Problem It Solves:

```
Complex question: "Compare JWT and OAuth for API authentication, 
                   and explain which is better for microservices"

This is ACTUALLY 3 questions:
  1. "What is JWT for API authentication?"
  2. "What is OAuth for API authentication?"
  3. "Which is better for microservices and why?"

Searching the vector DB with ONE complex query → misses pieces.
Searching with THREE focused queries → gets everything.
```

### Less Abstract = Chain-of-Thought (COT)

Chain-of-Thought decomposition means: **break the problem into a step-by-step plan**.

> Think of it like the **Google "Think Machine Learning" research paper** approach — instead of tackling "Machine Learning" as a whole:
> - Think about "Machine" (what is a machine?)
> - Think about "Learning" (what is learning?)
> - Think about "Machine Learning" (combine both)

```
User Query: "How does machine learning work in recommendation systems?"
               │
               ↓
LLM decomposes into a step-by-step plan:
               │
  Step 1: "What is machine learning?" (foundation)
  Step 2: "What are recommendation systems?" (foundation)
  Step 3: "What algorithms are used in ML-based recommendations?" (specific)
  Step 4: "How do collaborative filtering and content-based filtering work?" (detail)
               │
               ↓
Each step → Search Vector DB → Get chunks
               │
               ↓
Each step's chunks → LLM generates a mini-response
               │
               ↓
ALL mini-responses + Original Query → LLM → Final comprehensive answer
```

### Why This Works:

```
Looking at things in ISOLATION → brings in MORE CONTEXT

Before learning about "Machine Learning", we want to first understand:
  - What is a "Machine"?
  - What is "Learning"?

This is like NLP context enrichment — each sub-query pulls in sparse/relevant 
chunks that the original broad query would have missed.

The final answer has CONTEXT from all angles → more accurate.
```

### The Decomposition Spectrum:

```
More Abstract                                          Less Abstract
(Step-Back Prompting)                                  (Chain of Thought)
←──────────────────────────────────────────────────────────────────────→

Go BROADER:                                            Go DEEPER:
"What is authentication?"                              "How does JWT refresh
                                                        token rotation work
Step 1: Understand the big picture                      in Express.js?"
Step 2: Then zoom into specifics
                                                       Step 1: What is JWT?
                                                       Step 2: What is refresh token?
                                                       Step 3: How does rotation work?
                                                       Step 4: Express.js implementation?
```

> **Both directions are necessary** — abstract gives you the BIG PICTURE, detailed gives you PRECISION. The technique you pick depends on your use case.

---

## 6. Step-Back Prompting — Going Abstract

### What Is It?

Instead of answering the specific question directly, first **take a step back** and ask a **more general/abstract version** of it.

> **The Physics & World Example:**
> ```
> Student asks: "What happens to pressure when temperature increases 
>                in a container of ideal gas at constant volume?"
> 
> Step-Back question: "What are the general principles of gas laws in physics?"
> 
> → First understand the general principle (PV = nRT)
> → Then apply it to the specific scenario
> → Much more grounded and accurate answer
> ```

### Step-Back Prompting Diagram:

```
User Query: "How does JWT refresh token rotation work in Express.js?"
       │
       ↓
Step-Back (More Abstract): "What is JWT authentication and how does it work?"
       │
       ↓
Search Vector DB with STEP-BACK query
       │
       ↓
Get broad context about JWT, auth fundamentals
       │
       ↓
NOW search with ORIGINAL specific query
       │
       ↓
Get specific chunks about refresh token rotation
       │
       ↓
COMBINE: Broad context + Specific chunks + Original Query → LLM
       │
       ↓
Answer grounded in BOTH general principles AND specific details
```

### How It's Different from Decomposition:

| Technique | Direction | Example |
|-----------|----------|---------|
| **Step-Back** | Go UP (more abstract) | Specific Q → General Q → then answer |
| **Decomposition (COT)** | Go DOWN (more specific) | Complex Q → break into sub-Qs → answer each |
| **Both combined** | UP then DOWN | General context + specific sub-answers |

### When Step-Back Prompting Helps vs Doesn't:

| Helps A LOT | Not Needed |
|-------------|-----------|
| **Legal documents** — laws depend on broader principles | Simple factual Q&A chatbots |
| **Medical docs** — symptoms relate to general physiology | Code search / debugging |
| **Physics/Science** — specific cases need general laws | Direct lookup queries |
| **Policy compliance** — rules reference broader frameworks | Keyword-searchable content |

---

## 7. Few-Shot Prompting in Query Translation

### What Is It?

Give the LLM **examples** of how to translate/rewrite queries, so it produces better rewrites.

```python
system_prompt = """You are a query rewriter for a legal document search system.

Given a user question, rewrite it into a more precise search query.

Examples:

User: "Can I get fired for being late?"
Rewrite: "What are the company policies regarding termination due to 
          tardiness or repeated late arrivals?"

User: "vacation days?"
Rewrite: "How many paid vacation days are employees entitled to per year, 
          and what is the process for requesting time off?"

User: "boss being mean"
Rewrite: "What are the company's policies on workplace harassment, 
          hostile work environment, and the grievance reporting process?"

Now rewrite:
User: "{user_query}"
Rewrite: """
```

### Why Few-Shot Works Well Here:

```
Without examples:
  LLM might rewrite "vacation days?" as "How many vacation days?"
  → Still vague, barely improved

With examples:
  LLM sees the PATTERN — add context, be specific, mention policies
  LLM rewrites "vacation days?" as "How many paid vacation days are 
  employees entitled to per year, and what is the process for requesting 
  time off?"
  → MUCH better for vector search
```

### Few-Shot in Context of Step-Back vs Decomposition:

```
Step Back Prompting                    Decomposition (COT)
(More Abstract)                        (Less Abstract / More Specific)
                                       
Uses few-shot to teach LLM:           Uses few-shot to teach LLM:
"Given specific Q, ask broader Q"      "Given complex Q, break into steps"

Example:                               Example:
Q: "JWT refresh in Express?"           Q: "Compare JWT and OAuth?"
→ "What is JWT authentication?"        → 1. "What is JWT?"
                                         2. "What is OAuth?"
                                         3. "Key differences?"
```

### Does It Help in Legal Docs?

**YES — significantly.** Legal language is very specific. Without few-shot examples, the LLM won't know how to rewrite queries in legal terminology.

```
Without few-shot:
  "Can I sue?" → "Can a person file a lawsuit?" (generic)

With few-shot examples from legal domain:
  "Can I sue?" → "What are the grounds for filing a civil lawsuit, 
  what is the statute of limitations for this type of claim, and what 
  documentation is required to initiate legal proceedings?" (domain-specific)
```

---

## 8. HyDE — Hypothetical Document Embeddings

### What Is It?

Instead of embedding the **user's query** for search, ask the LLM to **write a hypothetical answer/document** first, then embed **THAT** and use it for search.

### Why It Exists:

```
User query:     "that serverless thing on AWS"
                → Vague, short → BAD embedding → BAD search

LLM-generated:  "AWS Lambda is a serverless computing service that 
                  lets you run code without provisioning servers.
                  It supports Node.js, Python, Java. It integrates 
                  with API Gateway, DynamoDB, S3..."
                → Rich, detailed → GREAT embedding → GREAT search
```

The hypothetical document is **CLOSER in embedding space** to the actual stored documents than the user's vague query.

### The Architecture:

```
┌──────────────────────────────────────────────────────────────┐
│                          HyDE                                │
│                                                              │
│  User Query: "that serverless thing on AWS"                  │
│       │                                                      │
│       ↓                                                      │
│  ┌─────────────────────────────────────┐                     │
│  │ LLM generates HYPOTHETICAL DOCUMENT │                     │
│  │                                     │                     │
│  │ "AWS Lambda is a serverless compute │                     │
│  │  service that lets you run code     │                     │
│  │  without provisioning servers.      │                     │
│  │  Functions are triggered by events  │                     │
│  │  like HTTP requests, S3 uploads,    │                     │
│  │  or DynamoDB stream changes..."     │                     │
│  └────────────┬────────────────────────┘                     │
│               ↓                                              │
│  ┌─────────────────────────┐                                 │
│  │ Embed the HYPOTHETICAL  │  ← NOT the user's query!        │
│  │ document → vector       │                                 │
│  │ [0.82, 0.76, 0.91, ...] │                                 │
│  └────────────┬────────────┘                                 │
│               ↓                                              │
│  ┌─────────────────────────┐                                 │
│  │ Search Vector DB with   │                                 │
│  │ this richer embedding   │  → Gets MUCH better matches!    │
│  └────────────┬────────────┘                                 │
│               ↓                                              │
│  ┌─────────────────────────┐                                 │
│  │ Found chunks + Original │                                 │
│  │ User Query → LLM       │  → Final answer                 │
│  └─────────────────────────┘                                 │
└──────────────────────────────────────────────────────────────┘
```

### Important: You Need LARGE, CAPABLE Models

HyDE works best with **highly capable models** because the LLM needs to generate a **realistic, relevant** hypothetical document.

```
Small model (e.g., tiny LLaMA):
  Input:  "that serverless thing on AWS"
  Output: "Serverless is a thing. AWS has it. It runs code." 
           ← Too generic, bad embedding

Large model (e.g., GPT-4, Gemini Pro):
  Input:  "that serverless thing on AWS"
  Output: "AWS Lambda is a serverless computing service that lets you 
           run code without provisioning or managing servers. It 
           automatically scales, supports multiple runtimes including 
           Node.js 18.x, Python 3.12, Java 17..." 
           ← Rich, specific, GREAT embedding
```

> **This is like video processing** — you need a powerful model. Small/cheap models can't generate realistic hypothetical documents with enough domain-specific detail.

### When HyDE Works vs Doesn't:

| Works Well ✅ | Doesn't Work ❌ |
|--------------|----------------|
| Vague/exploratory queries | Users know exactly what they want |
| Topics the LLM knows about | **Legal documents** — LLM might hallucinate wrong legal terms |
| General knowledge domains | Proprietary/confidential data |
| When model is large & capable | Small/weak models |

### Why HyDE Fails for Legal Docs:

```
User: "What recourse do I have if my landlord refuses repairs?"

HyDE-generated doc: "Under most tenant protection laws, tenants have 
the right to request repairs. If the landlord refuses, the tenant can 
file a complaint with local housing authorities..."

Problem: The LLM made up GENERIC legal advice. But the ACTUAL documents 
in your vector DB might be about SPECIFIC state laws (e.g., Maharashtra 
Rent Control Act) with very different provisions.

The HyDE embedding is now CLOSE to generic legal text, not your specific docs.
→ Retrieves wrong/generic chunks → Bad answer

For legal docs: Use Step-Back Prompting or Decomposition instead.
```

### HyDE vs Other Techniques:

```
Multi-Query:     Same question → multiple REWRITES → search each
Decomposition:   Same question → break into PARTS → search each part
Step-Back:       Same question → go BROADER → search broader
HyDE:            Same question → generate FAKE ANSWER → embed the fake answer → search

Each technique creates a BETTER embedding for the vector search,
just through different strategies.
```

---

## 9. Routing — Sending Queries to the Right Place

### What Is It?

Not all questions should go to the same data source. **Routing** decides WHERE to send the query based on its type.

### The Architecture:

```
User Query
    │
    ↓
┌──────────────────────┐
│  ROUTER (LLM-based)  │
│                      │
│  Classifies the      │
│  query type          │
└───┬──────┬───────┬───┘
    │      │       │
    ↓      ↓       ↓
┌──────┐┌──────┐┌──────┐
│Vector││Graph ││ SQL  │
│  DB  ││  DB  ││  DB  │
│(docs)││(Neo4j)││(data)│
└──────┘└──────┘└──────┘
```

### Two Types of Routing:

#### 1. Logical Routing — LLM Classifies Using Rules

```python
system_prompt = """Given a user question, classify it into one of these categories:

1. VECTOR_DB — Questions about document content, concepts, explanations
2. GRAPH_DB  — Questions about relationships between entities
3. SQL_DB    — Questions about numbers, statistics, aggregations

Return ONLY the category name."""

# "What is FS module?" → VECTOR_DB
# "What depends on FS module?" → GRAPH_DB
# "How many modules are in Node.js?" → SQL_DB
```

#### 2. Semantic Routing — Embed the Query and Compare

```
Pre-defined route embeddings:
  "document search" → [0.8, 0.2, 0.1]
  "relationship query" → [0.1, 0.9, 0.2]
  "data analysis" → [0.3, 0.1, 0.8]

User query embedding: [0.75, 0.25, 0.15]

Closest to "document search" → Route to Vector DB
```

---

## 10. Query Construction — Translating to DB Language

### What Is It?

Convert the user's **natural language** question into the **specific query language** of the target database.

```
User says: "Show me all employees hired after 2020 in the engineering team"

If routed to SQL DB:     SELECT * FROM employees WHERE hire_date > '2020-01-01' 
                          AND department = 'Engineering'

If routed to Neo4j:      MATCH (e:Employee)-[:WORKS_IN]->(d:Department {name: 'Engineering'})
                          WHERE e.hire_date > '2020-01-01' RETURN e

If routed to Vector DB:  Metadata filter: {department: "Engineering", hire_date: ">2020"}
                          + similarity_search("employees hired after 2020")
```

### Query Construction Types:

```
┌─────────────────────────────────────────────────────────────┐
│                   QUERY CONSTRUCTION                        │
│                                                             │
│  Natural Language ──→ Text-to-SQL    (Relational DBs)       │
│                  ──→ Text-to-Cypher  (Graph DBs / Neo4j)    │
│                  ──→ Text-to-Metadata (Vector DB filters)   │
│                  ──→ Text-to-API     (REST endpoints)       │
└─────────────────────────────────────────────────────────────┘
```

| Construction | Target | Example |
|-------------|--------|---------|
| **Text-to-SQL** | MySQL, PostgreSQL | `SELECT COUNT(*) FROM orders WHERE status='shipped'` |
| **Text-to-Cypher** | Neo4j | `MATCH (n)-[:DEPENDS_ON]->(m) RETURN m.name` |
| **Text-to-Metadata** | Vector DB filters | `filter={"source": "hr_policy.pdf", "page": 42}` |

---

## 11. Indexing — How You Store Matters

### What Is It?

Indexing is about HOW you chunk, embed, and store your documents. Bad indexing → bad retrieval, no matter how good your query is.

### Advanced Indexing Techniques:

```
┌─────────────────────────────────────────────────────────────┐
│                   INDEXING TECHNIQUES                        │
│                                                             │
│  ┌─────────────────────────────────────┐                    │
│  │ 1. Multi-Representation Indexing    │                    │
│  │    Store SUMMARIES for search,      │                    │
│  │    but retrieve FULL documents      │                    │
│  │                                     │                    │
│  │    Summary (embed this)             │                    │
│  │         ↓                           │                    │
│  │    search match → return full doc   │                    │
│  └─────────────────────────────────────┘                    │
│                                                             │
│  ┌─────────────────────────────────────┐                    │
│  │ 2. Parent-Child Indexing            │                    │
│  │    Small chunks for SEARCH          │                    │
│  │    Return parent chunk for CONTEXT  │                    │
│  │                                     │                    │
│  │    Child (500 chars) → matched      │                    │
│  │         ↓                           │                    │
│  │    Return parent (2000 chars)       │                    │
│  └─────────────────────────────────────┘                    │
│                                                             │
│  ┌─────────────────────────────────────┐                    │
│  │ 3. Specialized Embeddings          │                    │
│  │    Use DIFFERENT embedding models   │                    │
│  │    for different data types         │                    │
│  │                                     │                    │
│  │    Text → text-embedding model      │                    │
│  │    Code → code-embedding model      │                    │
│  │    Tables → table-embedding model   │                    │
│  └─────────────────────────────────────┘                    │
│                                                             │
│  ┌─────────────────────────────────────┐                    │
│  │ 4. ColBERT (Contextualized Late    │                    │
│  │    Interaction over BERT)           │                    │
│  │    Token-level embeddings instead   │                    │
│  │    of document-level                │                    │
│  │    → More precise matching          │                    │
│  └─────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

### In Our Rag_1.py:

```python
# We use basic indexing:
# RecursiveCharacterTextSplitter → OllamaEmbeddings → Qdrant
# This is "Naive Indexing" — works but can be improved
```

---

## 12. Retrieval — Getting the Right Chunks

### What Is It?

The actual search phase — how you find and rank the most relevant chunks from your vector database.

### Retrieval Improvement Techniques:

```
┌─────────────────────────────────────────────────────────────┐
│                   RETRIEVAL TECHNIQUES                       │
│                                                             │
│  1. Re-Ranking                                              │
│     Initial search → 20 candidates                          │
│     Cross-encoder re-ranks → Top 5 best                     │
│                                                             │
│  2. Hybrid Search                                           │
│     Vector (semantic) + BM25 (keyword) → Combined results   │
│                                                             │
│  3. Corrective RAG (CRAG)                                   │
│     LLM GRADES each retrieved chunk:                        │
│     ├── RELEVANT → keep it                                  │
│     ├── NOT RELEVANT → discard it                           │
│     └── ALL bad → search web / rewrite query                │
│                                                             │
│  4. Active Retrieval                                        │
│     Model decides WHEN to retrieve:                         │
│     ├── "I know this" → answer directly                     │
│     └── "I don't know" → search vector DB                   │
└─────────────────────────────────────────────────────────────┘
```

### Corrective RAG (CRAG) Architecture:

```
Query → Search Vector DB → Chunks [c1, c2, c3, c4, c5]
                                    │
                                    ↓
                           LLM grades EACH chunk:
                           c1: RELEVANT ✅
                           c2: NOT RELEVANT ❌
                           c3: RELEVANT ✅
                           c4: AMBIGUOUS 🤔
                           c5: NOT RELEVANT ❌
                                    │
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
              Most RELEVANT                   Most NOT RELEVANT
                    ↓                               ↓
              Use [c1, c3]                    Fallback:
              for answer                      - Rewrite query
                                              - Search web
                                              - Try different DB
```

---

## 13. Generation — The Final Answer

### What Is It?

The last stage — the LLM generates the answer from retrieved chunks. But we can optimize this too.

### Generation Techniques:

```
┌─────────────────────────────────────────────────────────────┐
│                   GENERATION TECHNIQUES                      │
│                                                             │
│  1. Self-RAG (Self-Reflective)                              │
│     LLM generates answer → checks itself:                   │
│     ├── "Is my answer grounded in the chunks?" → Yes → done │
│     └── "Am I hallucinating?" → Yes → retrieve more + retry │
│                                                             │
│  2. Citation / Grounding                                    │
│     LLM cites WHICH chunk each sentence came from:          │
│     "According to chunk 3, the policy allows..."            │
│                                                             │
│  3. Chain-of-Thought Generation                             │
│     LLM reasons step-by-step before answering:              │
│     "Based on chunk 1, I know X.                            │
│      Based on chunk 3, I know Y.                            │
│      Therefore, the answer is Z."                           │
│                                                             │
│  4. Faithfulness Check                                      │
│     Second LLM call verifies the answer against chunks:     │
│     "Does this answer contain info NOT in the context?"     │
│     → Yes → flag as potential hallucination                  │
└─────────────────────────────────────────────────────────────┘
```

### Self-RAG Architecture:

```
Query + Chunks → LLM generates answer
                      │
                      ↓
                 Self-Evaluate:
                 "Is this answer supported by the chunks?"
                      │
              ┌───────┴───────┐
              ↓               ↓
          YES → Return     NO → Loop back
          answer              │
                              ↓
                         Retrieve MORE chunks
                         or DIFFERENT chunks
                              │
                              ↓
                         Try generating again
```

---

## 14. Quick Revision Cheat Sheet

| Concept | One-liner |
|---------|-----------|
| **Query Translation** | Transform user query into better queries before searching vector DB |
| **Garbage In = Garbage Out** | Bad query → bad chunks → bad answer. Fix the query first! |
| **What user ASKS ≠ What user WANTS** | Query translation bridges this gap |
| **Abstraction Spectrum** | Too vague = random chunks. Too specific = misses context. Sweet spot = middle |
| **Multi-Query / RAG Fusion** | LLM generates 3-5 question versions → search each → merge results |
| **Fan-Out** | One input → multiple parallel outputs (like a fan spreading out) |
| **RRF (Reciprocal Rank Fusion)** | Ranking algorithm: `Σ 1/(k + rank)` — rewards chunks appearing in multiple lists |
| **k=60 in RRF** | Dampening constant — prevents rank 1 from dominating, rewards breadth |
| **Query Decomposition** | Break complex query into sub-questions → answer each → combine |
| **Step-Back Prompting** | Go MORE ABSTRACT — ask general question first, then specific |
| **Chain-of-Thought Decomp** | Go LESS ABSTRACT — break into detailed step-by-step plan |
| **Few-Shot for Translation** | Give LLM examples of good query rewrites so it produces better ones |
| **HyDE** | LLM generates hypothetical answer → embed THAT → search (needs large models) |
| **HyDE ❌ Legal Docs** | LLM might generate wrong legal terminology → bad embeddings |
| **Routing** | Classify query → send to right DB (Vector, Graph, SQL) |
| **Logical Routing** | LLM classifies query type using rules/prompt |
| **Semantic Routing** | Compare query embedding to pre-defined route embeddings |
| **Query Construction** | Natural language → Text-to-SQL, Text-to-Cypher, Text-to-Metadata |
| **Multi-Rep Indexing** | Embed summaries for search, return full docs for context |
| **Parent-Child Indexing** | Small chunks for search precision, return parent chunk for context |
| **ColBERT** | Token-level embeddings instead of document-level — more precise matching |
| **Re-Ranking** | Initial search gets candidates, cross-encoder re-ranks for precision |
| **Corrective RAG** | LLM grades each chunk as RELEVANT/NOT → discards bad ones → retries if all bad |
| **Self-RAG** | LLM evaluates its own answer → retries if not grounded in context |
| **Faithfulness Check** | Second LLM call verifies answer against chunks for hallucinations |
| **Song Analogy** | You hum a tune → friend recognizes the song = vague input → translated to precise query |
| **Legal Docs Need** | Step-Back + Decomposition + Few-Shot (NOT HyDE) — domain-specific accuracy |
| **Simple Chatbot Needs** | Basic RAG or Multi-Query — don't over-engineer |

### When to Use What — Decision Matrix:

```
┌─────────────────────────────────────────────────────────────┐
│  USE CASE                    →  TECHNIQUES TO USE           │
├─────────────────────────────────────────────────────────────┤
│  Simple coding chatbot       →  Naive RAG (Rag_1.py)        │
│  Legal document search       →  Step-Back + Decomposition   │
│                                  + Few-Shot + RRF            │
│  Exploratory/browse          →  HyDE + Multi-Query           │
│  Multi-source data           →  Routing + Query Construction │
│  High-accuracy required      →  CRAG + Self-RAG + Re-Rank    │
│  Tabular/structured data     →  TAG + Text-to-SQL            │
│  Relationship-heavy data     →  GraphRAG + Routing            │
└─────────────────────────────────────────────────────────────┘
```

### The Complete Advanced RAG Pipeline:

```
User Query
    │
    ↓
┌──── QUERY TRANSLATION ────┐
│ Multi-Query / Step-Back /  │
│ Decomposition / HyDE       │
└────────────┬───────────────┘
             ↓
┌──── ROUTING ──────────────┐
│ Where should this go?      │
│ Vector DB? Graph DB? SQL?  │
└────────────┬───────────────┘
             ↓
┌──── QUERY CONSTRUCTION ───┐
│ Natural lang → DB language │
│ Text-to-SQL / Cypher /     │
│ Metadata filters           │
└────────────┬───────────────┘
             ↓
┌──── RETRIEVAL ────────────┐
│ Search → RRF Ranking →     │
│ Re-Rank → CRAG grading     │
└────────────┬───────────────┘
             ↓
┌──── GENERATION ───────────┐
│ LLM answers with context   │
│ Self-RAG check →            │
│ Citations → Final answer    │
└────────────────────────────┘
```
