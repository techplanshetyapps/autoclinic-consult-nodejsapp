# autoclinic-consult-api

## Gemma Medical Q&A RAG Assistant (MedMCQA)

A Gemma-powered retrieval-augmented reference assistant built as the backend
for **AutoClinic Consult**. It ingests a filtered slice of the **MedMCQA**
dataset directly from Hugging Face, indexes it in an **in-memory vector
store** (no ChromaDB), and answers questions about the ingested records
using a **Gemma** model called through the **Vercel AI SDK (ai-sdk)**.

## Architecture

```
Hugging Face (openlifescienceai/medmcqa)
        │  datasets-server REST API
        ▼
  lib/dataset.js  ── fetch + filter on 4 params
        │  subjectName · topicName · choiceType · correctOption
        ▼
  lib/vectorstore.js
        │  LlamaIndex SentenceSplitter → chunks
        │  ai-sdk embed() → text-embedding-004
        ▼
   in-memory vector array (cosine similarity, NO ChromaDB)
        ▲
        │  similarity search (top-k)
  lib/vectorstore.js retrieve()
        │
        ▼
  index.js  /api/query
        │  builds RAG prompt with numbered context
        ▼
  lib/gemma.js  ── ai-sdk generateText() → Gemma (gemma-3-27b-it)
        ▼
   Answer + cited sources → public/index.html dashboard / SwiftUI app
```

## Why no ChromaDB

The reference architecture this project is modeled on uses ChromaDB as an
external vector store. This backend intentionally swaps that for a
module-level in-memory array searched with exact cosine similarity
(`lib/vectorstore.js`). For a small MedMCQA slice (a few hundred to a few
thousand chunks) exact search is fast enough, and it removes the need to
provision or pay for a separate vector database before deploying to Render —
the tradeoff is that the index resets on every cold start/restart, so
`/api/ingest` should be called again after a redeploy.

## Setup and execution

### 1. Install dependencies

```bash
npm install
```

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in:

- `GOOGLE_GENERATIVE_AI_API_KEY` — free key from [Google AI Studio](https://aistudio.google.com/apikey). Used for both Gemma generation and `text-embedding-004` embedding calls.
- `GEMMA_MODEL` — defaults to `gemma-3-27b-it`; any instruction-tuned Gemma model id works, e.g. `gemma-3-4b-it` for lower latency.
- `HF_DATASET_ID` — defaults to `openlifescienceai/medmcqa`.

### 3. Run locally

```bash
npm start
# Server on http://localhost:3000
```

### 4. Ingest the dataset (four selection parameters)

Open `http://localhost:3000` and pick **Subject / Topic / Choice type /
Correct option**, then click **"Ingest Medical Records"** — or call the API
directly:

```bash
curl -X POST http://localhost:3000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"limit": 150, "subjectName": "Medicine", "choiceType": "single"}'
```

### 5. Ask a question

```bash
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the recommended first-line option discussed for this condition?"}'
```

### 6. Deploy to Render

1. Push this folder to a GitHub repo.
2. In the [Render dashboard](https://dashboard.render.com), **New → Web
   Service** → connect the repo (or **New → Blueprint** using the included
   `render.yaml`).
3. Build command: `npm install` · Start command: `npm start`.
4. Set `GOOGLE_GENERATIVE_AI_API_KEY` (and optionally `GEMMA_MODEL`,
   `HF_DATASET_ID`) in the service's **Environment** tab.
5. Once deployed, your API is live at
   `https://<your-service-name>.onrender.com` — point the SwiftUI app's
   `APIClient.baseURL` and `public/index.html`'s `API` constant at that URL.

## API reference

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/api/health` | — | `{ status, recordsIndexed }` |
| GET | `/api/filters` | — | `{ subjectName[], topicName[], choiceType[], correctOption[] }` |
| POST | `/api/ingest` | `{ limit, subjectName?, topicName?, choiceType?, correctOption?, split? }` | `{ message, recordsIngested, chunksIngested, sample[] }` |
| POST | `/api/query` | `{ question, topK? }` | `{ answer, provider, model, sources[] }` |

`sources[]` items: `{ text, score, subjectName, topicName, choiceType, correctOption }`.

## Gemma implementation details

- **Model used:** `gemma-3-27b-it` (configurable via `GEMMA_MODEL`).
- **How it's accessed:** Google AI Studio's Generative Language API via the
  `@ai-sdk/google` provider inside the Vercel **ai-sdk** (`generateText`).
  An optional local fallback (`GEMMA_PROVIDER=ollama`) calls a locally-served
  Gemma model through Ollama for fully offline demos.
- **Approach:** Retrieval-Augmented Generation. The question is embedded and
  matched against the in-memory MedMCQA chunk vectors; the top-k chunks are
  inserted into a numbered context block, and Gemma is instructed to answer
  **only** from that context and cite snippet numbers.
- **Embeddings:** produced by `text-embedding-004`, not Gemma itself (Gemma
  is a text-generation model with no embeddings endpoint) — standard RAG
  pattern of a separate embedding model + separate generation model.

## Repository structure

```
.
├── index.js                     # Express app: /api/ingest, /api/query, /api/health, /api/filters
├── lib/
│   ├── dataset.js                # Hugging Face MedMCQA loader — 4 filter params
│   ├── vectorstore.js             # LlamaIndex chunking + in-memory cosine-similarity store (no ChromaDB)
│   └── gemma.js                   # ai-sdk Gemma generation + embeddings
├── public/index.html             # Web dashboard UI (4 filter dropdowns)
├── gemma-rag-medical.ipynb       # Notebook prototype of the same RAG pipeline
├── render.yaml                   # Render deployment blueprint
├── .env.example
└── package.json
```

## Reused resources

- **Dataset:** [`openlifescienceai/medmcqa`](https://huggingface.co/datasets/openlifescienceai/medmcqa) (Pal, Umapathi & Sankarasubbu, CHIL 2022), accessed live via the [Hugging Face datasets-server REST API](https://huggingface.co/docs/datasets-server) — no local copy is checked into this repo.
- **Libraries:** [Vercel AI SDK (`ai`, `@ai-sdk/google`)](https://sdk.vercel.ai/), [LlamaIndex.TS](https://ts.llamaindex.ai/) (chunking only), Express, cors, dotenv.


