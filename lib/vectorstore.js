// lib/vectorstore.js
//
// In-memory vector store — this is the one deliberate architectural change
// from the reference app: ChromaDB is NOT used. Instead this module keeps
// chunk vectors in a plain JS array (module-level state) and does exact
// cosine-similarity search over it. That's enough for a small MedMCQA slice
// (a few hundred to a few thousand chunks) and it means the whole backend
// stays stateless-deploy-friendly on Render with zero external vector DB to
// provision — the tradeoff (vs Chroma) is that the index is rebuilt from
// scratch on every cold start, which is fine since /api/ingest is cheap and
// fast for a corpus this size.
//
// Chunking still uses LlamaIndex.TS's SentenceSplitter, same as the
// reference app, so the "chunk shape" of the pipeline is unchanged — only
// the storage/retrieval backend differs.

const { SentenceSplitter } = require("llamaindex");
const { embedText } = require("./gemma");

/** @type {{ id: string, text: string, vector: number[], metadata: object }[]} */
let CHUNKS = [];

const splitter = new SentenceSplitter({ chunkSize: 400, chunkOverlap: 40 });

function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Chunk each record's text, embed every chunk, and upsert into the
 * in-memory store. Mirrors vectorstore.js's `upsert()` in the reference
 * backend, minus the Chroma client calls.
 */
async function upsertRecords(records, recordToText) {
  let chunkCount = 0;
  for (const record of records) {
    const text = recordToText(record);
    const nodes = splitter.splitText(text);
    for (const [i, chunkText] of nodes.entries()) {
      const vector = await embedText(chunkText);
      CHUNKS.push({
        id: `${record.id}-${i}`,
        text: chunkText,
        vector,
        metadata: {
          subjectName: record.subjectName,
          topicName: record.topicName,
          choiceType: record.choiceType,
          correctOption: record.correctOption,
        },
      });
      chunkCount++;
    }
  }
  return chunkCount;
}

/** Top-k nearest chunks to the query, by cosine similarity. */
async function retrieve(question, topK = 4) {
  if (CHUNKS.length === 0) return [];
  const queryVector = await embedText(question);
  return CHUNKS
    .map((c) => ({ ...c, score: cosineSimilarity(queryVector, c.vector) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

function chunkCount() {
  return CHUNKS.length;
}

function reset() {
  CHUNKS = [];
}

module.exports = { upsertRecords, retrieve, chunkCount, reset };
