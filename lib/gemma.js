// lib/gemma.js
//
// Same role as the reference app's lib/gemma.js: Gemma generation +
// text-embedding-004 embeddings via the Vercel AI SDK, with an optional
// local Ollama fallback for offline demos.

const { generateText, embed } = require("ai");
const { createGoogleGenerativeAI } = require("@ai-sdk/google");

const GEMMA_MODEL = process.env.GEMMA_MODEL || "gemma-3-27b-it";
const GEMMA_PROVIDER = process.env.GEMMA_PROVIDER || "google"; // "google" | "ollama"

const google = createGoogleGenerativeAI({
  apiKey: process.env.GOOGLE_GENERATIVE_AI_API_KEY,
});

/** Embed one chunk / question with text-embedding-004. */
async function embedText(text) {
  const { embedding } = await embed({
    model: google.textEmbeddingModel("text-embedding-004"),
    value: text,
  });
  return embedding;
}

/**
 * Build a numbered RAG prompt from retrieved chunks and ask Gemma to answer
 * ONLY from that context, citing snippet numbers — same "fact-checking
 * assistant" pattern as the reference app, applied to medical Q&A records
 * instead of news articles.
 */
async function gemmaGenerate(question, contextChunks) {
  const numberedContext = contextChunks
    .map((c, i) => `[${i + 1}] ${c.text}`)
    .join("\n\n");

  const systemPrompt = [
    "You are a careful clinical-reference assistant.",
    "Answer the user's question using ONLY the numbered context below.",
    "Cite the snippet numbers you used, e.g. [1][3].",
    "If the context does not contain the answer, say so plainly instead of guessing.",
    "This is for study/reference purposes, not a diagnosis of any real patient.",
  ].join(" ");

  const prompt = `${systemPrompt}\n\nContext:\n${numberedContext}\n\nQuestion: ${question}\n\nAnswer:`;

  if (GEMMA_PROVIDER === "ollama") {
    return ollamaGenerate(prompt);
  }

  const { text } = await generateText({
    model: google(GEMMA_MODEL),
    prompt,
  });
  return { text, provider: "google-ai-studio", model: GEMMA_MODEL };
}

/** Optional fully-offline fallback: a locally-served Gemma model via Ollama. */
async function ollamaGenerate(prompt) {
  const res = await fetch("http://localhost:11434/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: GEMMA_MODEL, prompt, stream: false }),
  });
  const json = await res.json();
  return { text: json.response, provider: "ollama", model: GEMMA_MODEL };
}

module.exports = { embedText, gemmaGenerate, GEMMA_MODEL, GEMMA_PROVIDER };
