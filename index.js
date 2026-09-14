// index.js
//
// Express app — same route shape as the reference ai-application-layer
// backend (/api/health, /api/ingest, /api/query), plus /api/filters so the
// SwiftUI and web dashboards can populate the four medical-dataset
// selection parameters. No ChromaDB anywhere in this file or in
// lib/vectorstore.js — retrieval is in-memory cosine similarity.

require("dotenv").config();
const express = require("express");
const cors = require("cors");
const path = require("path");

const { loadMedicalRecords, recordToText } = require("./lib/dataset");
const vectorstore = require("./lib/vectorstore");
const { gemmaGenerate, GEMMA_MODEL, GEMMA_PROVIDER } = require("./lib/gemma");

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// Static option lists for the dashboard's 4 selection parameters.
// (A live /api/filters?discover=true could page the dataset to build these
// dynamically; the static list keeps cold starts fast.)
const FILTER_OPTIONS = {
  subjectName: ["Anatomy", "Physiology", "Pharmacology", "Medicine", "Surgery", "Pediatrics", "Psychiatry", "Microbiology"],
  topicName: ["Cardiovascular", "Respiratory", "Nervous System", "Endocrine", "Renal", "Musculoskeletal", "Gastrointestinal"],
  choiceType: ["single", "multi"],
  correctOption: ["a", "b", "c", "d"],
};

app.get("/api/health", (_req, res) => {
  res.json({ status: "ok", recordsIndexed: vectorstore.chunkCount() });
});

app.get("/api/filters", (_req, res) => {
  res.json(FILTER_OPTIONS);
});

app.post("/api/ingest", async (req, res) => {
  try {
    const { limit = 200, split = "train", subjectName, topicName, choiceType, correctOption } = req.body || {};

    const records = await loadMedicalRecords({
      limit,
      split,
      subjectName,
      topicName,
      choiceType,
      correctOption,
    });

    if (records.length === 0) {
      return res.status(200).json({
        message: "No records matched the selected filters.",
        recordsIngested: 0,
        chunksIngested: 0,
        sample: [],
      });
    }

    const chunksIngested = await vectorstore.upsertRecords(records, recordToText);

    res.json({
      message: "Ingest complete.",
      recordsIngested: records.length,
      chunksIngested,
      sample: records.slice(0, 3).map((r) => r.question),
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/query", async (req, res) => {
  try {
    const { question, topK = 4 } = req.body || {};
    if (!question) return res.status(400).json({ error: "`question` is required" });

    const topChunks = await vectorstore.retrieve(question, topK);
    if (topChunks.length === 0) {
      return res.json({
        answer: "No records have been ingested yet — call /api/ingest first.",
        provider: GEMMA_PROVIDER,
        model: GEMMA_MODEL,
        sources: [],
      });
    }

    const { text, provider, model } = await gemmaGenerate(question, topChunks);

    res.json({
      answer: text,
      provider,
      model,
      sources: topChunks.map((c) => ({
        text: c.text,
        score: c.score,
        subjectName: c.metadata.subjectName,
        topicName: c.metadata.topicName,
        choiceType: c.metadata.choiceType,
        correctOption: c.metadata.correctOption,
      })),
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`autoclinic-consult-api listening on http://localhost:${PORT}`);
});

module.exports = app;
