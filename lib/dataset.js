// lib/dataset.js
//
// Hugging Face MedMCQA loader — the medical-dataset counterpart of the
// reference app's AG News Sci/Tech loader. Instead of filtering a single
// news category, this exposes FOUR independent selection parameters so the
// dashboard can slice the corpus the way a clinician would:
//
//   1. subjectName    e.g. "Medicine", "Pharmacology", "Anatomy"
//   2. topicName       e.g. "Cardiovascular", "Renal", "Endocrine"
//   3. choiceType       "single" | "multi"
//   4. correctOption    "a" | "b" | "c" | "d"  (derived from the `cop` field)
//
// Dataset: openlifescienceai/medmcqa (MedMCQA — Pal, Umapathi & Sankarasubbu,
// CHIL 2022), accessed live via the Hugging Face datasets-server REST API —
// no local copy is checked into this repo, mirroring how the reference app
// never vendors AG News.

const DATASETS_SERVER = "https://datasets-server.huggingface.co/rows";
const DATASET_ID = process.env.HF_DATASET_ID || "openlifescienceai/medmcqa";
const COP_LETTERS = ["a", "b", "c", "d"];

/**
 * Fetch a page of rows from the Hugging Face datasets-server REST API.
 */
async function fetchRows({ split = "train", offset = 0, length = 100 }) {
  const url = `${DATASETS_SERVER}?dataset=${encodeURIComponent(DATASET_ID)}&config=default&split=${split}&offset=${offset}&length=${length}`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Hugging Face datasets-server error ${res.status}: ${await res.text()}`);
  }
  const json = await res.json();
  return (json.rows || []).map((r) => r.row);
}

/**
 * Load MedMCQA records, paging through datasets-server, and apply the four
 * dashboard filters client-side (the datasets-server rows endpoint doesn't
 * support server-side filtering, so we page and filter — same approach the
 * reference app used for AG News' label == 3 filter).
 *
 * @param {Object} opts
 * @param {number} opts.limit          max number of matching records to return
 * @param {string} [opts.subjectName]  e.g. "Medicine"
 * @param {string} [opts.topicName]    e.g. "Cardiovascular"
 * @param {string} [opts.choiceType]   "single" | "multi"
 * @param {string} [opts.correctOption]"a" | "b" | "c" | "d"
 * @param {string} [opts.split]        "train" | "test" | "validation"
 */
async function loadMedicalRecords({
  limit = 200,
  subjectName,
  topicName,
  choiceType,
  correctOption,
  split = "train",
} = {}) {
  const matched = [];
  let offset = 0;
  const pageSize = 100;
  const maxPages = 40; // safety cap ~4000 rows scanned

  for (let page = 0; page < maxPages && matched.length < limit; page++) {
    let rows;
    try {
      rows = await fetchRows({ split, offset, length: pageSize });
    } catch (err) {
      // If the dataset id/split is unavailable, stop paging gracefully.
      break;
    }
    if (rows.length === 0) break;

    for (const row of rows) {
      const copLetter = typeof row.cop === "number" ? COP_LETTERS[row.cop] : row.cop;

      if (subjectName && (row.subject_name || "").toLowerCase() !== subjectName.toLowerCase()) continue;
      if (topicName && !(row.topic_name || "").toLowerCase().includes(topicName.toLowerCase())) continue;
      if (choiceType && (row.choice_type || "").toLowerCase() !== choiceType.toLowerCase()) continue;
      if (correctOption && copLetter !== correctOption.toLowerCase()) continue;

      matched.push({
        id: row.id,
        question: row.question,
        options: { a: row.opa, b: row.opb, c: row.opc, d: row.opd },
        correctOption: copLetter,
        choiceType: row.choice_type,
        explanation: row.exp,
        subjectName: row.subject_name,
        topicName: row.topic_name,
      });

      if (matched.length >= limit) break;
    }
    offset += pageSize;
  }

  return matched;
}

/** Render one MedMCQA record as flat text, ready for chunking/embedding. */
function recordToText(record) {
  const opts = record.options;
  return [
    `Question: ${record.question}`,
    `Options: (a) ${opts.a}  (b) ${opts.b}  (c) ${opts.c}  (d) ${opts.d}`,
    `Correct answer: ${record.correctOption}`,
    record.explanation ? `Explanation: ${record.explanation}` : null,
    `Subject: ${record.subjectName} | Topic: ${record.topicName} | Choice type: ${record.choiceType}`,
  ]
    .filter(Boolean)
    .join("\n");
}

module.exports = { loadMedicalRecords, recordToText, DATASET_ID };
