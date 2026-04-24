import { Client } from "@elastic/elasticsearch";
import { embed, EMBEDDING_DIMS } from "./embeddings";

const elastic = new Client({
  node: process.env.ELASTICSEARCH_URL || "http://localhost:9200",
});

export const INDEX = "forzasearch-windows";

export async function createIndex() {
  const exists = await elastic.indices.exists({ index: INDEX });
  if (exists) return;

  await elastic.indices.create({
    index: INDEX,
    body: {
      settings: {
        number_of_shards: 1,
        number_of_replicas: 0,
        analysis: {
          analyzer: {
            swedish_custom: { type: "custom", tokenizer: "standard", filter: ["lowercase", "swedish_stop", "swedish_stemmer"] },
            general: { type: "custom", tokenizer: "standard", filter: ["lowercase", "asciifolding"] },
          },
          filter: {
            swedish_stop: { type: "stop", stopwords: "_swedish_" },
            swedish_stemmer: { type: "stemmer", language: "swedish" },
          },
        },
      },
      mappings: {
        properties: {
          text: { type: "text", analyzer: "swedish_custom", fields: { general: { type: "text", analyzer: "general" } } },
          part: { type: "integer" },
          start_sec: { type: "float" },
          end_sec: { type: "float" },
          match_minute: { type: "integer" },
          match_id: { type: "keyword" },
          embedding: { type: "dense_vector", dims: EMBEDDING_DIMS, index: true, similarity: "cosine" },
        },
      },
    },
  });
}

export async function deleteIndex() {
  const exists = await elastic.indices.exists({ index: INDEX });
  if (exists) await elastic.indices.delete({ index: INDEX });
}

export async function bulkIndex(documents: Record<string, unknown>[]) {
  const body = documents.flatMap((doc) => [{ index: { _index: INDEX } }, doc]);
  return elastic.bulk({ body, refresh: true });
}

export async function searchWindows(query: string, matchId?: string, size = 1000) {
  // Hybrid retrieval: BM25 (lexical) + kNN (semantic) in a single search.
  // BM25 catches exact token matches (player names, specific Swedish terms);
  // kNN catches semantic similarity (cross-language, paraphrases, spelling
  // drift like Vukojevich ↔ Vukojevic). ES sums the two scores for overlapping
  // docs and unions non-overlapping ones.
  //
  // If the embedding model fails to load (e.g. onnxruntime-node missing native
  // deps on this machine), we fall back to pure BM25 so search still works —
  // just without the semantic layer.

  const must: Record<string, unknown>[] = [];
  if (matchId) must.push({ term: { match_id: matchId } });
  const knnFilter = matchId ? { term: { match_id: matchId } } : undefined;

  let queryVector: number[] | null = null;
  try {
    queryVector = await embed(query);
  } catch (err) {
    console.warn(
      "[search] embedding failed, falling back to BM25-only:",
      err instanceof Error ? err.message : err,
    );
  }

  const body: Record<string, unknown> = {
    query: {
      bool: {
        must,
        should: [
          { multi_match: { query, fields: ["text^2", "text.general^3"], type: "best_fields", fuzziness: "AUTO" } },
          { match_phrase: { text: { query, boost: 5 } } },
          { match_phrase: { "text.general": { query, boost: 5 } } },
        ],
        // When no vector is available, require at least one BM25 clause to match —
        // otherwise the match_id filter alone would return every window in the match.
        ...(queryVector ? {} : { minimum_should_match: 1 }),
      },
    },
  };

  if (queryVector) {
    body.knn = {
      field: "embedding",
      query_vector: queryVector,
      k: Math.max(size * 2, 10),
      num_candidates: 100,
      filter: knnFilter,
      boost: 5,
    };
  }

  const result = await elastic.search({ index: INDEX, size, body });
  return result.hits?.hits || [];
}

export default elastic;
