import { Client } from "@elastic/elasticsearch";

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
  const must: Record<string, unknown>[] = [];
  if (matchId) must.push({ term: { match_id: matchId } });

  const result = await elastic.search({
    index: INDEX,
    size,
    body: {
      query: {
        bool: {
          must,
          should: [
            { multi_match: { query, fields: ["text^2", "text.general^3"], type: "best_fields", fuzziness: "AUTO" } },
            { match_phrase: { text: { query, boost: 5 } } },
            { match_phrase: { "text.general": { query, boost: 5 } } },
          ],
          minimum_should_match: 1,
        },
      },
    },
  });

  return result.hits?.hits || [];
}

export default elastic;
