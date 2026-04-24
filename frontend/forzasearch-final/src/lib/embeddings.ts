// Local multilingual embeddings via @xenova/transformers. The model runs in
// Node through onnxruntime-node, so no Python or external API is needed. First
// run downloads ~50MB to ~/.cache/huggingface and is cached after.
//
// Model: paraphrase-multilingual-MiniLM-L12-v2 — 384 dims, handles Swedish and
// English in the same vector space, which is what lets an English query match a
// Swedish-indexed window.

import { pipeline, type FeatureExtractionPipeline } from "@xenova/transformers";

const MODEL_ID = "Xenova/paraphrase-multilingual-MiniLM-L12-v2";

export const EMBEDDING_DIMS = 384;

let extractorPromise: Promise<FeatureExtractionPipeline> | null = null;

function getExtractor(): Promise<FeatureExtractionPipeline> {
  if (!extractorPromise) {
    console.log(`[embeddings] loading ${MODEL_ID}...`);
    extractorPromise = pipeline("feature-extraction", MODEL_ID) as Promise<FeatureExtractionPipeline>;
  }
  return extractorPromise;
}

export async function embed(text: string): Promise<number[]> {
  const model = await getExtractor();
  const output = await model(text, { pooling: "mean", normalize: true });
  return Array.from(output.data as Float32Array);
}

export async function embedMany(texts: string[]): Promise<number[][]> {
  const model = await getExtractor();
  const results: number[][] = [];
  for (const t of texts) {
    const output = await model(t, { pooling: "mean", normalize: true });
    results.push(Array.from(output.data as Float32Array));
  }
  return results;
}
