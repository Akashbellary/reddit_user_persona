import { pipeline, env } from "@xenova/transformers"

// Configure transformers to use local models
env.allowLocalModels = false
env.allowRemoteModels = true

let embedder: any = null

export async function initializeEmbedder() {
  if (!embedder) {
    console.log("[v0] Initializing embedding model...")
    try {
      embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2")
      console.log("[v0] Embedding model initialized successfully")
    } catch (error) {
      console.error("[v0] Failed to initialize embedding model:", error)
      throw new Error(
        `Failed to initialize embedding model: ${error instanceof Error ? error.message : "Unknown error"}`,
      )
    }
  }
  return embedder
}

export async function generateEmbedding(text: string): Promise<number[]> {
  console.log("[v0] Generating embedding for text length:", text.length)

  try {
    const model = await initializeEmbedder()

    // Generate embedding
    const output = await model(text, { pooling: "mean", normalize: true })

    // Convert to regular array
    const embedding = Array.from(output.data)
    console.log("[v0] Generated embedding with dimension:", embedding.length)
    return embedding
  } catch (error) {
    console.error("[v0] Failed to generate embedding:", error)
    throw new Error(`Failed to generate embedding: ${error instanceof Error ? error.message : "Unknown error"}`)
  }
}

export async function generateEmbeddings(texts: string[]): Promise<number[][]> {
  console.log(`[v0] Generating embeddings for ${texts.length} texts...`)
  const embeddings: number[][] = []

  for (let i = 0; i < texts.length; i++) {
    console.log(`[v0] Processing embedding ${i + 1}/${texts.length}`)
    const embedding = await generateEmbedding(texts[i])
    embeddings.push(embedding)
  }

  console.log(`[v0] Successfully generated ${embeddings.length} embeddings`)
  return embeddings
}
