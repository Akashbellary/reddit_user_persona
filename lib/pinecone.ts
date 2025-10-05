import { Pinecone } from "@pinecone-database/pinecone"
import { generateEmbeddings } from "./embeddings"

const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY!,
})

const INDEX_NAME = "reddit-user-vdb"
const DIMENSION = 384 // all-MiniLM-L6-v2 dimension

export async function initializePinecone() {
  try {
    console.log("[v0] Initializing Pinecone connection...")

    // Check if index exists
    console.log("[v0] Checking if Pinecone index exists...")
    const indexes = await pc.listIndexes()
    const indexExists = indexes.indexes?.some((index) => index.name === INDEX_NAME)

    console.log(`[v0] Index ${INDEX_NAME} exists:`, indexExists)

    if (!indexExists) {
      console.log(`[v0] Creating Pinecone index: ${INDEX_NAME}`)
      await pc.createIndex({
        name: INDEX_NAME,
        dimension: DIMENSION,
        metric: "cosine",
        spec: {
          serverless: {
            cloud: "aws",
            region: "us-east-1",
          },
        },
      })

      // Wait for index to be ready
      console.log("[v0] Waiting for index to be ready...")
      await new Promise((resolve) => setTimeout(resolve, 10000))
    }

    const index = pc.index(INDEX_NAME)
    console.log(`[v0] Pinecone index ${INDEX_NAME} ready`)
    return index
  } catch (error) {
    console.error("[v0] Failed to initialize Pinecone:", error)
    throw new Error(`Failed to initialize Pinecone: ${error instanceof Error ? error.message : "Unknown error"}`)
  }
}

export async function storeChunksInPinecone(chunks: string[], username: string) {
  try {
    console.log(`[v0] Starting to store ${chunks.length} chunks for user: ${username}`)

    const index = await initializePinecone()

    if (chunks.length === 0) {
      console.log("[v0] No chunks to store")
      return 0
    }

    console.log(`[v0] Generating embeddings for ${chunks.length} chunks...`)
    const embeddings = await generateEmbeddings(chunks)
    console.log(`[v0] Generated ${embeddings.length} embeddings`)

    // Prepare vectors for upsert
    const vectors = chunks.map((chunk, i) => ({
      id: `${username}-${Date.now()}-${i}`,
      values: embeddings[i],
      metadata: {
        text: chunk,
        username,
        timestamp: new Date().toISOString(),
      },
    }))

    console.log(`[v0] Prepared ${vectors.length} vectors for upsert`)

    // Upsert vectors in batches
    const batchSize = 100
    let upsertedCount = 0

    for (let i = 0; i < vectors.length; i += batchSize) {
      const batch = vectors.slice(i, i + batchSize)
      console.log(`[v0] Upserting batch ${Math.floor(i / batchSize) + 1} with ${batch.length} vectors`)

      await index.namespace(username).upsert(batch)
      upsertedCount += batch.length
      console.log(`[v0] Upserted batch ${Math.floor(i / batchSize) + 1}, total: ${upsertedCount}`)
    }

    console.log(`[v0] Successfully stored ${upsertedCount} vectors for ${username}`)
    return upsertedCount
  } catch (error) {
    console.error("[v0] Failed to store chunks in Pinecone:", error)
    throw new Error(`Failed to store chunks in Pinecone: ${error instanceof Error ? error.message : "Unknown error"}`)
  }
}

export async function searchSimilarChunks(query: string, username: string, topK = 10): Promise<string[]> {
  try {
    console.log(`[v0] Searching for similar chunks for user: ${username}, query: ${query.substring(0, 50)}...`)

    const index = await initializePinecone()

    // Generate query embedding
    console.log("[v0] Generating query embedding...")
    const queryEmbedding = await generateEmbeddings([query])
    console.log("[v0] Query embedding generated")

    // Search in user's namespace
    console.log(`[v0] Searching in namespace: ${username} with topK: ${topK}`)
    const searchResults = await index.namespace(username).query({
      vector: queryEmbedding[0],
      topK,
      includeMetadata: true,
    })

    console.log(`[v0] Search returned ${searchResults.matches?.length || 0} matches`)

    // Extract text chunks from results
    const chunks = searchResults.matches?.map((match) => match.metadata?.text as string).filter(Boolean) || []

    console.log(`[v0] Found ${chunks.length} relevant chunks for query`)
    return chunks
  } catch (error) {
    console.error("[v0] Failed to search Pinecone:", error)
    // Return empty array on error to allow fallback
    console.log("[v0] Returning empty array due to search error")
    return []
  }
}
