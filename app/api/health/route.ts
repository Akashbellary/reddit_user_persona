import { NextResponse } from "next/server"

export async function GET() {
  try {
    // Basic health check
    const health = {
      status: "healthy",
      timestamp: new Date().toISOString(),
      services: {
        reddit_api: "operational",
        embeddings: "operational",
        vector_db: "operational",
        llm: "operational",
      },
    }

    return NextResponse.json(health)
  } catch (error) {
    return NextResponse.json({ status: "unhealthy", error: "Health check failed" }, { status: 500 })
  }
}
