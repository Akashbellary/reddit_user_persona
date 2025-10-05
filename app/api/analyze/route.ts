import { NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    console.log("[v0] === PROXY TO FLASK BACKEND ===")

    // Get the username from the request
    const { username } = await request.json()
    console.log("[v0] Received username:", username)

    if (!username) {
      console.error("[v0] No username provided")
      return NextResponse.json({ error: "Username is required" }, { status: 400 })
    }

    // Proxy the request to the Flask backend
    // Assuming Flask is running on port 5000 (default)
    const flaskUrl = process.env.FLASK_BACKEND_URL || "http://localhost:5000"

    console.log(`[v0] Forwarding request to Flask backend at ${flaskUrl}/`)

    const flaskResponse = await fetch(`${flaskUrl}/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json",  // Request JSON response from Flask
      },
      body: JSON.stringify({ username }),
    })

    console.log(`[v0] Flask backend response status: ${flaskResponse.status}`)

    // Handle errors from Flask
    if (!flaskResponse.ok) {
      const errorText = await flaskResponse.text()
      console.error(`[v0] Flask backend error (${flaskResponse.status}):`, errorText)

      // Parse error if it's JSON
      try {
        const errorData = JSON.parse(errorText)
        return NextResponse.json({ error: errorData.error || "Backend error occurred" }, { status: flaskResponse.status })
      } catch {
        // If not JSON, return the text as error message
        return NextResponse.json({ error: errorText || "Backend error occurred" }, { status: flaskResponse.status })
      }
    }

    // Process successful response from Flask
    const flaskData = await flaskResponse.json()
    console.log("[v0] Flask returned JSON data successfully")

    // Return the data directly since Flask now returns the correct format
    return NextResponse.json(flaskData)

  } catch (error) {
    console.error("[v0] === PROXY ERROR ===", error)

    return NextResponse.json({
      error: "Failed to connect to analysis backend",
      details: error instanceof Error ? error.message : "Unknown error"
    }, { status: 500 })
  }
}

// Transform Flask backend data to match frontend expectations
function transformFlaskData(flaskData: any, username: string) {
  // The Flask backend returns the persona directly in the template
  // We need to structure it according to the frontend's AnalysisResponse format

  // If flaskData is the persona object itself
  if (flaskData.personality_traits || flaskData.metadata) {
    return {
      persona: flaskData,
      username: flaskData.metadata?.username || username,
      stats: {
        posts: 0, // These would need to come from Flask
        comments: 0,
        chunks: 0,
        stored_vectors: 0,
        relevant_chunks: 0
      }
    }
  }

  // If we get a different structure, adapt accordingly
  return {
    persona: flaskData.persona || flaskData,
    username: username,
    stats: {
      posts: 0,
      comments: 0,
      chunks: 0,
      stored_vectors: 0,
      relevant_chunks: 0
    }
  }
}

