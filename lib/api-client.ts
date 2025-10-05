import type { AnalysisResponse, ApiError } from "./types"

export class ApiClient {
  private baseUrl: string

  constructor(baseUrl = "") {
    this.baseUrl = baseUrl
  }

  async analyzeUser(username: string): Promise<AnalysisResponse> {
    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 120000) // 2 minute timeout

      console.log("[v0] Starting API request for username:", username)

      const response = await fetch(`${this.baseUrl}/api/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ username }),
        signal: controller.signal,
      })

      clearTimeout(timeoutId)

      console.log("[v0] Response status:", response.status)
      console.log("[v0] Response headers:", Object.fromEntries(response.headers.entries()))

      const contentType = response.headers.get("content-type")
      let data: any

      if (contentType && contentType.includes("application/json")) {
        data = await response.json()
      } else {
        const text = await response.text()
        console.log("[v0] Non-JSON response received:", text.substring(0, 200))

        if (!response.ok) {
          throw new Error(`Server returned ${response.status}: ${text.substring(0, 100)}`)
        }

        // Try to parse as JSON anyway in case content-type is wrong
        try {
          data = JSON.parse(text)
        } catch {
          throw new Error("Server returned invalid response format")
        }
      }

      if (!response.ok) {
        const error: ApiError = data

        // Handle specific error cases
        switch (response.status) {
          case 404:
            throw new Error("Reddit user not found or profile is private")
          case 429:
            throw new Error("Rate limit exceeded. Please try again in a few minutes")
          case 500:
            throw new Error("Server error occurred. Please try again later")
          default:
            throw new Error(error.error || "Analysis failed")
        }
      }

      console.log("[v0] Successful response received")
      return data as AnalysisResponse
    } catch (error) {
      console.log("[v0] API client error:", error)

      if (error instanceof Error) {
        if (error.name === "AbortError") {
          throw new Error("Request timed out. The analysis is taking too long, please try again")
        }
        throw error
      }
      throw new Error("Network error occurred. Please check your connection and try again")
    }
  }
}

export const apiClient = new ApiClient()
