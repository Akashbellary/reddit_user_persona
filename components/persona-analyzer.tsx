"use client"

import type React from "react"
import { useState } from "react"
import { apiClient } from "@/lib/api-client"

const PersonaAnalyzer: React.FC = () => {
  const [username, setUsername] = useState("spez") // Pre-fill with spez for testing
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [persona, setPersona] = useState<any | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!username.trim()) return

    console.log("[v0] Starting analysis for username:", username)
    setIsLoading(true)
    setError(null)
    setPersona(null)

    try {
      console.log("[v0] Making API request...")
      const response = await apiClient.analyzeUser(username.trim()) // Use real API client
      console.log("[v0] API response received:", response)
      setPersona(response)
    } catch (err) {
      console.error("[v0] Analysis failed:", err)
      setError(err instanceof Error ? err.message : "An unexpected error occurred")
    } finally {
      setIsLoading(false)
    }
  }

  const handleTestSpez = () => {
    setUsername("spez")
    setTimeout(() => {
      const form = document.querySelector("form")
      if (form) {
        form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }))
      }
    }, 100)
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Reddit Persona Analyzer</h1>

      <form onSubmit={handleSubmit} className="mb-6">
        <div className="flex gap-4 mb-4">
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Enter Reddit username (e.g., spez)"
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !username.trim()}
            className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? "Analyzing..." : "Analyze"}
          </button>
        </div>

        <button
          type="button"
          onClick={handleTestSpez}
          disabled={isLoading}
          className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 disabled:opacity-50 text-sm"
        >
          Test with spez (Reddit CEO)
        </button>
      </form>

      {isLoading && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
          <div className="flex items-center">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500 mr-3"></div>
            <div>
              <p className="font-medium">Analyzing Reddit user: {username}</p>
              <p className="text-sm text-gray-600">This may take 1-2 minutes...</p>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
          <h3 className="font-medium text-red-800 mb-2">Analysis Failed</h3>
          <p className="text-red-700">{error}</p>
        </div>
      )}

      {persona && (
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h2 className="text-2xl font-bold mb-4">Persona Analysis for u/{persona.username}</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-lg font-semibold mb-3">Basic Info</h3>
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Username:</strong> {persona.persona?.metadata?.username || persona.username}
                </p>
                <p>
                  <strong>Total Karma:</strong> {persona.persona?.metadata?.total_karma || "N/A"}
                </p>
                <p>
                  <strong>Account Age:</strong>{" "}
                  {persona.persona?.metadata?.created_utc
                    ? new Date(persona.persona.metadata.created_utc).toLocaleDateString()
                    : "N/A"}
                </p>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-3">Stats</h3>
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Posts Analyzed:</strong> {persona.stats?.posts || 0}
                </p>
                <p>
                  <strong>Comments Analyzed:</strong> {persona.stats?.comments || 0}
                </p>
                <p>
                  <strong>Text Chunks:</strong> {persona.stats?.chunks || 0}
                </p>
              </div>
            </div>
          </div>

          {persona.persona && (
            <div className="mt-6">
              <h3 className="text-lg font-semibold mb-3">Personality Analysis</h3>
              <div className="bg-gray-50 rounded-lg p-4">
                <pre className="text-sm overflow-auto whitespace-pre-wrap">
                  {JSON.stringify(persona.persona, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default PersonaAnalyzer
