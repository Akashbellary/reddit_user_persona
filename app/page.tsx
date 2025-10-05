"use client"

import type React from "react"

import { useState, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Brain, Users, MessageSquare, TrendingUp, Target, AlertCircle, ArrowLeft, Info } from "lucide-react"
import { LoadingProgress } from "@/components/loading-progress"
import { PersonaResults } from "@/components/persona-results"
import { apiClient } from "@/lib/api-client"
import { useToast } from "@/hooks/use-toast"
import type { PersonaAnalysis } from "@/lib/types"

interface AnalysisStats {
  posts: number
  comments: number
  chunks: number
  stored_vectors: number
  relevant_chunks: number
}

export default function HomePage() {
  const [username, setUsername] = useState("")
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisStage, setAnalysisStage] = useState("fetching")
  const [progress, setProgress] = useState(0)
  const [persona, setPersona] = useState<PersonaAnalysis | null>(null)
  const [stats, setStats] = useState<AnalysisStats | null>(null)
  const [error, setError] = useState("")
  const { toast } = useToast()

  const handleTestSpez = useCallback(async () => {
    console.log("[v0] Testing with spez...")
    setUsername("spez")

    // Trigger analysis immediately
    setIsAnalyzing(true)
    setError("")
    setPersona(null)
    setStats(null)
    setProgress(0)
    setAnalysisStage("fetching")

    // Simulate progress updates
    const progressInterval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 90) return prev
        return prev + Math.random() * 10
      })
    }, 500)

    const stageInterval = setInterval(() => {
      setAnalysisStage((prev) => {
        const stages = ["fetching", "processing", "analyzing", "generating"]
        const currentIndex = stages.indexOf(prev)
        if (currentIndex < stages.length - 1) {
          return stages[currentIndex + 1]
        }
        return prev
      })
    }, 2000)

    try {
      const result = await apiClient.analyzeUser("spez")

      clearInterval(progressInterval)
      clearInterval(stageInterval)
      setProgress(100)

      // Brief delay to show completion
      setTimeout(() => {
        setPersona(result.persona)
        setStats(result.stats)
        toast({
          title: "Analysis Complete!",
          description: `Successfully analyzed u/${result.username} with ${result.stats.posts} posts and ${result.stats.comments} comments.`,
        })
      }, 500)
    } catch (err) {
      clearInterval(progressInterval)
      clearInterval(stageInterval)
      const errorMessage = err instanceof Error ? err.message : "Something went wrong"
      setError(errorMessage)
      toast({
        title: "Analysis Failed",
        description: errorMessage,
        variant: "destructive",
      })
    } finally {
      setIsAnalyzing(false)
    }
  }, [toast])

  const handleAnalyze = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault()
      if (!username.trim()) return

      setIsAnalyzing(true)
      setError("")
      setPersona(null)
      setStats(null)
      setProgress(0)
      setAnalysisStage("fetching")

      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) return prev
          return prev + Math.random() * 10
        })
      }, 500)

      const stageInterval = setInterval(() => {
        setAnalysisStage((prev) => {
          const stages = ["fetching", "processing", "analyzing", "generating"]
          const currentIndex = stages.indexOf(prev)
          if (currentIndex < stages.length - 1) {
            return stages[currentIndex + 1]
          }
          return prev
        })
      }, 2000)

      try {
        const result = await apiClient.analyzeUser(username.trim())

        clearInterval(progressInterval)
        clearInterval(stageInterval)
        setProgress(100)

        // Brief delay to show completion
        setTimeout(() => {
          setPersona(result.persona)
          setStats(result.stats)
          toast({
            title: "Analysis Complete!",
            description: `Successfully analyzed u/${result.username} with ${result.stats.posts} posts and ${result.stats.comments} comments.`,
          })
        }, 500)
      } catch (err) {
        clearInterval(progressInterval)
        clearInterval(stageInterval)
        const errorMessage = err instanceof Error ? err.message : "Something went wrong"
        setError(errorMessage)
        toast({
          title: "Analysis Failed",
          description: errorMessage,
          variant: "destructive",
        })
      } finally {
        setIsAnalyzing(false)
      }
    },
    [username, toast],
  )

  const handleBack = useCallback(() => {
    setPersona(null)
    setStats(null)
    setUsername("")
    setError("")
  }, [])

  if (persona) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-background via-background to-accent/5">
        <div className="container mx-auto px-4 py-8">
          <Button onClick={handleBack} variant="outline" className="mb-6 bg-transparent">
            <ArrowLeft className="w-4 h-4 mr-2" />
            New Analysis
          </Button>
          <PersonaResults persona={persona} stats={stats} />
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-accent/5">
      <div className="container mx-auto px-4 py-16">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <div className="inline-flex items-center gap-2 bg-accent/10 text-accent px-4 py-2 rounded-full text-sm font-medium mb-6">
            <Brain className="w-4 h-4" />
            AI-Powered Analysis
          </div>

          <h1 className="text-5xl md:text-6xl font-bold text-balance mb-6 bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent leading-tight">
            Discover the Psychology Behind Reddit Users
          </h1>

          <p className="text-xl text-muted-foreground text-balance mb-8 max-w-2xl mx-auto leading-relaxed">
            Analyze Reddit profiles to generate detailed psychological personas using advanced AI. Understand
            personality traits, communication styles, and behavioral patterns.
          </p>
        </div>

        {isAnalyzing ? (
          <LoadingProgress stage={analysisStage} progress={progress} />
        ) : (
          <Card className="max-w-2xl mx-auto shadow-xl border-0 bg-card/50 backdrop-blur">
            <CardHeader className="text-center pb-6">
              <CardTitle className="text-2xl">Start Your Analysis</CardTitle>
              <CardDescription className="text-base">Enter a Reddit username or profile URL to begin</CardDescription>
            </CardHeader>

            <CardContent>
              <form onSubmit={handleAnalyze} className="space-y-6">
                <div className="space-y-2">
                  <Input
                    type="text"
                    placeholder="e.g., u/spez or https://reddit.com/user/spez"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    className="h-12 text-base"
                    disabled={isAnalyzing}
                    aria-label="Reddit username or profile URL"
                  />
                  {error && (
                    <div className="flex items-center gap-2 text-destructive text-sm bg-destructive/10 p-3 rounded-lg">
                      <AlertCircle className="w-4 h-4" />
                      {error}
                    </div>
                  )}
                </div>

                <Button
                  type="submit"
                  className="w-full h-12 text-base font-medium"
                  disabled={isAnalyzing || !username.trim()}
                >
                  <Brain className="w-4 h-4 mr-2" />
                  Generate Persona
                </Button>

                <Button
                  type="button"
                  onClick={handleTestSpez}
                  variant="outline"
                  className="w-full h-12 text-base font-medium bg-transparent"
                  disabled={isAnalyzing}
                >
                  🧪 Test with u/spez (Reddit CEO)
                </Button>
              </form>

              <div className="mt-6 p-4 bg-muted/50 rounded-lg">
                <div className="flex items-start gap-2">
                  <Info className="w-4 h-4 text-muted-foreground mt-0.5 flex-shrink-0" />
                  <div className="text-sm text-muted-foreground">
                    <p className="font-medium mb-1">Privacy & Ethics</p>
                    <p>
                      This tool analyzes only public Reddit posts and comments. All analysis is performed using AI and
                      should be considered for entertainment and research purposes only.
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto mt-16">
          <Card className="text-center border-0 bg-card/30 hover:bg-card/40 transition-colors">
            <CardContent className="pt-6">
              <Users className="w-8 h-8 text-accent mx-auto mb-4" />
              <h3 className="font-semibold mb-2">Personality Analysis</h3>
              <p className="text-sm text-muted-foreground">
                Discover introversion, empathy levels, confidence, and behavioral patterns
              </p>
            </CardContent>
          </Card>

          <Card className="text-center border-0 bg-card/30 hover:bg-card/40 transition-colors">
            <CardContent className="pt-6">
              <MessageSquare className="w-8 h-8 text-accent mx-auto mb-4" />
              <h3 className="font-semibold mb-2">Communication Style</h3>
              <p className="text-sm text-muted-foreground">
                Analyze writing patterns, humor style, and interaction preferences
              </p>
            </CardContent>
          </Card>

          <Card className="text-center border-0 bg-card/30 hover:bg-card/40 transition-colors">
            <CardContent className="pt-6">
              <Brain className="w-8 h-8 text-accent mx-auto mb-4" />
              <h3 className="font-semibold mb-2">AI Insights</h3>
              <p className="text-sm text-muted-foreground">
                Powered by advanced language models and vector similarity analysis
              </p>
            </CardContent>
          </Card>
        </div>

        <div className="grid md:grid-cols-2 gap-8 max-w-3xl mx-auto mt-16">
          <div className="text-center">
            <TrendingUp className="w-12 h-12 text-accent mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">Behavioral Insights</h3>
            <p className="text-muted-foreground">
              Identify goals, needs, frustrations, and likely profession based on posting patterns
            </p>
          </div>

          <div className="text-center">
            <Target className="w-12 h-12 text-accent mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">Comprehensive Profiling</h3>
            <p className="text-muted-foreground">
              Generate detailed personas with interests, habits, and psychological traits
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
