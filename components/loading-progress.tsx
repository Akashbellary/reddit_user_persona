import { Progress } from "@/components/ui/progress"
import { Card, CardContent } from "@/components/ui/card"
import { Brain, Download, Search, Sparkles } from "lucide-react"

interface LoadingProgressProps {
  stage: string
  progress: number
}

export function LoadingProgress({ stage, progress }: LoadingProgressProps) {
  const stages = [
    { key: "fetching", label: "Fetching Reddit Data", icon: Download },
    { key: "processing", label: "Processing Content", icon: Search },
    { key: "analyzing", label: "AI Analysis", icon: Brain },
    { key: "generating", label: "Generating Persona", icon: Sparkles },
  ]

  const currentStageIndex = stages.findIndex((s) => s.key === stage)

  return (
    <Card className="max-w-2xl mx-auto shadow-xl border-0 bg-card/50 backdrop-blur">
      <CardContent className="pt-8 pb-6">
        <div className="text-center mb-6">
          <div className="inline-flex items-center gap-2 bg-accent/10 text-accent px-4 py-2 rounded-full text-sm font-medium mb-4">
            <Brain className="w-4 h-4 animate-pulse" />
            Analyzing Profile
          </div>
          <h3 className="text-xl font-semibold mb-2">Processing Your Request</h3>
          <p className="text-muted-foreground">This may take a few moments...</p>
        </div>

        <div className="space-y-4">
          <Progress value={progress} className="h-2" />

          <div className="grid grid-cols-2 gap-4">
            {stages.map((stageInfo, index) => {
              const Icon = stageInfo.icon
              const isActive = index === currentStageIndex
              const isComplete = index < currentStageIndex

              return (
                <div
                  key={stageInfo.key}
                  className={`flex items-center gap-3 p-3 rounded-lg transition-all ${
                    isActive
                      ? "bg-accent/10 text-accent"
                      : isComplete
                        ? "bg-muted/50 text-muted-foreground"
                        : "text-muted-foreground/50"
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? "animate-pulse" : ""}`} />
                  <span className="text-sm font-medium">{stageInfo.label}</span>
                  {isComplete && <div className="ml-auto w-2 h-2 bg-accent rounded-full" />}
                </div>
              )
            })}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
