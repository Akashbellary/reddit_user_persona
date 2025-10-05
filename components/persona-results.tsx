import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import {
  User,
  Brain,
  MessageSquare,
  Target,
  Briefcase,
  MapPin,
  PenTool,
  TrendingUp,
  AlertTriangle,
  Calendar,
  Award,
  Shield,
  Users,
} from "lucide-react"
import type { PersonaAnalysis } from "@/lib/types"

interface PersonaResultsProps {
  persona: PersonaAnalysis
  stats?: {
    posts: number
    comments: number
    chunks: number
    stored_vectors: number
    relevant_chunks: number
  }
}

export function PersonaResults({ persona, stats }: PersonaResultsProps) {
  const { metadata, personality_traits, communication_style, interests, likely_profession } = persona

  return (
    <div className="space-y-8">
      {/* Header Section */}
      <div className="text-center">
        <div className="inline-flex items-center gap-2 bg-accent/10 text-accent px-4 py-2 rounded-full text-sm font-medium mb-4">
          <Brain className="w-4 h-4" />
          Analysis Complete
        </div>
        <h1 className="text-4xl font-bold mb-2">u/{metadata.username}</h1>
        <p className="text-muted-foreground text-lg">Psychological Profile & Persona Analysis</p>
      </div>

      {/* Stats Overview */}
      {stats && (
        <Card className="bg-gradient-to-r from-accent/5 to-accent/10 border-accent/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              Analysis Overview
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
              <div className="text-center">
                <div className="text-3xl font-bold text-accent mb-1">{stats.posts}</div>
                <div className="text-sm text-muted-foreground">Posts Analyzed</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-accent mb-1">{stats.comments}</div>
                <div className="text-sm text-muted-foreground">Comments Analyzed</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-accent mb-1">{stats.chunks}</div>
                <div className="text-sm text-muted-foreground">Text Segments</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-accent mb-1">{stats.stored_vectors}</div>
                <div className="text-sm text-muted-foreground">Vector Embeddings</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-accent mb-1">{stats.relevant_chunks}</div>
                <div className="text-sm text-muted-foreground">Key Insights</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid lg:grid-cols-3 gap-8">
        {/* Left Column */}
        <div className="lg:col-span-2 space-y-6">
          {/* Personality Traits */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="w-5 h-5" />
                Personality Traits
              </CardTitle>
              <CardDescription>Core psychological characteristics and behavioral patterns</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Social Orientation */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-medium">Social Orientation</span>
                  <Badge variant={personality_traits.introvert ? "secondary" : "default"}>
                    {personality_traits.introvert ? "Introvert" : "Extrovert"}
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground">
                  {personality_traits.introvert
                    ? personality_traits.introvert_reason
                    : personality_traits.extrovert_reason}
                </p>
              </div>

              <Separator />

              {/* Numerical Traits */}
              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Anger Level</span>
                    <span className="text-sm text-muted-foreground">{personality_traits.anger_level}/5</span>
                  </div>
                  <Progress value={(personality_traits.anger_level / 5) * 100} className="h-2" />
                  <p className="text-xs text-muted-foreground">{personality_traits.anger_level_reason}</p>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Empathy Level</span>
                    <span className="text-sm text-muted-foreground">{personality_traits.empathy_level}/5</span>
                  </div>
                  <Progress value={(personality_traits.empathy_level / 5) * 100} className="h-2" />
                  <p className="text-xs text-muted-foreground">{personality_traits.empathy_level_reason}</p>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Confidence Level</span>
                    <span className="text-sm text-muted-foreground">{personality_traits.confidence_level}/5</span>
                  </div>
                  <Progress value={(personality_traits.confidence_level / 5) * 100} className="h-2" />
                  <p className="text-xs text-muted-foreground">{personality_traits.confidence_level_reason}</p>
                </div>
              </div>

              <Separator />

              {/* Boolean Traits */}
              <div className="grid md:grid-cols-2 gap-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Analytical</span>
                  <Badge variant={personality_traits.analytical ? "default" : "secondary"}>
                    {personality_traits.analytical ? "Yes" : "No"}
                  </Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Judgmental</span>
                  <Badge variant={personality_traits.judgmental ? "destructive" : "secondary"}>
                    {personality_traits.judgmental ? "Yes" : "No"}
                  </Badge>
                </div>
              </div>

              {/* Humor Style */}
              {personality_traits.humor_style && (
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="font-medium">Humor Style</span>
                    <Badge variant="outline">{personality_traits.humor_style}</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">{personality_traits.humor_style_reason}</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Communication Style */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MessageSquare className="w-5 h-5" />
                Communication & Writing Style
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span className="font-medium">Communication Style</span>
                  <Badge>{communication_style}</Badge>
                </div>
                <p className="text-sm text-muted-foreground">{persona.communication_style_reason}</p>
              </div>

              <Separator />

              <div>
                <div className="flex items-center gap-2 mb-2">
                  <PenTool className="w-4 h-4" />
                  <span className="font-medium">Writing Style</span>
                  <Badge variant="outline">{persona.writing_style}</Badge>
                </div>
                <p className="text-sm text-muted-foreground">{persona.writing_style_reason}</p>
              </div>
            </CardContent>
          </Card>

          {/* Behavioral Insights */}
          <div className="grid md:grid-cols-3 gap-4">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <Target className="w-4 h-4" />
                  Goals & Needs
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {persona.goals_and_needs.map((goal, index) => (
                    <li key={index} className="text-sm flex items-start gap-2">
                      <div className="w-1.5 h-1.5 bg-accent rounded-full mt-2 flex-shrink-0" />
                      {goal}
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <Users className="w-4 h-4" />
                  Behavior & Habits
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {persona.behaviour_and_habits.map((habit, index) => (
                    <li key={index} className="text-sm flex items-start gap-2">
                      <div className="w-1.5 h-1.5 bg-accent rounded-full mt-2 flex-shrink-0" />
                      {habit}
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4" />
                  Frustrations
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {persona.frustrations.map((frustration, index) => (
                    <li key={index} className="text-sm flex items-start gap-2">
                      <div className="w-1.5 h-1.5 bg-destructive rounded-full mt-2 flex-shrink-0" />
                      {frustration}
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Right Column */}
        <div className="space-y-6">
          {/* User Metadata */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <User className="w-5 h-5" />
                Profile Information
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                {metadata.created_utc && (
                  <div className="flex items-center gap-2 text-sm">
                    <Calendar className="w-4 h-4 text-muted-foreground" />
                    <span className="text-muted-foreground">Joined:</span>
                    <span>{new Date(metadata.created_utc).toLocaleDateString()}</span>
                  </div>
                )}

                {metadata.total_karma && (
                  <div className="flex items-center gap-2 text-sm">
                    <Award className="w-4 h-4 text-muted-foreground" />
                    <span className="text-muted-foreground">Total Karma:</span>
                    <span className="font-medium">{metadata.total_karma.toLocaleString()}</span>
                  </div>
                )}

                {metadata.verified_email && (
                  <div className="flex items-center gap-2 text-sm">
                    <Shield className="w-4 h-4 text-green-500" />
                    <span>Verified Email</span>
                  </div>
                )}

                {metadata.is_gold && (
                  <div className="flex items-center gap-2 text-sm">
                    <Award className="w-4 h-4 text-yellow-500" />
                    <span>Reddit Premium</span>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Professional Profile */}
          {likely_profession && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Briefcase className="w-5 h-5" />
                  Professional Profile
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div>
                    <span className="text-sm text-muted-foreground">Likely Profession</span>
                    <div className="font-medium">{likely_profession}</div>
                  </div>
                  <p className="text-sm text-muted-foreground">{persona.likely_profession_reason}</p>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Location */}
          {persona.location_mentioned && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MapPin className="w-5 h-5" />
                  Location
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="font-medium">{persona.location_mentioned}</div>
              </CardContent>
            </Card>
          )}

          {/* Interests */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="w-5 h-5" />
                Interests & Topics
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {interests.map((interest, index) => (
                  <Badge key={index} variant="secondary">
                    {interest}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
