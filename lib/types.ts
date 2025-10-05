export interface RedditUser {
  username: string
  created_utc: string | null
  link_karma: number | null
  comment_karma: number | null
  total_karma: number | null
  verified_email: boolean | null
  is_gold: boolean | null
  is_mod: boolean | null
  has_subreddit: boolean | null
  profile_title: string | null
  public_description: string | null
  subscribers: number | null
  user_flair_text: string | null
}

export interface PersonalityTraits {
  introvert: boolean
  introvert_reason: string
  extrovert: boolean
  extrovert_reason: string
  anger_level: number
  anger_level_reason: string
  empathy_level: number
  empathy_level_reason: string
  judgmental: boolean
  judgmental_reason: string
  analytical: boolean
  analytical_reason: string
  humor_style: string | null
  humor_style_reason: string
  confidence_level: number
  confidence_level_reason: string
}

export interface PersonaAnalysis {
  metadata: RedditUser
  personality_traits: PersonalityTraits
  communication_style: string
  communication_style_reason: string
  interests: string[]
  likely_profession: string | null
  likely_profession_reason: string
  location_mentioned: string | null
  writing_style: string
  writing_style_reason: string
  behaviour_and_habits: string[]
  goals_and_needs: string[]
  frustrations: string[]
}

export interface AnalysisResponse {
  persona: PersonaAnalysis
  username: string
  stats: {
    posts: number
    comments: number
    chunks: number
    stored_vectors: number
    relevant_chunks: number
  }
}

export interface ApiError {
  error: string
  details?: string
}
