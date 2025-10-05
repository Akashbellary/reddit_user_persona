import OpenAI from "openai"

const client = new OpenAI({
  baseURL: "https://integrate.api.nvidia.com/v1",
  apiKey: process.env.NVIDIA_API_KEY!,
})

export interface PersonaData {
  metadata: any
  personality_traits: {
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

export async function generatePersonaFromChunks(chunks: string[], metadata: any): Promise<PersonaData | null> {
  if (chunks.length === 0) {
    console.log("[v0] No chunks provided for persona generation")
    return null
  }

  console.log(`[v0] Generating persona from ${chunks.length} chunks`)

  // Combine chunks with a reasonable limit
  const context = chunks.slice(0, 10).join("\n\n").substring(0, 3000)
  console.log(`[v0] Context length for LLM: ${context.length} characters`)

  const prompt = `You are an AI tasked with building a **detailed and structured psychological profile/persona** of a Reddit user 
based on their posts, comments, and metadata. 

Your goal is to return **ONLY valid JSON**, with the exact same keys, types, and structure every time, 
so it can be directly used in a dashboard. Do NOT include Markdown, explanations, or extra text outside JSON.

**Reddit History Context:**
"""
${context}
"""

**Output JSON structure:**

{
  "metadata": {
    "username": "string",
    "created_utc": "ISO 8601 string or null",
    "link_karma": number or null,
    "comment_karma": number or null,
    "total_karma": number or null,
    "verified_email": true/false/null,
    "is_gold": true/false/null,
    "is_mod": true/false/null,
    "has_subreddit": true/false/null,
    "profile_title": "string or null",
    "public_description": "string or null",
    "subscribers": number or null,
    "user_flair_text": "string or null"
  },
  "personality_traits": {
    "introvert": true/false,
    "introvert_reason": "string",
    "extrovert": true/false,
    "extrovert_reason": "string",
    "anger_level": 0-5,
    "anger_level_reason": "string",
    "empathy_level": 0-5,
    "empathy_level_reason": "string",
    "judgmental": true/false,
    "judgmental_reason": "string",
    "analytical": true/false,
    "analytical_reason": "string",
    "humor_style": "string or null",
    "humor_style_reason": "string",
    "confidence_level": 0-5,
    "confidence_level_reason": "string"
  },
  "communication_style": "string",
  "communication_style_reason": "string",
  "interests": ["list of strings, empty if unknown"],
  "likely_profession": "string or null",
  "likely_profession_reason": "string",
  "location_mentioned": "string or null",
  "writing_style": "string",
  "writing_style_reason": "string",
  "behaviour_and_habits": ["list of up to 5 strings, each describing a habit or behavior"],
  "goals_and_needs": ["list of up to 5 strings"],
  "frustrations": ["list of up to 5 strings"]
}

**Instructions:**
1. Always include all keys. Use \`null\`, empty strings, or empty lists if data is not available.
2. Infer personality, behavior, and interests strictly from the Reddit history provided.
3. Keep each field concise, factual, and structured.
4. Do not include any explanation, extra text, or Markdown—output must be directly parseable JSON.

Generate the JSON now.`

  try {
    console.log("[v0] Making LLM API call...")
    const response = await client.chat.completions.create({
      model: "nvidia/llama-3.3-nemotron-super-49b-v1",
      messages: [
        {
          role: "system",
          content: "ONLY return valid JSON. No Markdown, comments, or extra text.",
        },
        {
          role: "user",
          content: prompt,
        },
      ],
      temperature: 0.6,
      top_p: 0.9,
      max_tokens: 2048,
    })

    console.log("[v0] LLM API call completed")

    const content = response.choices[0]?.message?.content
    if (!content) {
      console.error("[v0] No content returned from LLM")
      return null
    }

    console.log("[v0] LLM response length:", content.length)
    console.log("[v0] LLM response preview:", content.substring(0, 200))

    // Try to parse JSON
    const parsed = tryParseJSON(content.trim())
    if (parsed) {
      // Ensure metadata is included
      parsed.metadata = { ...metadata, ...parsed.metadata }
      console.log("[v0] Persona JSON generated and parsed successfully")
      return parsed as PersonaData
    } else {
      console.error("[v0] Failed to parse persona JSON from LLM response")
      console.error("[v0] Raw LLM response:", content)
      return null
    }
  } catch (error) {
    console.error("[v0] LLM call failed:", error)
    return null
  }
}

function tryParseJSON(text: string): any {
  if (!text) {
    console.log("[v0] No text provided for JSON parsing")
    return null
  }

  console.log("[v0] Attempting to parse JSON...")

  try {
    const parsed = JSON.parse(text)
    console.log("[v0] JSON parsed successfully on first attempt")
    return parsed
  } catch (firstError) {
    console.log("[v0] First JSON parse failed, trying to extract JSON block...")

    // Try to extract JSON block
    const match = text.match(/\{[\s\S]*\}/)
    if (!match) {
      console.log("[v0] No JSON block found in text")
      return null
    }

    let block = match[0]
    console.log("[v0] Extracted JSON block length:", block.length)

    // Clean up common JSON issues
    block = block.replace(/,\s*}/g, "}").replace(/,\s*]/g, "]")

    try {
      const parsed = JSON.parse(block)
      console.log("[v0] JSON parsed successfully after cleanup")
      return parsed
    } catch (error) {
      console.error("[v0] Fallback JSON parse failed:", error)
      console.error("[v0] Cleaned JSON block:", block.substring(0, 500))
      return null
    }
  }
}
