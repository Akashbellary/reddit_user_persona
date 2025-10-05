export interface RedditPost {
  id: string
  subreddit: string
  created_utc: string
  score: number
  num_comments: number
  text: string
}

export interface RedditComment {
  id: string
  subreddit: string
  created_utc: string
  score: number
  text: string
}

export interface UserMetadata {
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

class RedditAPIClient {
  private accessToken: string | null = null
  private tokenExpiry = 0

  constructor(
    private clientId: string,
    private clientSecret: string,
    private userAgent: string,
  ) {}

  async getAccessToken(): Promise<string> {
    console.log("[v0] Getting Reddit access token...")

    if (this.accessToken && Date.now() < this.tokenExpiry) {
      console.log("[v0] Using cached access token")
      return this.accessToken
    }

    const auth = Buffer.from(`${this.clientId}:${this.clientSecret}`).toString("base64")

    try {
      const response = await fetch("https://www.reddit.com/api/v1/access_token", {
        method: "POST",
        headers: {
          Authorization: `Basic ${auth}`,
          "User-Agent": this.userAgent,
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: "grant_type=client_credentials",
      })

      console.log("[v0] Token request status:", response.status)

      if (!response.ok) {
        const errorText = await response.text()
        console.error("[v0] Token request failed:", errorText)
        throw new Error(`Failed to get access token: ${response.status} ${errorText}`)
      }

      const data = await response.json()
      console.log("[v0] Token response received:", {
        token_type: data.token_type,
        expires_in: data.expires_in,
        has_access_token: !!data.access_token,
      })

      this.accessToken = data.access_token
      this.tokenExpiry = Date.now() + data.expires_in * 1000 - 60000 // 1 minute buffer

      return this.accessToken
    } catch (error) {
      console.error("[v0] Error getting access token:", error)
      throw new Error(
        `Failed to authenticate with Reddit API: ${error instanceof Error ? error.message : "Unknown error"}`,
      )
    }
  }

  async makeRequest(endpoint: string): Promise<any> {
    const token = await this.getAccessToken()
    const url = `https://oauth.reddit.com${endpoint}`

    console.log("[v0] Making Reddit API request to:", endpoint)

    try {
      const response = await fetch(url, {
        headers: {
          Authorization: `Bearer ${token}`,
          "User-Agent": this.userAgent,
        },
      })

      console.log("[v0] API request status:", response.status)

      if (!response.ok) {
        const errorText = await response.text()
        console.error("[v0] API request failed:", errorText)

        if (response.status === 404) {
          throw new Error("User not found or profile is private")
        }
        if (response.status === 403) {
          throw new Error("Access denied - profile may be private")
        }
        if (response.status === 429) {
          throw new Error("Rate limit exceeded")
        }

        throw new Error(`Reddit API error: ${response.status} ${errorText}`)
      }

      const data = await response.json()
      console.log("[v0] API request successful, data keys:", Object.keys(data))
      return data
    } catch (error) {
      console.error("[v0] Error making Reddit request:", error)
      throw error
    }
  }
}

let redditClient: RedditAPIClient | null = null

function getRedditClient(): RedditAPIClient {
  if (!redditClient) {
    console.log("[v0] Initializing Reddit API client...")

    const userAgent = process.env.REDDIT_USER_AGENT
    const clientId = process.env.REDDIT_CLIENT_ID
    const clientSecret = process.env.REDDIT_SECRET

    console.log("[v0] Environment variables check:", {
      userAgent: !!userAgent,
      clientId: !!clientId,
      clientSecret: !!clientSecret,
    })

    if (!userAgent || !clientId || !clientSecret) {
      throw new Error("Missing Reddit API credentials. Please check environment variables.")
    }

    redditClient = new RedditAPIClient(clientId, clientSecret, userAgent)
    console.log("[v0] Reddit API client initialized successfully")
  }

  return redditClient
}

export async function getRedditData(
  username: string,
  postsLimit = 100,
  commentsLimit = 200,
): Promise<{ posts: RedditPost[]; comments: RedditComment[]; metadata: UserMetadata }> {
  try {
    console.log(`[v0] Starting Reddit data fetch for: ${username}`)

    const client = getRedditClient()

    // Fetch user info
    console.log("[v0] Fetching user metadata...")
    const userResponse = await client.makeRequest(`/user/${username}/about`)
    const userData = userResponse.data

    console.log(`[v0] Successfully found user: ${username}`)

    // Build metadata
    const metadata: UserMetadata = {
      username: userData.name,
      created_utc: userData.created_utc ? new Date(userData.created_utc * 1000).toISOString() : null,
      link_karma: userData.link_karma || null,
      comment_karma: userData.comment_karma || null,
      total_karma: (userData.link_karma || 0) + (userData.comment_karma || 0),
      verified_email: userData.has_verified_email || null,
      is_gold: userData.is_gold || null,
      is_mod: userData.is_mod || null,
      has_subreddit: !!userData.subreddit,
      profile_title: userData.subreddit?.title || null,
      public_description: userData.subreddit?.public_description || null,
      subscribers: userData.subreddit?.subscribers || null,
      user_flair_text: null,
    }

    console.log("[v0] User metadata collected:", {
      username: metadata.username,
      karma: metadata.total_karma,
      created: metadata.created_utc,
    })

    // Fetch posts
    console.log("[v0] Fetching user posts...")
    const posts: RedditPost[] = []
    try {
      const postsResponse = await client.makeRequest(`/user/${username}/submitted?limit=${postsLimit}`)

      if (postsResponse.data && postsResponse.data.children) {
        for (const child of postsResponse.data.children) {
          const post = child.data
          const title = post.title || ""
          const selftext = post.selftext || ""

          posts.push({
            id: post.id,
            subreddit: post.subreddit,
            created_utc: new Date(post.created_utc * 1000).toISOString(),
            score: post.score,
            num_comments: post.num_comments,
            text: `[POST] ${title}\n\n${selftext}`.trim(),
          })
        }
      }

      console.log(`[v0] Fetched ${posts.length} posts`)
    } catch (error) {
      console.warn("[v0] Error fetching posts:", error)
    }

    // Fetch comments
    console.log("[v0] Fetching user comments...")
    const comments: RedditComment[] = []
    try {
      const commentsResponse = await client.makeRequest(`/user/${username}/comments?limit=${commentsLimit}`)

      if (commentsResponse.data && commentsResponse.data.children) {
        for (const child of commentsResponse.data.children) {
          const comment = child.data
          const body = comment.body || ""

          comments.push({
            id: comment.id,
            subreddit: comment.subreddit,
            created_utc: new Date(comment.created_utc * 1000).toISOString(),
            score: comment.score,
            text: `[COMMENT] ${body}`.trim(),
          })
        }
      }

      console.log(`[v0] Fetched ${comments.length} comments`)
    } catch (error) {
      console.warn("[v0] Error fetching comments:", error)
    }

    console.log(`[v0] Total data fetched - Posts: ${posts.length}, Comments: ${comments.length}`)
    return { posts, comments, metadata }
  } catch (error) {
    console.error("[v0] Reddit API error:", error)

    if (error instanceof Error) {
      if (error.message.includes("not found") || error.message.includes("404")) {
        throw new Error(`Reddit user '${username}' not found or profile is private`)
      }
      if (error.message.includes("403") || error.message.includes("Forbidden")) {
        throw new Error(`Access denied to Reddit user '${username}' - profile may be private`)
      }
      if (error.message.includes("429") || error.message.includes("rate")) {
        throw new Error("Reddit API rate limit exceeded. Please try again later.")
      }
    }

    throw new Error(
      `Failed to fetch Reddit data for '${username}': ${error instanceof Error ? error.message : "Unknown error"}`,
    )
  }
}

export function chunkText(text: string, maxLength = 500): string[] {
  if (!text) return []

  const paragraphs = text.split("\n\n").filter((p) => p.trim())
  const chunks: string[] = []
  let current = ""

  for (const para of paragraphs) {
    if (current.length + para.length + 2 <= maxLength) {
      current += para + "\n\n"
    } else {
      if (current) {
        chunks.push(current.trim())
      }
      current = para + "\n\n"
    }
  }

  if (current) {
    chunks.push(current.trim())
  }

  return chunks
}

export function combineUserData(posts: RedditPost[], comments: RedditComment[], metadata: UserMetadata): string {
  const sections = []

  // Add metadata section
  sections.push("🔸 METADATA:")
  sections.push(JSON.stringify(metadata, null, 2))

  // Add posts section
  if (posts.length > 0) {
    sections.push("\n\n🔸 POSTS:\n")
    sections.push(posts.map((p) => p.text).join("\n\n"))
  }

  // Add comments section
  if (comments.length > 0) {
    sections.push("\n\n🔸 COMMENTS:\n")
    sections.push(comments.map((c) => c.text).join("\n\n"))
  }

  return sections.join("")
}
