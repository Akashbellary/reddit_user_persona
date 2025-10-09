import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import Snoowrap from 'snoowrap';
import { Pinecone } from '@pinecone-database/pinecone';
import OpenAI from 'openai';
import { v4 as uuidv4 } from 'uuid';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

// Get current directory in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load .env file from current directory
const envPath = join(__dirname, '.env');
const envResult = dotenv.config({ path: envPath });

if (envResult.error) {
    console.error('❌ Error loading .env file:', envResult.error);
    console.error('Looking for .env at:', envPath);
} else {
    console.log('✓ .env file loaded from:', envPath);
}

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Configuration
const CONFIG = {
    reddit: {
        clientId: process.env.REDDIT_CLIENT_ID,
        clientSecret: process.env.REDDIT_SECRET,
        userAgent: process.env.REDDIT_USER_AGENT,
    },
    pinecone: {
        apiKey: process.env.PINECONE_API_KEY,
        indexName: 'reddit-persona-2048',
        dimension: 2048, // nvidia/llama-3.2-nv-embedqa-1b-v2 produces 2048-dim vectors
    },
    nvidia: {
        apiKey: process.env.NVIDIA_API_KEY,
        embeddingModel: 'nvidia/llama-3.2-nv-embedqa-1b-v2',
        llmModel: process.env.LLM_MODEL_NAME || 'meta/llama-3.1-70b-instruct',
    },
    port: process.env.PORT || 5000,
};

// Validate required environment variables
const required = [
    'REDDIT_CLIENT_ID',
    'REDDIT_SECRET',
    'REDDIT_USER_AGENT',
    'PINECONE_API_KEY',
    'NVIDIA_API_KEY',
];

console.log('\n🔍 Environment Variables Check:');
for (const key of required) {
    const value = process.env[key];
    if (!value) {
        console.error(`❌ Missing: ${key}`);
        throw new Error(`Missing required environment variable: ${key}`);
    } else {
        console.log(`✓ ${key}: ${value.substring(0, 10)}...`);
    }
}
console.log('');

// Initialize clients
let reddit;
let pinecone;
let pineconeIndex;
let openaiClient;

async function initializeClients() {
    try {
        // Reddit client - Manual OAuth token retrieval
        const tokenResponse = await fetch('https://www.reddit.com/api/v1/access_token', {
            method: 'POST',
            headers: {
                'Authorization': 'Basic ' + Buffer.from(`${CONFIG.reddit.clientId}:${CONFIG.reddit.clientSecret}`).toString('base64'),
                'Content-Type': 'application/x-www-form-urlencoded',
                'User-Agent': CONFIG.reddit.userAgent,
            },
            body: 'grant_type=client_credentials',
        });

        if (!tokenResponse.ok) {
            throw new Error(`Reddit OAuth failed: ${tokenResponse.status} ${tokenResponse.statusText}`);
        }

        const tokenData = await tokenResponse.json();

        // Initialize Snoowrap with access token
        reddit = new Snoowrap({
            userAgent: CONFIG.reddit.userAgent,
            accessToken: tokenData.access_token,
        });

        console.log('✓ Reddit client initialized with OAuth token');

        // Pinecone client (latest SDK)
        pinecone = new Pinecone({
            apiKey: CONFIG.pinecone.apiKey,
        });

        // Check if index exists, create if not
        const indexes = await pinecone.listIndexes();
        const indexExists = indexes.indexes?.some(idx => idx.name === CONFIG.pinecone.indexName);

        if (!indexExists) {
            console.log(`Creating Pinecone index: ${CONFIG.pinecone.indexName}`);
            await pinecone.createIndex({
                name: CONFIG.pinecone.indexName,
                dimension: CONFIG.pinecone.dimension,
                metric: 'cosine',
                spec: {
                    serverless: {
                        cloud: 'aws',
                        region: 'us-east-1',
                    },
                },
            });
            console.log('⏳ Waiting for index to initialize...');
            await new Promise(resolve => setTimeout(resolve, 5000));
        }

        pineconeIndex = pinecone.index(CONFIG.pinecone.indexName);
        console.log('✓ Pinecone client initialized');

        // OpenAI-compatible client for NVIDIA
        openaiClient = new OpenAI({
            baseURL: 'https://integrate.api.nvidia.com/v1',
            apiKey: CONFIG.nvidia.apiKey,
        });
        console.log('✓ NVIDIA LLM client initialized');

    } catch (error) {
        console.error('Failed to initialize clients:', error);
        throw error;
    }
}

// Utility: Extract username from various formats
function extractUsername(input) {
    if (!input) return null;

    const trimmed = input.trim().split(/\s+/)[0];

    // Check URL patterns
    const urlMatch = trimmed.match(/reddit\.com\/(?:user|u)\/([A-Za-z0-9_-]+)/i);
    if (urlMatch) return urlMatch[1];

    // Remove leading u/ or /u/ or user/ or /user/
    const cleaned = trimmed.replace(/^\/?(?:u|user)\//i, '').replace(/^\//, '');

    // Validate username format
    if (/^[A-Za-z0-9_-]{1,50}$/.test(cleaned)) {
        return cleaned;
    }

    return null;
}

// Fetch Reddit data (in-memory, no file storage)
async function getRedditData(username, postsLimit = 100, commentsLimit = 200) {
    console.log(`Fetching Reddit data for: ${username}`);

    try {
        const user = await reddit.getUser(username).fetch();

        // Build metadata
        const metadata = {
            username: user.name,
            created_utc: user.created_utc ? new Date(user.created_utc * 1000).toISOString() : null,
            link_karma: user.link_karma || null,
            comment_karma: user.comment_karma || null,
            total_karma: (user.link_karma || 0) + (user.comment_karma || 0),
            verified_email: user.has_verified_email || null,
            is_gold: user.is_gold || null,
            is_mod: user.is_mod || null,
        };

        // Fetch posts
        const posts = [];
        try {
            const submissions = await user.getSubmissions({ limit: postsLimit });
            for (const post of submissions) {
                posts.push({
                    id: post.id,
                    subreddit: post.subreddit.display_name,
                    created_utc: new Date(post.created_utc * 1000).toISOString(),
                    score: post.score,
                    num_comments: post.num_comments,
                    text: `[POST] ${post.title}\n\n${post.selftext || ''}`.trim(),
                });
            }
            console.log(`✓ Fetched ${posts.length} posts`);
        } catch (err) {
            console.error('Error fetching posts:', err);
        }

        // Fetch comments
        const comments = [];
        try {
            const userComments = await user.getComments({ limit: commentsLimit });
            for (const comment of userComments) {
                comments.push({
                    id: comment.id,
                    subreddit: comment.subreddit.display_name,
                    created_utc: new Date(comment.created_utc * 1000).toISOString(),
                    score: comment.score,
                    text: `[COMMENT] ${comment.body || ''}`.trim(),
                });
            }
            console.log(`✓ Fetched ${comments.length} comments`);
        } catch (err) {
            console.error('Error fetching comments:', err);
        }

        return { posts, comments, metadata };

    } catch (error) {
        console.error(`Error fetching Reddit data for ${username}:`, error);
        throw new Error(`Reddit user '${username}' not found or inaccessible`);
    }
}

// Create text chunks from content (reduced size to fit within 512 token limit)
function chunkText(text, maxLen = 300) {
    if (!text) return [];

    const paragraphs = text
        .split('\n\n')
        .map(p => p.trim())
        .filter(p => p.length > 0);

    const chunks = [];
    let current = '';

    for (const para of paragraphs) {
        if (current.length + para.length + 2 <= maxLen) {
            current += para + '\n\n';
        } else {
            if (current) chunks.push(current.trim());
            current = para + '\n\n';
        }
    }

    if (current) chunks.push(current.trim());

    console.log(`✓ Created ${chunks.length} text chunks`);
    return chunks;
}

// Get embeddings from NVIDIA NIM API (batched to handle token limits)
async function getEmbeddings(texts, inputType = 'passage') {
    try {
        const batchSize = 20; // Process 20 chunks at a time to avoid rate limits
        const allEmbeddings = [];

        for (let i = 0; i < texts.length; i += batchSize) {
            const batch = texts.slice(i, i + batchSize);

            // Use raw fetch with input_type parameter for asymmetric embedding model
            const response = await fetch('https://integrate.api.nvidia.com/v1/embeddings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${CONFIG.nvidia.apiKey}`,
                },
                body: JSON.stringify({
                    model: CONFIG.nvidia.embeddingModel,
                    input: batch,
                    input_type: inputType, // Required for asymmetric models: 'query' or 'passage'
                    encoding_format: 'float'
                }),
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error('NVIDIA API Error Response:', errorText);
                throw new Error(`NVIDIA API error: ${response.status} ${response.statusText} - ${errorText}`);
            }

            const data = await response.json();
            allEmbeddings.push(...data.data.map(item => item.embedding));

            // Small delay between batches to respect rate limits
            if (i + batchSize < texts.length) {
                await new Promise(resolve => setTimeout(resolve, 100));
            }
        }

        return allEmbeddings;
    } catch (error) {
        console.error('Error getting embeddings:', error);
        throw error;
    }
}

// Embed and upsert to Pinecone (uses namespace per user, batched to stay under 2MB limit)
async function embedAndUpsert(chunks, username) {
    if (!chunks || chunks.length === 0) return 0;

    try {
        console.log(`Generating embeddings for ${chunks.length} chunks...`);
        // Use 'passage' input_type for storing documents (asymmetric embedding)
        const embeddings = await getEmbeddings(chunks, 'passage');

        const vectors = chunks.map((chunk, i) => ({
            id: uuidv4(),
            values: embeddings[i],
            metadata: { text: chunk },
        }));

        // Batch upsert to stay under Pinecone's 2MB request size limit
        // With 2048-dim vectors, ~50 vectors per batch keeps us well under 2MB
        const upsertBatchSize = 50;
        let totalUpserted = 0;

        for (let i = 0; i < vectors.length; i += upsertBatchSize) {
            const batch = vectors.slice(i, i + upsertBatchSize);
            await pineconeIndex.namespace(username).upsert(batch);
            totalUpserted += batch.length;
            console.log(`  ✓ Upserted batch ${Math.floor(i / upsertBatchSize) + 1}: ${batch.length} vectors`);

            // Small delay between batches to respect rate limits
            if (i + upsertBatchSize < vectors.length) {
                await new Promise(resolve => setTimeout(resolve, 100));
            }
        }

        console.log(`✓ Total upserted: ${totalUpserted} vectors for user '${username}'`);
        return totalUpserted;
    } catch (error) {
        console.error('Error in embedAndUpsert:', error);
        throw error;
    }
}

// Search Pinecone for relevant chunks
async function searchChunks(query, topK = 10, usernameNamespace = null, rawChunks = null) {
    try {
        // Use 'query' input_type for search queries (asymmetric embedding)
        const [queryEmbedding] = await getEmbeddings([query], 'query');

        // Use Pinecone query API (latest SDK) with namespace
        const queryResponse = await pineconeIndex.namespace(usernameNamespace).query({
            vector: queryEmbedding,
            topK,
            includeMetadata: true,
        });

        const matches = queryResponse.matches || [];
        const chunks = matches.map(match => match.metadata?.text || '').filter(t => t);

        if (chunks.length === 0 && rawChunks) {
            console.warn('No chunks from Pinecone, using raw chunks');
            return rawChunks.slice(0, topK);
        }

        return chunks;
    } catch (error) {
        console.error('Error searching Pinecone:', error);
        if (rawChunks) return rawChunks.slice(0, topK);
        throw error;
    }
}

// Try to parse JSON from LLM response
function tryParseJsonFallback(text) {
    if (!text) return null;

    try {
        return JSON.parse(text);
    } catch {
        // Try to extract JSON block
        const match = text.match(/\{[\s\S]*\}/);
        if (!match) return null;

        let block = match[0];
        // Remove trailing commas
        block = block.replace(/,(\s*[}\]])/g, '$1');

        try {
            return JSON.parse(block);
        } catch {
            console.error('Failed to parse JSON from LLM response');
            return null;
        }
    }
}

// Generate persona using LLM
async function generatePersonaJson(chunks) {
    if (!chunks || chunks.length === 0) return null;

    const context = chunks.join('\n\n').slice(0, 3000);

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
1. Always include all keys. Use null, empty strings, or empty lists if data is not available.
2. Infer personality, behavior, and interests strictly from the Reddit history provided.
3. Keep each field concise, factual, and structured.
4. Do not include any explanation, extra text, or Markdown—output must be directly parseable JSON.

Generate the JSON now.`;

    try {
        const response = await openaiClient.chat.completions.create({
            model: CONFIG.nvidia.llmModel,
            messages: [
                { role: 'system', content: 'ONLY return valid JSON. No Markdown, comments, or extra text.' },
                { role: 'user', content: prompt },
            ],
            temperature: 0.6,
            top_p: 0.9,
            max_tokens: 2048,
        });

        const text = response.choices[0]?.message?.content;

        if (!text) {
            console.error('No content from LLM');
            return null;
        }

        const parsed = tryParseJsonFallback(text.trim());

        if (parsed) {
            console.log('✓ Persona JSON generated');
            return parsed;
        } else {
            console.error('Failed to parse persona JSON');
            return null;
        }
    } catch (error) {
        console.error('LLM call failed:', error);
        throw error;
    }
}

// API Routes

// Health check
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', message: 'Reddit Persona Analyzer API is running' });
});

// Main analysis endpoint
app.post('/', async (req, res) => {
    try {
        const { username: userInput } = req.body;

        if (!userInput || !userInput.trim()) {
            return res.status(400).json({ error: 'Please enter a Reddit username or profile link.' });
        }

        const username = extractUsername(userInput);

        if (!username) {
            return res.status(400).json({ error: 'Could not extract username. Provide a Reddit username or profile URL.' });
        }

        console.log(`\n=== Processing username: ${username} ===`);

        // Step 1: Fetch Reddit data (in-memory)
        const { posts, comments, metadata } = await getRedditData(username);

        if (!posts.length && !comments.length) {
            return res.status(404).json({ error: `No data found for username '${username}'.` });
        }

        // Step 2: Create combined text and chunk it (in-memory, smaller chunks for token limit)
        const redditText = [
            '🔸 METADATA:',
            JSON.stringify(metadata, null, 2),
            '\n\n🔸 POSTS:\n',
            posts.map(p => p.text).join('\n\n'),
            '\n\n🔸 COMMENTS:\n',
            comments.map(c => c.text).join('\n\n'),
        ].join('\n');

        const chunks = chunkText(redditText, 300);

        if (chunks.length === 0) {
            return res.status(400).json({ error: 'No textual content to analyze for this user.' });
        }

        // Step 3: Embed & upsert to Pinecone (namespaced by username)
        const upserted = await embedAndUpsert(chunks, username);

        // Step 4: Search relevant chunks
        const query = 'Using the user\'s posts, comments and metadata, produce a concise persona summary and JSON.';
        const searchedChunks = await searchChunks(
            query,
            Math.min(10, chunks.length),
            username,
            chunks
        );

        console.log(`✓ Found ${searchedChunks.length} relevant chunks`);

        // Step 5: Generate persona via LLM
        const personaJson = await generatePersonaJson(searchedChunks);

        if (!personaJson) {
            return res.status(500).json({ error: 'Failed to generate persona from LLM. Try again later.' });
        }

        // Return response (no file storage)
        const response = {
            persona: personaJson,
            username,
            stats: {
                posts: posts.length,
                comments: comments.length,
                chunks: chunks.length,
                stored_vectors: upserted,
                relevant_chunks: searchedChunks.length,
            },
        };

        console.log(`✓ Analysis complete for ${username}\n`);

        return res.json(response);

    } catch (error) {
        console.error('Processing error:', error);
        return res.status(500).json({
            error: `Error processing username: ${error.message}`,
        });
    }
});

// Start server
async function startServer() {
    try {
        await initializeClients();

        app.listen(CONFIG.port, () => {
            console.log(`\n🚀 Server running on http://localhost:${CONFIG.port}`);
            console.log(`✓ All systems ready\n`);
        });
    } catch (error) {
        console.error('Failed to start server:', error);
        process.exit(1);
    }
}

startServer();
