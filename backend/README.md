# Reddit User Persona Analyzer - Node.js Backend

A Node.js backend service that analyzes Reddit user activity to generate detailed psychological personas using AI.

## Features

- **In-Memory Processing**: No file storage, all processing happens in memory
- **NVIDIA NIM API**: Uses NVIDIA's embeddings and LLM for analysis
- **Pinecone Vector DB**: Stores user activity embeddings for semantic search
- **Reddit API Integration**: Fetches user posts, comments, and metadata
- **RESTful API**: Simple JSON API for frontend integration

## Prerequisites

- Node.js 18+ (ES modules support)
- Reddit API credentials
- Pinecone API key
- NVIDIA NIM API key

## Installation

```bash
cd backend
npm install
```

## Environment Variables

Create a `.env` file in the backend directory:

```env
# Reddit API
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_SECRET=your_reddit_secret
REDDIT_USER_AGENT=user-persona-script/0.1 by YourName

# Pinecone API
PINECONE_API_KEY=your_pinecone_api_key

# NVIDIA NIM API
NVIDIA_API_KEY=your_nvidia_api_key

# Optional
PORT=5000
LLM_MODEL_NAME=nvidia/llama-3.3-nemotron-super-49b-v1
```

## Running the Server

### Development (with auto-reload)
```bash
npm run dev
```

### Production
```bash
npm start
```

The server will start on `http://localhost:5000`

## API Endpoints

### POST /
Analyze a Reddit user and generate persona

**Request Body:**
```json
{
  "username": "spez"
}
```

**Response:**
```json
{
  "persona": {
    "metadata": {...},
    "personality_traits": {...},
    "communication_style": "...",
    "interests": [...],
    ...
  },
  "username": "spez",
  "stats": {
    "posts": 100,
    "comments": 200,
    "chunks": 50,
    "stored_vectors": 50,
    "relevant_chunks": 10
  }
}
```

### GET /api/health
Health check endpoint

**Response:**
```json
{
  "status": "ok",
  "message": "Reddit Persona Analyzer API is running"
}
```

## Architecture

1. **Reddit Data Fetching**: Uses `snoowrap` to fetch user posts and comments
2. **Text Chunking**: Splits content into manageable chunks (~500 chars)
3. **Embedding Generation**: NVIDIA NIM API generates embeddings (384d vectors)
4. **Vector Storage**: Pinecone stores embeddings with user namespace
5. **Semantic Search**: Retrieves most relevant chunks for persona generation
6. **LLM Analysis**: NVIDIA LLM generates structured persona JSON

## Key Differences from Python Version

- ✅ **No File Storage**: Everything processed in-memory
- ✅ **ES Modules**: Modern JavaScript with import/export
- ✅ **NVIDIA NIM API**: Direct API calls for embeddings (instead of sentence-transformers)
- ✅ **Same Functionality**: Identical output format and behavior
- ✅ **Same UI/UX**: Frontend unchanged, drop-in replacement

## License

MIT
