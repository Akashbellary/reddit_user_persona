# 🧠 Reddit Persona Analyzer

A sophisticated AI-powered application that analyzes Reddit user profiles to generate detailed psychological personas. Built with Next.js, TypeScript, Node.js backend, and advanced AI technologies.

## 🌟 Features

- **AI-Powered Analysis**: Uses NVIDIA NIM LLM and embeddings to analyze user behavior
- **Comprehensive Profiling**: Generates detailed personality traits, communication styles, and behavioral insights
- **Modern UI**: Beautiful, responsive interface with real-time progress tracking
- **Vector Search**: Utilizes Pinecone for intelligent content similarity matching
- **In-Memory Processing**: No file storage, all processing happens in memory for deployment efficiency
- **Privacy-Focused**: Analyzes only public Reddit data with ethical considerations

## 🛠️ Technology Stack

- **Frontend**: Next.js 15, React 19, TypeScript, Tailwind CSS
- **Backend**: Node.js (Express), ES Modules
- **AI/ML**: NVIDIA NIM API (Embeddings + LLM), Pinecone Vector Database
- **Reddit API**: snoowrap (Reddit API wrapper)
- **UI Components**: shadcn/ui, Lucide React icons

## 🚀 Getting Started

---

## 🏗️ Architecture

### System Overview
```
Next.js Frontend ──► Flask Backend ──► External APIs
                                    ├── Reddit API (PRAW)
                                    ├── NVIDIA NIM
                                    └── Pinecone Vector DB
```

**Frontend:** Next.js 15, React 19, TypeScript, Tailwind CSS, shadcn/ui  
**Backend:** Python Flask, PRAW, SentenceTransformers, Pinecone, OpenAI client

- Node.js 18+ 
- Reddit API credentials ([Get them here](https://www.reddit.com/prefs/apps))
- Pinecone API key ([Get it here](https://www.pinecone.io/))
- NVIDIA NIM API key ([Get it here](https://build.nvidia.com/))

### 1. Clone & Install
```bash
git clone https://github.com/Akashbellary/reddit_user_persona.git
cd reddit_user_persona

Create a `.env` file in the **backend** directory:

\`\`\`env
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
\`\`\`

For the **frontend**, create `.env.local` in the root directory:

\`\`\`env
FLASK_BACKEND_URL=http://localhost:5000
\`\`\`

### Installation & Running

#### 1. Install Frontend Dependencies
\`\`\`bash
npm install
\`\`\`

#### 2. Install Backend Dependencies
\`\`\`bash
cd backend
npm install
\`\`\`

#### 3. Start Backend Server
\`\`\`bash
cd backend
npm start
# or for development with auto-reload
npm run dev
\`\`\`

#### 4. Start Frontend (in a new terminal)
\`\`\`bash
npm run dev
\`\`\`

---

## 🔄 How It Works

1. **Data Collection**: Fetches public Reddit posts and comments using the Reddit API
2. **Text Processing**: Chunks and processes text content for analysis (in-memory)
3. **Embedding Generation**: Creates vector embeddings using NVIDIA NIM API (384d vectors)
4. **Vector Storage**: Stores embeddings in Pinecone with user-specific namespaces
5. **Semantic Search**: Retrieves most relevant chunks for persona generation
6. **AI Analysis**: Uses NVIDIA's LLM to generate structured psychological insights
7. **Results Display**: Presents comprehensive persona analysis in an intuitive interface

## 📡 API Endpoints

### Backend (Port 5000)
- `POST /` - Analyze a Reddit user
  - Request: `{ "username": "spez" }`
  - Response: `{ "persona": {...}, "username": "...", "stats": {...} }`
- `GET /api/health` - Health check endpoint

### Frontend (Port 3000)
- `POST /api/analyze` - Proxy to backend analysis endpoint
- `GET /api/health` - Frontend health check

## 🔒 Privacy & Ethics

This application:
- ✅ Only analyzes publicly available Reddit data
- ✅ Does not store personal information (in-memory processing)
- ✅ No file writes or persistent storage of user data
- ✅ Provides analysis for research and entertainment purposes
- ✅ Respects Reddit's API terms of service
- ✅ Uses namespaced vector storage for data isolation

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Built by [Akashbellary](https://github.com/Akashbellary) for AI-powered social media analysis**
