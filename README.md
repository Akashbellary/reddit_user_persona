# 🧠 Reddit Persona Analyzer

[Watch Demo Video](https://github.com/Akashbellary/reddit_user_persona/blob/main/VID_20251006_012138.mp4)


A **sophisticated AI-powered application** that analyzes Reddit user profiles to generate detailed psychological personas using advanced machine learning, vector embeddings, and large language models. Built with **Next.js 15**, **Python Flask**, and cutting-edge AI technologies.

---

## 🚀 Features

- **AI-Powered Analysis** – Leverages NVIDIA NIM API with Llama 3.3 Nemotron for psychological analysis
- **Vector Embeddings** – Uses SentenceTransformers for semantic content representation  
- **Comprehensive Profiling** – Generates personality traits, communication styles, and behavioral insights
- **Modern UI** – Responsive interface with real-time progress tracking and dark mode
- **Privacy-Focused** – Analyzes only public Reddit data with no persistent storage

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

---

## ⚙️ Installation & Setup

### Prerequisites
- Node.js 18+
- Python 3.9+
- Reddit API credentials
- Pinecone API key
- NVIDIA API key

### 1. Clone & Install
```bash
git clone https://github.com/Akashbellary/reddit_user_persona.git
cd reddit_user_persona

# Frontend
npm install

# Backend
cd backend
pip install -r requirements.txt
pip install praw python-dotenv pinecone-client openai
```

### 2. Environment Setup
Create `.env.local` in root directory:
```env
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=RedditPersonaAnalyzer/1.0 by YourUsername
PINECONE_API_KEY=your_pinecone_api_key
NVIDIA_API_KEY=your_nvidia_api_key
FLASK_BACKEND_URL=http://localhost:5000
```

### 3. API Configuration

**Reddit API Setup:**
1. Visit [Reddit App Preferences](https://www.reddit.com/prefs/apps)
2. Create new app (type: "script")
3. Copy client ID and secret

**Pinecone Setup:**
1. Create account at [pinecone.io](https://www.pinecone.io)
2. Create index: `reddit-user-vdb`, dimensions: `384`, metric: `cosine`

---

## 🚦 Running the Application

**Start Backend:**
```bash
cd backend
python app.py
# Runs on http://localhost:5000
```

**Start Frontend:**
```bash
npm run dev  
# Runs on http://localhost:3000
```

---

## 🧩 How It Works

1. **Data Collection** – Fetches Reddit posts/comments via PRAW API
2. **Text Processing** – Chunks content into ~500 character segments
3. **Vector Embedding** – Generates 384-dimensional embeddings using SentenceTransformers
4. **Vector Storage** – Stores embeddings in Pinecone with user namespaces
5. **AI Analysis** – Uses NVIDIA's Llama model to generate structured persona JSON
6. **Results Display** – Renders comprehensive personality profile in React UI

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze` | Analyze Reddit user |
| `GET` | `/api/health` | Health check |

**Request Format:**
```json
{"username": "reddit_username"}
```

**Response Format:**
```typescript
{
  persona: PersonaAnalysis,
  username: string,
  stats: {
    posts: number,
    comments: number,
    chunks: number,
    stored_vectors: number,
    relevant_chunks: number
  }
}
```

---

## 🔒 Privacy & Ethics

- **Public Data Only** – Analyzes exclusively public Reddit posts and comments
- **No Data Storage** – User information is not permanently stored
- **Research Purpose** – Intended for educational and entertainment use
- **Ethical AI** – Results should not be used for harassment or discrimination

---

## 🐛 Troubleshooting

**Common Issues:**
- `Reddit user not found` → Verify username exists and profile is public
- `Pinecone connection failed` → Check API key and index configuration  
- `Rate limit exceeded` → Wait before retrying or upgrade API tier
- `Backend connection failed` → Ensure Flask server is running on port 5000

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Next.js 15, React 19, TypeScript, Tailwind CSS |
| Backend | Python Flask, PRAW, SentenceTransformers |
| AI/ML | NVIDIA NIM API, Pinecone Vector Database |
| UI | shadcn/ui, Lucide React Icons |

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Built by [Akashbellary](https://github.com/Akashbellary) for AI-powered social media analysis**
