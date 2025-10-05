# Reddit Persona Analyzer

A sophisticated AI-powered application that analyzes Reddit user profiles to generate detailed psychological personas. Built with Next.js, TypeScript, and advanced AI technologies.

## Features

- **AI-Powered Analysis**: Uses advanced language models and vector embeddings to analyze user behavior
- **Comprehensive Profiling**: Generates detailed personality traits, communication styles, and behavioral insights
- **Modern UI**: Beautiful, responsive interface with real-time progress tracking
- **Vector Search**: Utilizes Pinecone for intelligent content similarity matching
- **Privacy-Focused**: Analyzes only public Reddit data with ethical considerations

## Technology Stack

- **Frontend**: Next.js 15, React 19, TypeScript, Tailwind CSS
- **Backend**: Next.js API Routes, Node.js
- **AI/ML**: NVIDIA NIM API, Transformers.js, Pinecone Vector Database
- **Reddit API**: snoowrap (Reddit API wrapper)
- **UI Components**: shadcn/ui, Lucide React icons

## Getting Started

### Prerequisites

- Node.js 18+ 
- Reddit API credentials
- Pinecone API key
- NVIDIA API key

### Environment Variables

Create a `.env.local` file with:

\`\`\`env
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_SECRET=your_reddit_secret
REDDIT_USER_AGENT=your_app_name/1.0 by YourUsername
PINECONE_API_KEY=your_pinecone_api_key
NVIDIA_API_KEY=your_nvidia_api_key
\`\`\`

### Installation

\`\`\`bash
npm install
npm run dev
\`\`\`

Open [http://localhost:3000](http://localhost:3000) to view the application.

## How It Works

1. **Data Collection**: Fetches public Reddit posts and comments using the Reddit API
2. **Text Processing**: Chunks and processes text content for analysis
3. **Embedding Generation**: Creates vector embeddings using Transformers.js
4. **Vector Storage**: Stores embeddings in Pinecone for similarity search
5. **AI Analysis**: Uses NVIDIA's language models to generate psychological insights
6. **Results Display**: Presents comprehensive persona analysis in an intuitive interface

## API Endpoints

- `POST /api/analyze` - Analyze a Reddit user
- `GET /api/health` - Health check endpoint

## Privacy & Ethics

This application:
- Only analyzes publicly available Reddit data
- Does not store personal information
- Provides analysis for research and entertainment purposes
- Respects Reddit's API terms of service

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
