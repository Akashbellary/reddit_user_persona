# Reddit Persona Generator

This project is a Flask-based web application that builds psychological profiles of Reddit users based on their posts, comments, and metadata. It fetches user data from Reddit, processes it using text chunking and embeddings, stores embeddings in Pinecone, generates a persona using NVIDIA's Llama model, and displays the results in a clean web interface.

## Features
- **Reddit Data Fetching**: Pulls up to 100 posts and 200 comments for a given Reddit username using the PRAW library.
- **Text Processing**: Chunks user data into manageable pieces for analysis.
- **Vector Storage**: Embeds text using SentenceTransformers and stores vectors in a Pinecone index for similarity search.
- **Persona Generation**: Creates a detailed JSON persona (metadata, personality traits, communication style, interests, etc.) using NVIDIA's Llama model via their API.
- **Web Interface**: A simple Flask app with a form to input a Reddit username and a results page to display the persona.
- **Debugging**: Saves raw Reddit data to `debug/{username}_raw.txt` and persona output to `persona_output_{username}.json` for inspection.

## Why API Keys Are Included
The `.env` file includes API keys for Reddit, Pinecone, and NVIDIA to make it easy for recruiters or evaluators to run the project without needing to generate their own keys. This ensures a smooth setup and testing experience, as the keys are already configured for immediate use. In a production environment, these keys should be secured and not committed to version control.

## Prerequisites
- Python 3.8 or higher
- A Reddit account (for API credentials, though pre-configured in `.env`)
- A Pinecone account (pre-configured in `.env`)
- An NVIDIA API key (pre-configured in `.env`)

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd grok_final
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install flask python-dotenv praw sentence-transformers pinecone-client openai requests
   ```

4. **Verify `.env` File**:
   The `.env` file is included with pre-configured API keys for convenience:
   ```
   REDDIT_CLIENT_ID=<API KEY>
   REDDIT_SECRET=<API KEY>
   REDDIT_USER_AGENT=user-persona-script/0.1
   PINECONE_API_KEY=<API KEY>
   NVIDIA_API_KEY=your_nvidia_api_key
   ```
   The `NVIDIA_API_KEY` is a placeholder. If it doesn't work, replace it with a valid key from NVIDIA's API dashboard or contact me for an updated key.

5. **Directory Structure**:
   Ensure the following structure:
   ```
   reddit_user_persona/
   ├── app.py
   ├── .env
   ├── templates/
   │   ├── index.html
   │   ├── persona.html
   ├── static/
   │   ├── (optional CSS/JS files)
   ├── debug/
   │   ├── (e.g., Hungry-Move-6603_raw.txt)
   ├── persona_output_{username}.json
   ├── requirements.txt
   ```

6. **Run the Application**:
   ```bash
   python app.py
   ```
   The app will start at `http://127.0.0.1:5000`.

## Usage
1. Open `http://127.0.0.1:5000` in your browser.
2. Enter a Reddit username (e.g., `u/Hungry-Move-6603` or `https://www.reddit.com/user/kojied`) in the form.
3. Submit to fetch the user's data, generate a persona, and view the results on the `persona.html` page.
4. Check the `debug/` folder for raw data and the project root for `persona_output_{username}.json`.

## How It Works
1. **Input**: The user submits a Reddit username or profile link via `index.html`.
2. **Reddit Data**: The app uses PRAW to fetch user metadata, posts, and comments, saving them to `debug/{username}_raw.txt`.
3. **Text Processing**: The raw text is chunked into smaller pieces (max 500 characters).
4. **Pinecone**: The chunks are embedded using SentenceTransformers (`all-MiniLM-L6-v2`) and stored in a Pinecone index (`reddit-user-vdb`). The index is cleared before each new user to avoid data mixing.
5. **Persona Generation**: A similarity search retrieves relevant chunks, which are sent to NVIDIA's Llama model to generate a JSON persona.
6. **Output**: The persona is saved to `persona_output_{username}.json` and rendered on `persona.html`.

## Notes
- **Pinecone Index Clearing**: The app clears the `reddit-user-vdb` index for each new user to prevent data from previous users affecting results.
- **Debugging**: Check logs in the console for detailed steps (e.g., data fetching, embedding, JSON generation). Raw data and JSON outputs are saved for inspection.
- **Performance**: For users with little data (e.g., few posts/comments), the app falls back to raw chunks if Pinecone search returns no results.
- **Styling**: The `persona.html` template uses Tailwind CSS (via CDN) for a clean, responsive layout.

## Troubleshooting
- **Pinecone Issues**: If Pinecone fails (e.g., upsert or search errors), verify the `PINECONE_API_KEY` in `.env` and check the Pinecone dashboard for the `reddit-user-vdb` index.
- **NVIDIA API**: If persona generation fails, ensure the `NVIDIA_API_KEY` is valid. Test it with:
  ```python
  from openai import OpenAI
  client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key="your_nvidia_api_key")
  response = client.chat.completions.create(model="nvidia/llama-3.3-nemotron-super-49b-v1", messages=[{"role": "user", "content": "Test"}])
  print(response.choices[0].message.content)
  ```
- **Reddit API**: If data fetching fails, ensure the Reddit API keys are valid and the user exists.
- **Empty Results**: If `persona.html` shows incomplete data, check `persona_output_{username}.json` and `debug/{username}_raw.txt` to verify the data fetched and processed.
- **Logs**: Console logs provide detailed debugging info (e.g., number of chunks, Pinecone search results, JSON content).

## Example
- Input: `u/kojied`
- Output: `debug/kojied_raw.txt` (raw Reddit data), `persona_output_kojied.json` (persona JSON), and a rendered `persona.html` with metadata, personality traits, interests, etc.
- Try users with active profiles for best results (e.g., `u/Hungry-Move-6603` or `u/kojied`).

## Limitations
- The app processes only the most recent 100 posts and 200 comments due to Reddit API limits.
- Small datasets (e.g., users with few posts) may lead to less detailed personas, mitigated by the raw chunk fallback.
- The NVIDIA API key may need updating if it expires or reaches usage limits.
- The app is single-user focused; each request clears the Pinecone index to avoid data mixing.

## Future Improvements
- Add user-specific namespaces in Pinecone to support multiple users without clearing the index.
- Cache Reddit data to reduce API calls for repeated users.
- Enhance `persona.html` with interactive visualizations of personality traits.
- Support batch processing for multiple users.

## Author
Akash  
Built as a demonstration of integrating Reddit data, vector databases, and AI-driven persona generation for a web application.
