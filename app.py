import os
import json
import re
import logging
from flask import Flask, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv
import praw
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from uuid import uuid4

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For flash messages

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_SECRET = os.getenv("REDDIT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
INDEX_NAME = "reddit-user-vdb"
MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"
DIMENSION = 384

# Validate environment variables
required_env_vars = {
    "REDDIT_CLIENT_ID": REDDIT_CLIENT_ID,
    "REDDIT_SECRET": REDDIT_SECRET,
    "REDDIT_USER_AGENT": REDDIT_USER_AGENT,
    "PINECONE_API_KEY": PINECONE_API_KEY,
    "NVIDIA_API_KEY": NVIDIA_API_KEY
}
for var_name, var_value in required_env_vars.items():
    if not var_value:
        raise ValueError(f"Environment variable {var_name} is not set in .env file")

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in pc.list_indexes().names():
        logger.info(f"Creating Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    pinecone_index = pc.Index(INDEX_NAME)
    logger.info(f"Pinecone index '{INDEX_NAME}' initialized successfully")
except Exception as e:
    logger.error(f"Pinecone initialization failed: {e}")
    raise

# Initialize SentenceTransformer
try:
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("SentenceTransformer initialized successfully")
except Exception as e:
    logger.error(f"SentenceTransformer initialization failed: {e}")
    raise

# Initialize NVIDIA LLM Client
try:
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY
    )
    logger.info("NVIDIA client initialized successfully")
except Exception as e:
    logger.error(f"NVIDIA client initialization failed: {e}")
    raise

# Extract username from Reddit URL or direct username
def extract_username(input_str):
    logger.debug(f"Extracting username from input: {input_str}")
    if 'reddit.com' in input_str:
        match = re.search(r'reddit\.com/u(?:ser)?/([^\s/]+)', input_str)
        username = match.group(1) if match else input_str.strip()
    else:
        username = input_str.strip()
    logger.debug(f"Extracted username: {username}")
    return username

# Get Reddit Data
def get_reddit_data(username):
    logger.debug(f"Fetching Reddit data for user: {username}")
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    try:
        user = reddit.redditor(username)
        # Check if user exists
        user.name
        metadata = {
            "username": user.name,
            "created_utc": str(datetime.fromtimestamp(user.created_utc, tz=timezone.utc).date()),
            "link_karma": user.link_karma,
            "comment_karma": user.comment_karma,
            "total_karma": user.link_karma + user.comment_karma,
            "verified_email": getattr(user, "has_verified_email", False),
            "is_gold": getattr(user, "is_gold", False),
            "is_mod": getattr(user, "is_mod", False),
            "has_subreddit": hasattr(user, "subreddit")
        }
        if hasattr(user, "subreddit"):
            sub = user.subreddit
            metadata.update({
                "profile_title": sub.title,
                "public_description": sub.public_description,
                "subscribers": getattr(sub, "subscribers", None),
                "user_flair_text": getattr(sub, "user_flair_text", None)
            })
        logger.debug(f"Metadata fetched: {metadata}")
    except Exception as e:
        logger.error(f"Metadata error for {username}: {e}")
        metadata = {}

    posts = []
    try:
        for post in user.submissions.new(limit=100):
            posts.append(f"[POST] {post.title}\n{post.selftext or ''}")
        logger.debug(f"Fetched {len(posts)} posts")
    except Exception as e:
        logger.error(f"Error fetching posts: {e}")

    comments = []
    try:
        for comment in user.comments.new(limit=200):
            comments.append(f"[COMMENT] {comment.body}")
        logger.debug(f"Fetched {len(comments)} comments")
    except Exception as e:
        logger.error(f"Error fetching comments: {e}")

    return posts, comments, metadata

# Save raw Reddit data to file
def save_raw_data(username, posts, comments, metadata):
    logger.debug(f"Saving raw data for {username}")
    os.makedirs('debug', exist_ok=True)
    raw_file = f"debug/{username}_raw.txt"
    try:
        with open(raw_file, "w", encoding="utf-8") as f:
            f.write("🔸 METADATA:\n")
            f.write(json.dumps(metadata, indent=2))
            f.write("\n\n🔸 POSTS:\n\n")
            f.write("\n\n".join(posts))
            f.write("\n\n🔸 COMMENTS:\n\n")
            f.write("\n\n".join(comments))
        logger.info(f"Raw data saved to {raw_file}")
    except Exception as e:
        logger.error(f"Error saving raw data: {e}")
        raise

# Chunk text
def chunk_text(text, max_len=500):
    logger.debug("Chunking text")
    paragraphs = text.strip().split("\n\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) < max_len:
            current += para + "\n\n"
        else:
            chunks.append(current.strip())
            current = para + "\n\n"
    if current:
        chunks.append(current.strip())
    logger.debug(f"Created {len(chunks)} chunks")
    return chunks

# Embed and upsert to Pinecone
def embed_and_upsert(chunks):
    logger.debug("Embedding and upserting to Pinecone")
    try:
        vectors = [
            (str(uuid4()), embed_model.encode(chunk).tolist(), {"text": chunk})
            for chunk in chunks
        ]
        pinecone_index.upsert(vectors=vectors)
        logger.info(f"Upserted {len(vectors)} vectors to Pinecone")
    except Exception as e:
        logger.error(f"Pinecone upsert failed: {e}")
        raise

# Search chunks in Pinecone
def search_chunks(query, top_k=10, raw_chunks=None):
    logger.debug(f"Searching Pinecone with query: {query[:50]}...")
    try:
        vector = embed_model.encode(query).tolist()
        results = pinecone_index.query(vector=vector, top_k=top_k, include_metadata=True)
        chunks = [match['metadata']['text'] for match in results['matches']]
        logger.debug(f"Retrieved {len(chunks)} chunks from Pinecone")
        if not chunks and raw_chunks:
            logger.warning("No chunks retrieved from Pinecone, falling back to raw chunks")
            return raw_chunks[:top_k]  # Fallback to raw chunks if none retrieved
        return chunks
    except Exception as e:
        logger.error(f"Pinecone search failed: {e}")
        if raw_chunks:
            logger.warning("Falling back to raw chunks due to Pinecone search failure")
            return raw_chunks[:top_k]
        raise

# Robust JSON parsing
def try_parse_json_fallback(text):
    logger.debug("Attempting to parse JSON response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            json_block = re.search(r'\{[\s\S]+\}', text).group(0)
            json_block = re.sub(r",\s*}", "}", json_block)
            json_block = re.sub(r",\s*]", "]", json_block)
            parsed = json.loads(json_block)
            logger.debug("Successfully parsed JSON with fallback")
            return parsed
        except Exception as e:
            logger.error(f"JSON parse error: {e}")
            return None

# Generate Persona JSON
def generate_persona_json(chunks):
    logger.debug("Generating persona JSON")
    if not chunks:
        logger.error("No chunks provided for persona generation")
        return None
    context = "\n\n".join(chunks)[:3000]
    logger.debug(f"Context length: {len(context)} characters")
    prompt = f"""
You are an AI tasked with building a psychological profile/persona of a Reddit user from their comment and post history, along with basic Reddit metadata.

Return ONLY valid JSON. No Markdown, comments, or extra text. The JSON must begin with {{ and end with }}.

Reddit History: \""" {context} \"""

JSON format:
{{
  "metadata": {{
    "username": "string or null",
    "created_utc": "string (UTC timestamp) or null",
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
  }},
  "personality_traits": {{
    "introvert": true/false,
    "introvert_reason": "brief reason based on evidence from the text",
    "extrovert": true/false,
    "extrovert_reason": "brief reason",
    "anger_level": 0-5,
    "anger_level_reason": "reason or example post indicating anger expression or suppression",
    "empathy_level": 0-5,
    "empathy_level_reason": "reason or examples indicating empathy or lack of it",
    "judgmental": true/false,
    "judgmental_reason": "evidence of strong opinions or open-mindedness",
    "analytical": true/false,
    "analytical_reason": "indication of logical thinking, breaking down problems, etc.",
    "humor_style": "dry/sarcastic/wholesome/dark/observational/etc" or null,
    "humor_style_reason": "examples of humor style used",
    "confidence_level": 0-5,
    "confidence_level_reason": "level of certainty/assertiveness in language"
  }},
  "communication_style": "casual/formal/assertive/passive/etc",
  "communication_style_reason": "justification from writing tone and structure",
  "interests": ["list of interests if detectable"],
  "likely_profession": "null or best guess",
  "likely_profession_reason": "why this profession fits (based on vocabulary, knowledge, topics)",
  "location_mentioned": "null or city/state if mentioned",
  "writing_style": "short/long/stream-of-consciousness/structured/etc",
  "writing_style_reason": "supporting explanation based on structure and coherence",
  "behaviour_and_habits": [
    "bullet point of one observable or inferred habit or behavior",
    "another habit or routine pattern",
    "up to 5 such points"
  ],
  "goals_and_needs": [
    "deduced goals or needs (e.g. self-improvement, community belonging, recognition)",
    "up to 5"
  ],
  "frustrations": [
    "challenges or irritants mentioned explicitly or indirectly",
    "up to 5"
  ]
}}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "ONLY return valid JSON. No Markdown, comments, or extra text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            top_p=0.9,
            max_tokens=2048,
            stream=False
        )
        output = response.choices[0].message.content.strip()
        parsed = try_parse_json_fallback(output)
        if parsed:
            logger.info("Persona JSON generated successfully")
            logger.debug(f"Persona JSON content: {json.dumps(parsed, indent=2)[:500]}...")  # Log first 500 chars
        else:
            logger.error("Failed to parse JSON response")
        return parsed
    except Exception as e:
        logger.error(f"Error generating persona JSON: {e}")
        return None

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form.get('username')
        if not user_input:
            flash("Please enter a Reddit username or profile link.")
            logger.warning("No username provided in form")
            return redirect(url_for('index'))

        username = extract_username(user_input)
        logger.info(f"Processing username: {username}")
        try:
            # Step 1: Clear Pinecone index
            logger.debug(f"Clearing Pinecone index '{INDEX_NAME}' for user {username}")
            pinecone_index.delete(delete_all=True)
            logger.info(f"Pinecone index '{INDEX_NAME}' cleared successfully")

            # Step 2: Get Reddit data and save raw file
            posts, comments, metadata = get_reddit_data(username)
            if not metadata and not posts and not comments:
                flash(f"No data found for username '{username}'. Please check the username and try again.")
                logger.warning(f"No data found for username: {username}")
                return redirect(url_for('index'))
            save_raw_data(username, posts, comments, metadata)

            # Step 3: Load raw file and chunk it
            raw_file = f"debug/{username}_raw.txt"
            logger.debug(f"Loading raw file: {raw_file}")
            with open(raw_file, "r", encoding="utf-8") as f:
                reddit_text = f.read()
            chunks = chunk_text(reddit_text)

            # Step 4: Embed and upsert to Pinecone
            embed_and_upsert(chunks)

            # Step 5: Search and generate persona
            query = (
                "Generate a detailed psychological profile of a Reddit user using post/comment history and metadata. "
                "Include the following: metadata (username, created_utc, link_karma, comment_karma, total_karma, verified_email, "
                "is_gold, is_mod, has_subreddit, profile_title, public_description, subscribers, user_flair_text), "
                "personality_traits with reasons, communication_style with reason, tone, interests, likely_profession with reason, "
                "writing_style with reason, behavior_and_habits, goals_and_needs, and frustrations."
            )
            chunks = search_chunks(query, top_k=min(10, len(chunks)), raw_chunks=chunks)
            persona_json = generate_persona_json(chunks)

            if persona_json:
                # Save persona JSON
                json_file = f"persona_output_{username}.json"
                try:
                    with open(json_file, "w", encoding="utf-8") as f:
                        json.dump(persona_json, f, indent=2, ensure_ascii=False)
                    logger.info(f"Persona JSON saved to {json_file}")
                except Exception as e:
                    logger.error(f"Error saving persona JSON: {e}")
                    flash(f"Error saving persona JSON: {str(e)}")
                    return redirect(url_for('index'))

                # Render persona.html with the generated JSON
                logger.info(f"Rendering persona.html for {username}")
                return render_template('persona.html', persona=persona_json)
            else:
                flash("Failed to generate persona. Please check the username or try again later.")
                logger.error("Persona JSON is None or invalid")
                return redirect(url_for('index'))

        except Exception as e:
            flash(f"Error processing username '{username}': {str(e)}")
            logger.error(f"Processing error for {username}: {e}")
            return redirect(url_for('index'))

    logger.debug("Rendering index.html")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)