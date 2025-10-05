# app.py
import os
import re
import json
import logging
from uuid import uuid4
from datetime import datetime, timezone

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from dotenv import load_dotenv
import praw
from sentence_transformers import SentenceTransformer

# Pinecone (new class-based client) and serverless spec
from pinecone import Pinecone, ServerlessSpec

# NVIDIA "OpenAI-compatible" wrapper used previously in your repo
from openai import OpenAI

# -------------------------
# Basic setup
# -------------------------
load_dotenv()  # load .env

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.urandom(24)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------
# Config (from your .env)
# -------------------------
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_SECRET = os.getenv("REDDIT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# fixed values from your environment/info
INDEX_NAME = "reddit-user-vdb"
DIMENSION = 384
PINECONE_REGION = "us-east-1"
PINECONE_CLOUD = "aws"

# Validate
required = {
    "REDDIT_CLIENT_ID": REDDIT_CLIENT_ID,
    "REDDIT_SECRET": REDDIT_SECRET,
    "REDDIT_USER_AGENT": REDDIT_USER_AGENT,
    "PINECONE_API_KEY": PINECONE_API_KEY,
    "NVIDIA_API_KEY": NVIDIA_API_KEY,
}
missing = [k for k, v in required.items() if not v]
if missing:
    raise EnvironmentError(f"Missing required env vars: {', '.join(missing)}")

# -------------------------
# Initialize Pinecone (class-based API)
# -------------------------
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    # pc.list_indexes() returns an object with .names() in your earlier code; handle both shapes
    try:
        existing_names = pc.list_indexes().names()
    except Exception:
        # fallback: list of strings
        existing_names = pc.list_indexes() or []

    if INDEX_NAME not in existing_names:
        logger.info("Index '%s' not found; creating it.", INDEX_NAME)
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    pinecone_index = pc.Index(INDEX_NAME)
    logger.info("Pinecone index initialized: %s (dim=%d)", INDEX_NAME, DIMENSION)
except Exception as e:
    logger.exception("Failed to initialize Pinecone client")
    raise

# -------------------------
# Initialize Reddit (PRAW)
# -------------------------
try:
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        check_for_async=False,  # avoid async warnings in some environments
    )
    logger.info("PRAW Reddit client initialized")
except Exception:
    logger.exception("Failed to initialize PRAW")
    raise

# -------------------------
# Embedding model
# -------------------------
try:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384d
    logger.info("SentenceTransformer loaded (all-MiniLM-L6-v2)")
except Exception:
    logger.exception("Failed to load SentenceTransformer")
    raise

# -------------------------
# LLM client (NVIDIA via OpenAI-like wrapper)
# -------------------------
try:
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
    logger.info("NVIDIA/OpenAI-compatible client initialized")
except Exception:
    logger.exception("Failed to initialize LLM client")
    raise

# -------------------------
# Utilities
# -------------------------
def extract_username(input_str: str) -> str | None:
    """
    Robustly extract reddit username from:
      - 'spez' or '/u/spez' or 'u/spez'
      - 'https://www.reddit.com/user/spez/'
      - 'https://reddit.com/u/spez'
    Returns None if nothing valid found.
    """
    if not input_str:
        return None
    s = input_str.strip()
    # if it looks like JSON/form payload or accidental newline, normalize
    s = s.split()[0]

    # Try URL patterns first
    m = re.search(r"(?:reddit\.com\/(?:user|u)\/)([A-Za-z0-9_-]+)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    # Try leading u/ or /u/ or user/ or /user/
    s = re.sub(r"^\/?(?:u|user)\/", "", s, flags=re.IGNORECASE)
    s = s.strip("/")

    # finally, simple username validation (reddit usernames: 3-20 chars typically; allow short too)
    if re.match(r"^[A-Za-z0-9_-]{1,50}$", s):
        return s

    return None


def get_reddit_data(username: str, posts_limit: int = 100, comments_limit: int = 200):
    """
    Returns (posts:list[str], comments:list[str], metadata:dict)
    Posts and comments are formatted strings ready for chunking/embedding.
    """
    logger.debug("Fetching reddit data for username=%s", username)
    try:
        user = reddit.redditor(username)
        # Access a property to validate existence
        _ = user.name
    except Exception as e:
        logger.error("Error accessing redditor '%s': %s", username, e)
        raise Exception(f"Reddit user '{username}' not found or inaccessible: {e}")

    # metadata
    try:
        metadata = {
            "username": user.name,
            "created_utc": None,
            "link_karma": getattr(user, "link_karma", None),
            "comment_karma": getattr(user, "comment_karma", None),
            "total_karma": None,
            "verified_email": getattr(user, "has_verified_email", None),
            "is_gold": getattr(user, "is_gold", None),
            "is_mod": getattr(user, "is_mod", None),
            "has_subreddit": bool(getattr(user, "subreddit", None)),
        }
        if getattr(user, "created_utc", None):
            metadata["created_utc"] = datetime.fromtimestamp(user.created_utc, tz=timezone.utc).isoformat()
        if metadata["link_karma"] is not None and metadata["comment_karma"] is not None:
            metadata["total_karma"] = metadata["link_karma"] + metadata["comment_karma"]

        if getattr(user, "subreddit", None):
            sub = user.subreddit
            metadata.update({
                "profile_title": getattr(sub, "title", None),
                "public_description": getattr(sub, "public_description", None),
                "subscribers": getattr(sub, "subscribers", None),
                "user_flair_text": getattr(sub, "user_flair_text", None)
            })
    except Exception:
        logger.exception("Error building metadata for %s", username)
        metadata = {}

    # posts
    posts = []
    try:
        for p in user.submissions.new(limit=posts_limit):
            title = p.title or ""
            selftext = p.selftext or ""
            posts.append({
                "id": getattr(p, "id", None),
                "subreddit": str(getattr(p, "subreddit", None)),
                "created_utc": datetime.fromtimestamp(getattr(p, "created_utc", 0), tz=timezone.utc).isoformat() if getattr(p, "created_utc", None) else None,
                "score": getattr(p, "score", None),
                "num_comments": getattr(p, "num_comments", None),
                "text": f"[POST] {title}\n\n{selftext}".strip()
            })
        logger.info("Fetched %d posts for %s", len(posts), username)
    except Exception:
        logger.exception("Error fetching posts for %s", username)

    # comments
    comments = []
    try:
        for c in user.comments.new(limit=comments_limit):
            body = c.body or ""
            comments.append({
                "id": getattr(c, "id", None),
                "subreddit": str(getattr(c, "subreddit", None)),
                "created_utc": datetime.fromtimestamp(getattr(c, "created_utc", 0), tz=timezone.utc).isoformat() if getattr(c, "created_utc", None) else None,
                "score": getattr(c, "score", None),
                "text": f"[COMMENT] {body}".strip()
            })
        logger.info("Fetched %d comments for %s", len(comments), username)
    except Exception:
        logger.exception("Error fetching comments for %s", username)

    return posts, comments, metadata


def save_raw_data(username: str, posts: list, comments: list, metadata: dict):
    os.makedirs("debug", exist_ok=True)
    raw_file = os.path.join("debug", f"{username}_raw.txt")
    try:
        with open(raw_file, "w", encoding="utf-8") as fh:
            fh.write("🔸 METADATA:\n")
            fh.write(json.dumps(metadata, indent=2, ensure_ascii=False))
            fh.write("\n\n🔸 POSTS:\n\n")
            fh.write("\n\n".join([p["text"] for p in posts]))
            fh.write("\n\n🔸 COMMENTS:\n\n")
            fh.write("\n\n".join([c["text"] for c in comments]))
        logger.info("Raw data saved to %s", raw_file)
    except Exception:
        logger.exception("Failed to save raw data")
        raise
    return raw_file


def chunk_text(text: str, max_len: int = 500):
    """
    Split by paragraphs into chunks ~max_len characters.
    """
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_len:
            current += para + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            current = para + "\n\n"
    if current:
        chunks.append(current.strip())
    logger.debug("Created %d chunks", len(chunks))
    return chunks


def embed_and_upsert(chunks: list, username: str):
    """
    Embed each chunk and upsert into Pinecone.
    We namespace using the username so each user's vectors are isolated.
    """
    if not chunks:
        return 0
    try:
        vectors = []
        for chunk in chunks:
            vec = embed_model.encode(chunk).tolist()
            vectors.append((str(uuid4()), vec, {"text": chunk}))
        # prefer namespace to avoid deleting whole index; safe upsert signature flexible across SDK versions
        try:
            pinecone_index.upsert(vectors=vectors, namespace=username)
        except TypeError:
            # fallback if client expects different keyword
            pinecone_index.upsert(vectors=vectors)
        logger.info("Upserted %d vectors for user '%s'", len(vectors), username)
        return len(vectors)
    except Exception:
        logger.exception("Failed to embed/upsert for %s", username)
        raise


def search_chunks(query: str, top_k: int = 10, username_namespace: str | None = None, raw_chunks: list | None = None):
    """
    Search Pinecone using an embedded vector for the query.
    If namespace provided, search that namespace only.
    """
    try:
        qvec = embed_model.encode(query).tolist()
        kwargs = {"vector": qvec, "top_k": top_k, "include_metadata": True}
        if username_namespace:
            kwargs["namespace"] = username_namespace
        res = pinecone_index.query(**kwargs)
        # handle dict-style or object-style responses
        matches = []
        if isinstance(res, dict):
            matches = res.get("matches", [])
        else:
            matches = getattr(res, "matches", []) or []
        chunks = []
        for m in matches:
            if isinstance(m, dict):
                md = m.get("metadata") or {}
            else:
                md = getattr(m, "metadata", {}) or {}
            text = md.get("text") if isinstance(md, dict) else None
            chunks.append(text or str(md))
        if not chunks and raw_chunks:
            logger.warning("No chunks returned from Pinecone; falling back to raw chunks")
            return raw_chunks[:top_k]
        return chunks
    except Exception:
        logger.exception("Pinecone search failed")
        if raw_chunks:
            return raw_chunks[:top_k]
        raise


def try_parse_json_fallback(text: str):
    """
    Try to parse JSON from text. If direct parse fails extract {...} block and try again.
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # extract first JSON-like block
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        block = m.group(0)
        block = re.sub(r",\s*}", "}", block)
        block = re.sub(r",\s*]", "]", block)
        try:
            return json.loads(block)
        except Exception:
            logger.exception("Fallback JSON parse failed")
            return None


def generate_persona_json(chunks: list):
    """
    Use LLM to create persona JSON from chunks.
    """
    if not chunks:
        return None
    context = "\n\n".join(chunks)[:3000]
    prompt = f"""
You are an AI tasked with building a **detailed and structured psychological profile/persona** of a Reddit user 
based on their posts, comments, and metadata. 

Your goal is to return **ONLY valid JSON**, with the exact same keys, types, and structure every time, 
so it can be directly used in a dashboard. Do NOT include Markdown, explanations, or extra text outside JSON.

**Reddit History Context:**
\"\"\"
{context}
\"\"\"

**Output JSON structure:**

{{
  "metadata": {{
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
  }},
  "personality_traits": {{
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
  }},
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
}}

**Instructions:**
1. Always include all keys. Use `null`, empty strings, or empty lists if data is not available.
2. Infer personality, behavior, and interests strictly from the Reddit history provided.
3. Keep each field concise, factual, and structured.
4. Do not include any explanation, extra text, or Markdown—output must be directly parseable JSON.

Generate the JSON now.
"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL_NAME", "nvidia/llama-3.3-nemotron-super-49b-v1"),
            messages=[
                {"role": "system", "content": "ONLY return valid JSON. No Markdown, comments, or extra text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            top_p=0.9,
            max_tokens=2048,
            stream=False
        )
        # support different response shapes
        text = None
        try:
            text = response.choices[0].message.content
        except Exception:
            try:
                # dict-like
                text = response["choices"][0].get("message", {}).get("content")
            except Exception:
                text = None

        if not text:
            logger.error("No content returned from LLM")
            return None

        parsed = try_parse_json_fallback(text.strip())
        if parsed:
            logger.info("Persona JSON generated")
            return parsed
        else:
            logger.error("Failed to parse persona JSON from LLM response")
            return None
    except Exception:
        logger.exception("LLM call failed")
        return None

# -------------------------
# Flask routes
# -------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Accept form-data or JSON payloads
        user_input = None
        # prefer form
        if request.form and request.form.get("username"):
            user_input = request.form.get("username")
        elif request.json and request.json.get("username"):
            user_input = request.json.get("username")
        elif request.values and request.values.get("username"):
            user_input = request.values.get("username")
        else:
            # Last resort: raw body
            try:
                raw = request.data.decode("utf-8").strip()
                if raw:
                    user_input = raw
            except Exception:
                user_input = None

        if not user_input or not user_input.strip():
            # Check if client wants JSON response
            if request.headers.get("Accept") and "application/json" in request.headers.get("Accept"):
                return jsonify({"error": "Please enter a Reddit username or profile link."}), 400
            flash("Please enter a Reddit username or profile link.", "error")
            logger.warning("No username provided")
            return redirect(url_for("index"))

        username = extract_username(user_input)
        if not username:
            # Check if client wants JSON response
            if request.headers.get("Accept") and "application/json" in request.headers.get("Accept"):
                return jsonify({"error": "Could not extract username. Provide a Reddit username or profile URL."}), 400
            flash("Could not extract username. Provide a Reddit username or profile URL.", "error")
            logger.warning("Extract username failed for input: %s", user_input)
            return redirect(url_for("index"))

        logger.info("Processing username: %s", username)

        try:
            # Step 1: Collect data
            posts, comments, metadata = get_reddit_data(username)
            if not (posts or comments or metadata):
                # Check if client wants JSON response
                if request.headers.get("Accept") and "application/json" in request.headers.get("Accept"):
                    return jsonify({"error": f"No data found for username '{username}'."}), 404
                flash(f"No data found for username '{username}'.", "error")
                logger.warning("No data for %s", username)
                return redirect(url_for("index"))

            # Step 2: Save raw combined text and chunk
            raw_file = save_raw_data(username, posts, comments, metadata)
            with open(raw_file, "r", encoding="utf-8") as fh:
                reddit_text = fh.read()

            chunks = chunk_text(reddit_text, max_len=500)
            logger.info("Total chunks created: %d", len(chunks))
            if not chunks:
                # Check if client wants JSON response
                if request.headers.get("Accept") and "application/json" in request.headers.get("Accept"):
                    return jsonify({"error": "No textual content to analyze for this user."}), 400
                flash("No textual content to analyze for this user.", "error")
                return redirect(url_for("index"))

            # Step 3: Embed & upsert into Pinecone (namespace = username)
            upserted = embed_and_upsert(chunks, username)
            logger.info("Upserted %d vectors for %s", upserted, username)

            # Step 4: Search relevant chunks for persona prompt
            query = (
                "Using the user's posts, comments and metadata, produce a concise persona summary and JSON."
            )
            searched = search_chunks(query, top_k=min(10, len(chunks)), username_namespace=username, raw_chunks=chunks)
            logger.info("Found %d relevant chunks for persona generation", len(searched))

            # Step 5: Generate persona via LLM
            persona_json = generate_persona_json(searched)
            if not persona_json:
                # Check if client wants JSON response
                if request.headers.get("Accept") and "application/json" in request.headers.get("Accept"):
                    return jsonify({"error": "Failed to generate persona from LLM. Try again later or with another user."}), 500
                flash("Failed to generate persona from LLM. Try again later or with another user.", "error")
                return redirect(url_for("index"))

            # Save persona file
            out_file = f"persona_output_{username}.json"
            with open(out_file, "w", encoding="utf-8") as fh:
                json.dump(persona_json, fh, indent=2, ensure_ascii=False)
            logger.info("Persona saved to %s", out_file)

            # Check if client wants JSON response
            if request.headers.get("Accept") and "application/json" in request.headers.get("Accept"):
                # Add stats to the response for consistency with frontend expectations
                response_data = {
                    "persona": persona_json,
                    "username": username,
                    "stats": {
                        "posts": len(posts),
                        "comments": len(comments),
                        "chunks": len(chunks),
                        "stored_vectors": upserted,
                        "relevant_chunks": len(searched)
                    }
                }
                return jsonify(response_data)

            # Render results for browser requests
            return render_template("persona.html", persona=persona_json)

        except Exception as e:
            logger.exception("Processing error for %s", user_input)
            # Check if client wants JSON response
            if request.headers.get("Accept") and "application/json" in request.headers.get("Accept"):
                return jsonify({"error": f"Error processing username '{user_input}': {str(e)}"}), 500
            flash(f"Error processing username '{user_input}': {str(e)}", "error")
            return redirect(url_for("index"))

    # GET
    return render_template("index.html")


if __name__ == "__main__":
    # run on 0.0.0.0 so you can access it from other devices in local network if needed
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
