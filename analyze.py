#!/usr/bin/env python3
"""
YouTube Cinematography Analyzer
Analyzes new videos from a YouTube playlist from a senior cinematographer's POV.
Saves individual JSON files per video. Tracks processed videos to avoid re-analysis.
"""

import os
import json
import re
import base64
import ssl
import urllib.request
import urllib.error
import httplib2
import requests
from pathlib import Path
from datetime import datetime
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import anthropic

# Honor system CA bundle (self-signed cert in sandbox chain)
_CA_BUNDLE = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")

# ── Config ────────────────────────────────────────────────────────────────────
YOUTUBE_API_KEY = os.environ["YOUTUBE_API_KEY"]
PLAYLIST_ID     = os.environ.get("PLAYLIST_ID", "PL-7ziKOnwNCshaIEzDNgGttU7ms2Bcbo3")
ANTHROPIC_KEY   = os.environ["ANTHROPIC_API_KEY"]
GH_TOKEN        = os.environ["GH_TOKEN"]
GH_REPO         = "fourestbox-creator/Learner"
OUTPUT_DIR      = Path("output")
PROCESSED_FILE  = Path("processed.json")

SYSTEM_PROMPT = """You are a senior cinematographer with 25+ years of experience across feature films,
documentaries, music videos, and commercial work. You have a trained eye for visual storytelling,
technical cinematographic craft, and artistic intent.

Your job: analyze a YouTube video from a deep cinematographic perspective — shot design, movement,
light, color, sound, editing rhythm, and overall visual language. Write as if preparing masterclass
notes for film students. Be specific, technical, and insightful. Use proper industry terminology."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_processed() -> dict:
    if PROCESSED_FILE.exists():
        return json.loads(PROCESSED_FILE.read_text())
    return {}


def save_processed(data: dict):
    PROCESSED_FILE.write_text(json.dumps(data, indent=2))


def get_playlist_videos(youtube) -> list[dict]:
    videos, request = [], youtube.playlistItems().list(
        part="snippet,contentDetails",
        playlistId=PLAYLIST_ID,
        maxResults=50
    )
    while request:
        response = request.execute()
        for item in response.get("items", []):
            snippet  = item["snippet"]
            video_id = item["contentDetails"]["videoId"]
            if snippet["title"] == "Deleted video":
                continue
            videos.append({
                "video_id":    video_id,
                "title":       snippet["title"],
                "description": snippet.get("description", ""),
                "published_at": snippet.get("publishedAt", ""),
                "thumbnails":  snippet.get("thumbnails", {}),
            })
        request = youtube.playlistItems().list_next(request, response)
    return videos


def enrich_video(youtube, video: dict) -> dict:
    resp = youtube.videos().list(
        part="snippet,contentDetails,statistics",
        id=video["video_id"]
    ).execute()
    if not resp["items"]:
        return video
    item = resp["items"][0]
    video["duration"]   = item["contentDetails"].get("duration", "")
    video["tags"]       = item["snippet"].get("tags", [])
    video["thumbnails"] = item["snippet"].get("thumbnails", video["thumbnails"])
    video["view_count"] = item["statistics"].get("viewCount", "")
    return video


def get_transcript(video_id: str) -> str | None:
    try:
        entries = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join(e["text"] for e in entries)
    except Exception:
        return None


def fetch_thumbnail_b64(thumbnails: dict) -> str | None:
    for quality in ("maxres", "standard", "high", "medium", "default"):
        if quality in thumbnails:
            try:
                r = requests.get(thumbnails[quality]["url"], timeout=10)
                if r.status_code == 200:
                    return base64.standard_b64encode(r.content).decode()
            except Exception:
                continue
    return None


def slugify(title: str) -> str:
    s = re.sub(r"[^\w\s-]", "", title.lower())
    s = re.sub(r"[\s_-]+", "-", s)
    return s[:60].strip("-")


# ── Core analysis ─────────────────────────────────────────────────────────────

def build_analysis_prompt(video: dict, transcript: str | None) -> str:
    tags       = ", ".join(video.get("tags", [])) or "N/A"
    desc_snip  = (video.get("description") or "")[:600]
    xscript    = (transcript or "")[:4000] or "No transcript available."
    ts_now     = datetime.utcnow().isoformat() + "Z"

    return f"""
Analyze the following YouTube video from a senior cinematographer's perspective.

--- VIDEO METADATA ---
Title:       {video['title']}
Video ID:    {video['video_id']}
Duration:    {video.get('duration', 'Unknown')}
Views:       {video.get('view_count', 'Unknown')}
Tags:        {tags}
Description: {desc_snip}

--- TRANSCRIPT (first 4000 chars) ---
{xscript}

--- TASK ---
Return ONLY valid JSON in exactly this structure (no markdown fences, no extra keys):

{{
  "video_id": "{video['video_id']}",
  "title": "{video['title']}",
  "analyzed_at": "{ts_now}",
  "cinematography": {{
    "dominant_shot_types": ["<list of shot types seen/inferred>"],
    "camera_angles": ["<list of angles>"],
    "lens_choice_inference": "<prime vs zoom, focal length estimate, reasoning>",
    "framing_techniques": ["<rule of thirds, leading lines, negative space, etc.>"],
    "composition_principles": ["<golden ratio, symmetry, layering, etc.>"],
    "depth_of_field_usage": "<shallow / deep / mixed — with context>"
  }},
  "motion": {{
    "camera_movements": ["<pan, tilt, dolly, crane, handheld, steadicam, etc.>"],
    "movement_motivation": "<why these moves serve the story/emotion>",
    "subject_motion_handling": "<how subject movement is accommodated>",
    "pacing_rhythm": "<slow burn / kinetic / rhythmic — describe>",
    "shot_sequences": [
      {{
        "sequence_description": "<what happens visually>",
        "purpose": "<emotional/narrative intent>"
      }}
    ],
    "chops_and_cuts": {{
      "editing_style": "<continuity / montage / jump cut / match cut>",
      "cut_frequency": "<fast / medium / slow>",
      "transitions_used": ["<cut, dissolve, wipe, L-cut, J-cut, etc.>"],
      "rhythm_with_audio": "<how cuts sync with music/sound>"
    }}
  }},
  "audio": {{
    "foley_layers": ["<specific foley elements: footsteps, fabric, objects, etc.>"],
    "sound_design_notes": "<overall approach to sound design>",
    "music_placement_and_role": "<where music enters/exits, what it does emotionally>",
    "dialogue_clarity_and_treatment": "<treatment of spoken word if present>",
    "ambient_sound_strategy": "<use of room tone, environment, atmos>",
    "audio_visual_sync": "<how sound and picture work together>"
  }},
  "lighting": {{
    "inferred_lighting_setup": "<3-point, natural, motivated, practical, etc.>",
    "key_light_direction": "<angle and quality of key light>",
    "color_temperature": "<warm / cool / mixed — Kelvin range estimate>",
    "color_grading_style": "<teal-orange, desaturated, high contrast, film emulation, etc.>",
    "contrast_and_dynamic_range": "<flat / contrasty / crushed blacks / lifted shadows>",
    "mood_through_light_and_color": "<emotional effect of the lighting/color choices>"
  }},
  "visual_storytelling": {{
    "narrative_visual_language": "<how visuals advance story or message>",
    "symbolism_or_motifs": ["<recurring visual ideas, symbols>"],
    "emotional_impact_techniques": ["<specific techniques that drive emotion>"]
  }},
  "technical_execution": {{
    "estimated_equipment": ["<camera body, lenses, rigs, stabilizers inferred>"],
    "technical_strengths": ["<what is executed exceptionally well>"],
    "areas_for_improvement": ["<constructive critique from a senior DP perspective>"]
  }},
  "senior_cinematographer_notes": "<2-3 paragraph masterclass-style commentary on this work>"
}}
"""


def analyze_video(client: anthropic.Anthropic, video: dict, transcript: str | None, thumbnail_b64: str | None) -> dict:
    content = []

    if thumbnail_b64:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": thumbnail_b64}
        })

    content.append({"type": "text", "text": build_analysis_prompt(video, transcript)})

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}]
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(raw)


# ── GitHub API push ───────────────────────────────────────────────────────────

def _gh_api(method: str, path: str, data: dict | None = None) -> dict:
    url = f"https://api.github.com/repos/{GH_REPO}/contents/{path}"
    headers = {
        "Authorization": f"Bearer {GH_TOKEN}",
        "Accept":        "application/vnd.github+json",
        "Content-Type":  "application/json",
    }
    req = urllib.request.Request(url, headers=headers, method=method)
    if data:
        req.data = json.dumps(data).encode()
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


def github_upload(local_path: Path, repo_path: str, commit_msg: str):
    with open(local_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode()
    existing = _gh_api("GET", repo_path)
    sha = existing.get("sha")
    payload: dict = {"message": commit_msg, "content": content_b64}
    if sha:
        payload["sha"] = sha
    result = _gh_api("PUT", repo_path, payload)
    if "content" in result:
        print(f"  Uploaded: {repo_path}")
    else:
        print(f"  Upload FAILED {repo_path}: {result.get('message', result)}")


def git_commit_push(new_files: list[Path]):
    if not new_files:
        return
    msg = f"feat: add {len(new_files)} cinematography analysis JSON(s)"
    for f in new_files:
        github_upload(f, str(f), msg)
    github_upload(PROCESSED_FILE, str(PROCESSED_FILE), msg)
    print(f"Pushed {len(new_files)} file(s) via GitHub API.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n=== YouTube Cinematography Analyzer === {datetime.utcnow().isoformat()}Z")

    OUTPUT_DIR.mkdir(exist_ok=True)
    processed = load_processed()

    http_obj = httplib2.Http(ca_certs=_CA_BUNDLE) if _CA_BUNDLE else None
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY, http=http_obj)
    client  = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    print(f"Fetching playlist: {PLAYLIST_ID}")
    all_videos  = get_playlist_videos(youtube)
    new_videos  = [v for v in all_videos if v["video_id"] not in processed]

    print(f"Total: {len(all_videos)} | Already processed: {len(all_videos) - len(new_videos)} | New: {len(new_videos)}")

    if not new_videos:
        print("No new videos to analyze. Exiting.")
        return

    saved_files = []

    for video in new_videos:
        video_id = video["video_id"]
        print(f"\n→ Analyzing: {video['title']} ({video_id})")

        try:
            video        = enrich_video(youtube, video)
            transcript   = get_transcript(video_id)
            thumb_b64    = fetch_thumbnail_b64(video.get("thumbnails", {}))

            print(f"  Transcript: {'found' if transcript else 'not available'}")
            print(f"  Thumbnail:  {'fetched' if thumb_b64 else 'unavailable'}")

            analysis     = analyze_video(client, video, transcript, thumb_b64)

            filename     = f"{video_id}_{slugify(video['title'])}.json"
            out_path     = OUTPUT_DIR / filename
            out_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False))

            saved_files.append(out_path)
            processed[video_id] = {
                "title":       video["title"],
                "analyzed_at": datetime.utcnow().isoformat() + "Z",
                "file":        str(out_path)
            }
            save_processed(processed)
            print(f"  Saved: {out_path}")

        except Exception as e:
            print(f"  ERROR analyzing {video_id}: {e}")
            continue

    git_commit_push(saved_files)
    print(f"\n=== Done. Analyzed and saved {len(saved_files)} video(s). ===\n")


if __name__ == "__main__":
    main()
