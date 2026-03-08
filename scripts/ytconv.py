import os
import re

from youtube_transcript_api import YouTubeTranscriptApi

def extract_video_id(url_or_id: str) -> str:
    """Extract video ID from YouTube URL or return as-is if already an ID."""
    m = re.search(r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url_or_id)
    return m.group(1) if m else url_or_id

video_url = "https://www.youtube.com/watch?v=rEnf_CFoyv0"
video_id = extract_video_id(video_url)
ytt_api = YouTubeTranscriptApi()
fetched_transcript = ytt_api.fetch(video_id)
text = " ".join(snippet.text for snippet in fetched_transcript)

from openai import OpenAI

client = OpenAI(
  base_url="https://api.featherless.ai/v1",
  api_key=os.environ.get("FEATHERLESS_API_KEY"),
)

response = client.chat.completions.create(
  model='deepseek-ai/DeepSeek-V3.2',
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize this video into a concise summary of the maingovernmental policies in 3 short sentences for this starter Do you support the policy that : " + text}
  ],
)
print("Do you support the policy that " + response.model_dump()['choices'][0]['message']['content'])