from fastapi import FastAPI, UploadFile, File
from gradio_client import Client, handle_file
from bs4 import BeautifulSoup
import tempfile
import os
import re

app = FastAPI(
    title="Stuttering Classification API",
    description="API for stuttering classification using Hugging Face Space"
)

client = Client("vocametrix/stuttering-classification")


def parse_result(html_result):

    soup = BeautifulSoup(html_result, "html.parser")

    chunks = []

    for row in soup.find_all("tr"):

        cells = row.find_all("td")

        if len(cells) < 4:
            continue

        chunk_text = cells[0].get_text(strip=True)
        timestamp_text = cells[1].get_text(strip=True)
        label = cells[2].get_text(" ", strip=True)
        confidence_text = cells[3].get_text(" ", strip=True)

        if not chunk_text.isdigit():
            continue

        time_match = re.search(
            r"([\d.]+)s\s*[—-]\s*([\d.]+)s",
            timestamp_text
        )

        if not time_match:
            continue

        start_time = float(time_match.group(1))
        end_time = float(time_match.group(2))

        confidence_match = re.search(
            r"([\d.]+)\s*%",
            confidence_text
        )

        confidence = (
            float(confidence_match.group(1))
            if confidence_match
            else None
        )

        chunks.append({
            "chunk": int(chunk_text),
            "start": start_time,
            "end": end_time,
            "label": label,
            "confidence": confidence
        })

    return chunks


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Save uploaded audio temporarily
    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(await file.read())
        audio_path = temp_file.name

    try:

        # Send audio to Hugging Face Space
        result = client.predict(
            audio_input=handle_file(audio_path),
            chunk_duration=4,
            overlap_pct=50,
            api_name="/process_audio"
        )

        # Convert HTML → structured data
        chunks = parse_result(result)

        return {
            "filename": file.filename,
            "total_chunks": len(chunks),
            "chunks": chunks
        }

    finally:

        if os.path.exists(audio_path):
            os.remove(audio_path)