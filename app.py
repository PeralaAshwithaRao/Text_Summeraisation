"""
Text Summarization App

Summarizes long text, TXT files, and PDF files using BART models.
Long documents are divided into chunks, summarized, and combined.

Runs locally. No API key required.
"""

import streamlit as st
import torch
from pypdf import PdfReader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide"
)


# --------------------------------------------------
# Models
# --------------------------------------------------

MODELS = {
    "DistilBART (Faster, Lighter)": "sshleifer/distilbart-cnn-12-6",
    "BART Large (Better Quality)": "facebook/bart-large-cnn"
}


# --------------------------------------------------
# Summary Lengths
# --------------------------------------------------

LENGTHS = {
    "Short": (30, 80),
    "Medium": (60, 150),
    "Long": (100, 250)
}


# Number of words in each input chunk
CHUNK_WORDS = 500


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading model... First run may take some time.")
def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    # Use CPU
    model.to("cpu")
    model.eval()

    return tokenizer, model


# --------------------------------------------------
# Split Text Into Chunks
# --------------------------------------------------

def chunk_text(text, size=CHUNK_WORDS):
    words = text.split()

    chunks = []

    for i in range(0, len(words), size):
        chunk = " ".join(words[i:i + size])

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# --------------------------------------------------
# Summarize One Chunk
# --------------------------------------------------

def summarize_chunk(text, tokenizer, model, min_len, max_len):

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    # Prevent minimum length from becoming too large
    input_words = len(text.split())

    safe_min_len = min(min_len, max(10, input_words // 3))

    # Make sure min length is smaller than max length
    safe_max_len = max(max_len, safe_min_len + 10)

    with torch.no_grad():

        summary_ids = model.generate(
            **inputs,
            min_length=safe_min_len,
            max_length=safe_max_len,
            num_beams=4,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary


# --------------------------------------------------
# Summarize Complete Text
# --------------------------------------------------

def summarize_text(text, tokenizer, model, min_len, max_len):

    chunks = chunk_text(text)

    if not chunks:
        return ""

    progress = st.progress(
        0,
        text="Starting summarization..."
    )

    summaries = []

    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks, start=1):

        summary = summarize_chunk(
            chunk,
            tokenizer,
            model,
            min_len,
            max_len
        )

        summaries.append(summary)

        progress.progress(
            i / total_chunks,
            text=f"Summarized chunk {i}/{total_chunks}"
        )

    progress.empty()

    # Combine summaries
    combined_summary = " ".join(summaries)

    # If there are many chunks, summarize the combined result again
    if len(chunks) > 1 and len(combined_summary.split()) > CHUNK_WORDS:

        combined_summary = summarize_chunk(
            combined_summary,
            tokenizer,
            model,
            min_len,
            min(max_len * 2, 500)
        )

    return combined_summary


# --------------------------------------------------
# Read PDF
# --------------------------------------------------

def read_pdf(file):

    reader = PdfReader(file)

    pages = []

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            pages.append(page_text)

    return "\n".join(pages)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("⚙️ Settings")

model_name = st.sidebar.selectbox(
    "Choose Model",
    list(MODELS.keys())
)

length = st.sidebar.radio(
    "Summary Length",
    list(LENGTHS.keys()),
    index=1
)


# --------------------------------------------------
# Main Application
# --------------------------------------------------

st.title("📝 Text Summarization App")

st.write(
    "Paste text or upload a TXT/PDF file to generate a concise summary."
)


# --------------------------------------------------
# Input Type
# --------------------------------------------------

source = st.radio(
    "Select Input Type",
    ["Paste Text", "Upload File"],
    horizontal=True
)


text = ""


# --------------------------------------------------
# Paste Text
# --------------------------------------------------

if source == "Paste Text":

    text = st.text_area(
        "Enter your text",
        height=300,
        placeholder="Paste an article, notes, report, or any long text here..."
    )


# --------------------------------------------------
# Upload File
# --------------------------------------------------

else:

    uploaded_file = st.file_uploader(
        "Upload a TXT or PDF file",
        type=["txt", "pdf"]
    )

    if uploaded_file is not None:

        if uploaded_file.name.lower().endswith(".pdf"):

            text = read_pdf(uploaded_file)

        else:

            text = uploaded_file.read().decode(
                "utf-8",
                errors="ignore"
            )

        if text.strip():

            with st.expander("📄 Preview Extracted Text"):

                preview = text[:3000]

                if len(text) > 3000:
                    preview += "..."

                st.write(preview)


# --------------------------------------------------
# Summarize Button
# --------------------------------------------------

if st.button(
    "⚡ Summarize",
    type="primary",
    use_container_width=True
):

    text = text.strip()

    word_count = len(text.split())

    # Check minimum words
    if word_count < 50:

        st.warning(
            "⚠️ Please provide at least 50 words of text."
        )

    else:

        # Load selected model
        model_id = MODELS[model_name]

        tokenizer, model = load_model(model_id)

        # Get selected summary length
        min_len, max_len = LENGTHS[length]

        # Generate summary
        with st.spinner("Generating summary..."):

            summary = summarize_text(
                text,
                tokenizer,
                model,
                min_len,
                max_len
            )

        # --------------------------------------------------
        # Display Results
        # --------------------------------------------------

        col1, col2 = st.columns(2)

        # Original Text
        with col1:

            st.subheader("📄 Original Text")

            st.caption(
                f"{word_count} words"
            )

            original_preview = text[:5000]

            if len(text) > 5000:
                original_preview += "..."

            st.write(original_preview)

        # Summary
        with col2:

            st.subheader("✨ Summary")

            summary_word_count = len(summary.split())

            if word_count > 0:

                reduction = max(
                    0,
                    100 - (summary_word_count * 100 // word_count)
                )

            else:

                reduction = 0

            st.caption(
                f"{summary_word_count} words • "
                f"{reduction}% shorter"
            )

            st.success(summary)

            # Download button
            st.download_button(
                label="⬇️ Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )