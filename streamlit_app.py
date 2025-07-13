import streamlit as st
import openai
from PyPDF2 import PdfReader
from PIL import Image
import base64
import io
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ───────────────────────────────────────────────────────────────
# 📌 Page Configuration
st.set_page_config(page_title="LLM Chatbot", layout="centered")
st.title("AI-Powered Chatbot with RAG")

# ───────────────────────────────────────────────────────────────
# 🔐 OpenAI API Key (via OpenRouter)
openai.api_key = st.secrets["OPENROUTER_API_KEY"]
openai.api_base = "https://openrouter.ai/api/v1"

# ───────────────────────────────────────────────────────────────
# 📚 Available Models
available_models = {
    "LLaMA 3 (8B)": "meta-llama/llama-3-8b-instruct",
    "Mistral 7B": "mistralai/mistral-7b-instruct",
    "Mixtral 8x7B": "mistralai/mixtral-8x7b-instruct",
    "Qwen": "qwen/qwen2.5-vl-32b-instruct:free"
}

# ───────────────────────────────────────────────────────────────
# 🧠 Text Chunking & RAG Helpers
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

def embed_chunks(chunks):
    if not chunks:
        return np.array([]), None
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(chunks)
    return np.array(embeddings), embed_model

def build_faiss_index(embeddings):
    if len(embeddings.shape) != 2:
        raise ValueError("Embeddings must be a 2D array. Likely cause: empty or invalid text.")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_relevant_chunks(query, chunks, index, embed_model, top_k=5):
    query_embedding = embed_model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in I[0]]

# ───────────────────────────────────────────────────────────────
# ⏱ Sidebar Options
with st.sidebar:
    selected_model_label = st.selectbox("Choose LLM model", list(available_models.keys()))
    selected_model_id = available_models[selected_model_label]

    uploaded_file = st.file_uploader("📄 Upload a text or PDF file", type=["txt", "pdf"])
    uploaded_image = st.file_uploader("🖼️ Upload an image", type=["jpg", "jpeg", "png"])

    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []

# Store selected model in session state
st.session_state["llm_model"] = selected_model_id

# ───────────────────────────────────────────────────────────────
# 🧠 Check if model supports images
def is_vision_model(model_id: str) -> bool:
    return any(tag in model_id.lower() for tag in ["llava", "qwen", "vision"])

# ───────────────────────────────────────────────────────────────
# 📄 Extract Text from File and Build RAG Index
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    else:
        text = uploaded_file.read().decode("utf-8")

    if not text.strip():
        st.error("❌ No text found in the uploaded file. Make sure it's not a scanned image or empty.")
        st.stop()

    st.session_state["uploaded_file_text"] = text
    st.success("✅ Text extracted from uploaded file.")

    with st.spinner("🔍 Preparing document for retrieval..."):
        chunks = chunk_text(text)
        if not chunks:
            st.error("❌ Failed to split the document into chunks. Please upload a longer or more readable file.")
            st.stop()

        embeddings, embed_model = embed_chunks(chunks)
        try:
            index = build_faiss_index(embeddings)
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.stop()

        st.session_state["rag_chunks"] = chunks
        st.session_state["rag_embed_model"] = embed_model
        st.session_state["rag_index"] = index

# ───────────────────────────────────────────────────────────────
# 🖼️ Image Handling
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    st.session_state["uploaded_image_base64"] = img_b64

# ───────────────────────────────────────────────────────────────
# 💬 Chat Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ───────────────────────────────────────────────────────────────
# 💬 Handle User Input
if prompt := st.chat_input("Talk to your chatbot..."):
    prompt = prompt.strip()
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            base_messages = st.session_state.messages.copy()

            # 🧠 RAG CONTEXT INJECTION
            retrieved_context = ""
            if "rag_chunks" in st.session_state:
                chunks = st.session_state["rag_chunks"]
                index = st.session_state["rag_index"]
                embed_model = st.session_state["rag_embed_model"]
                top_chunks = retrieve_relevant_chunks(prompt, chunks, index, embed_model)
                retrieved_context = "\n\n".join(top_chunks)

                system_message = {
                    "role": "system",
                    "content": f"Use the following context to answer the query:\n{retrieved_context}"
                }
                base_messages = [system_message] + base_messages

            # 🖼️ Vision Model Handling
            if is_vision_model(selected_model_id) and "uploaded_image_base64" in st.session_state:
                stream = openai.ChatCompletion.create(
                    model=selected_model_id,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{st.session_state['uploaded_image_base64']}"
                                },
                            },
                        ],
                    }],
                    stream=True,
                )
            else:
                stream = openai.ChatCompletion.create(
                    model=selected_model_id,
                    messages=base_messages,
                    stream=True,
                )

            # 📤 Stream Response
            full_response = ""
            response_box = st.empty()
            for chunk in stream:
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if "content" in delta:
                    full_response += delta["content"]
                    response_box.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
