import streamlit as st
import openai
from PyPDF2 import PdfReader
from PIL import Image
import base64, io, difflib

# -----------------------
# Page setup
# -----------------------
st.set_page_config(page_title="LLM Chatbot", layout="centered")
st.title("AI-Powered Chatbot with Consistency Check")

openai.api_key = st.secrets["OPENROUTER_API_KEY"]
openai.api_base = "https://openrouter.ai/api/v1"

# -----------------------
# Model selection
# -----------------------
available_models = {
    "LLaMA 3 (8B)": "meta-llama/llama-3-8b-instruct",
    "Mistral 7B": "mistralai/mistral-7b-instruct",
    "Mixtral 8x7B": "mistralai/mixtral-8x7b-instruct",
    "Qwen": "qwen/qwen2.5-vl-32b-instruct:free"
}

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    selected_model_label = st.selectbox("Choose LLM model", list(available_models.keys()))
    selected_model_id = available_models[selected_model_label]

    consistency_mode = st.radio(
        "Consistency Mode",
        ["Normal", "Reflection", "Dual-Response"],
        index=0
    )

    uploaded_file = st.file_uploader("📄 Upload a text or PDF file", type=["txt", "pdf"])
    uploaded_image = st.file_uploader("🖼️ Upload an image", type=["jpg", "jpeg", "png"])

    if st.button("🧹 Clear Chat History"):
        st.session_state["messages"] = []

# -----------------------
# Session state init
# -----------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "uploaded_file_text" not in st.session_state:
    st.session_state["uploaded_file_text"] = ""
if "uploaded_image_base64" not in st.session_state:
    st.session_state["uploaded_image_base64"] = None
if "reflections" not in st.session_state:  # keep reflections separate
    st.session_state["reflections"] = []

st.session_state["llm_model"] = selected_model_id

# -----------------------
# Helper functions
# -----------------------
def is_vision_model(model_id: str) -> bool:
    return any(tag in model_id.lower() for tag in ["llava", "qwen", "vision"])

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        try:
            return "\n".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            st.error(f"PDF extraction error: {e}")
            return ""
    else:
        return uploaded_file.read().decode("utf-8")

def encode_image_to_b64(uploaded_image):
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def stream_response(model_id, messages, vision_b64=None):
    """Stream a response from OpenRouter."""
    if vision_b64 and is_vision_model(model_id):
        payload = [{
            "role": "user",
            "content": [
                {"type": "text", "text": messages[-1]["content"]},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{vision_b64}"}}
            ],
        }]
    else:
        payload = messages

    try:
        stream = openai.ChatCompletion.create(
            model=model_id,
            messages=payload,
            stream=True,
        )
    except Exception as e:
        st.error(f"⚠️ Model error: {e}")
        return "Error: model call failed."

    full_response, response_box = "", st.empty()
    for chunk in stream:
        delta = chunk["choices"][0].get("delta", {})
        if "content" in delta:
            full_response += delta["content"]
            response_box.markdown(full_response)
    return full_response

# --- Consistency check functions ---
def reflection_check(model_id, answer):
    reflection_prompt = [
        {"role": "system", "content": "You are a critic. Assess the assistant's last answer for correctness, clarity, and consistency. Suggest improvements if needed."},
        {"role": "user", "content": answer}
    ]
    try:
        resp = openai.ChatCompletion.create(model=model_id, messages=reflection_prompt)
        return resp["choices"][0].get("message", {}).get("content", "No reflection generated.")
    except Exception as e:
        return f"Reflection error: {e}"

def dual_response_check(model_id, base_messages):
    try:
        # First answer
        resp1 = openai.ChatCompletion.create(model=model_id, messages=base_messages)
        ans1 = resp1["choices"][0]["message"]["content"]

        # Second answer with variation
        alt_messages = base_messages + [
            {"role": "user", "content": "Please answer again with slightly different phrasing or reasoning."}
        ]
        resp2 = openai.ChatCompletion.create(model=model_id, messages=alt_messages)
        ans2 = resp2["choices"][0]["message"]["content"]

        ratio = difflib.SequenceMatcher(None, ans1, ans2).ratio()
        return ans1, ans2, ratio
    except Exception as e:
        return "Error in dual-response", f"Error: {e}", 0.0

# -----------------------
# Handle file & image
# -----------------------
if uploaded_file:
    text = extract_text_from_file(uploaded_file)
    st.session_state["uploaded_file_text"] = text
    st.success("✅ Text extracted from uploaded file.")

if uploaded_image:
    st.session_state["uploaded_image_base64"] = encode_image_to_b64(uploaded_image)

# -----------------------
# Display history
# -----------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display reflections separately
for reflection in st.session_state["reflections"]:
    st.info(f"🧐 Consistency Check:\n\n{reflection}")

# -----------------------
# Handle user input
# -----------------------
if prompt := st.chat_input("Talk to the chatbot..."):
    prompt = prompt.strip()
    if prompt:
        # Log user input
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Build base messages with context
            base_messages = st.session_state.messages.copy()
            file_context = st.session_state.get("uploaded_file_text", "")
            if file_context and not any(m["role"] == "system" for m in base_messages):
                base_messages.insert(0, {
                    "role": "system",
                    "content": f"Use the following file content as context:\n{file_context}"
                })

            # -----------------------
            # Consistency Modes
            # -----------------------
            if consistency_mode == "Normal":
                answer = stream_response(selected_model_id, base_messages, st.session_state["uploaded_image_base64"])
                st.session_state.messages.append({"role": "assistant", "content": answer})

            elif consistency_mode == "Reflection":
                answer = stream_response(selected_model_id, base_messages, st.session_state["uploaded_image_base64"])
                st.session_state.messages.append({"role": "assistant", "content": answer})

                reflection = reflection_check(selected_model_id, answer)
                st.session_state["reflections"].append(reflection)
                st.info(f"🧐 Consistency Check:\n\n{reflection}")

            elif consistency_mode == "Dual-Response":
                ans1, ans2, ratio = dual_response_check(selected_model_id, base_messages)
                st.markdown(ans1)

                if ratio < 0.7:
                    st.warning("⚠️ Inconsistency detected between two generated answers.")
                    with st.expander("Alternative Answer"):
                        st.markdown(ans2)
                else:
                    with st.expander("Alternative phrasing (consistent)"):
                        st.markdown(ans2)

                st.session_state.messages.append({"role": "assistant", "content": ans1})
