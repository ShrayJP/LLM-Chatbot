import streamlit as st
import openai
from PyPDF2 import PdfReader
from PIL import Image
import base64
import io

# Set page title and layout
st.set_page_config(page_title="LLM Chatbot", layout="centered")
st.title("AI-Powered Chatbot")

# Load API key and base URL for OpenRouter
openai.api_key = st.secrets["OPENROUTER_API_KEY"]
openai.api_base = "https://openrouter.ai/api/v1"

# List of available models for the user to choose from
available_models = {
    "LLaMA 3 (8B)": "meta-llama/llama-3-8b-instruct",
    "Mistral 7B": "mistralai/mistral-7b-instruct",
    "Mixtral 8x7B": "mistralai/mixtral-8x7b-instruct",
    "Qwen": "qwen/qwen2.5-vl-32b-instruct:free"
}

# Sidebar for model selection, file uploads, and reset option
with st.sidebar:
    selected_model_label = st.selectbox("Choose LLM model", list(available_models.keys()))
    selected_model_id = available_models[selected_model_label]

    uploaded_file = st.file_uploader("📄 Upload a text or PDF file", type=["txt", "pdf"])
    uploaded_image = st.file_uploader("🖼️ Upload an image", type=["jpg", "jpeg", "png"])

    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []  # Clear stored messages

# Store selected model in session state
st.session_state["llm_model"] = selected_model_id

# Check if model is vision-capable (e.g., Qwen or LLaVA)
def is_vision_model(model_id: str) -> bool:
    return any(tag in model_id.lower() for tag in ["llava", "qwen", "vision"])

# Extract text from uploaded PDF or TXT
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    else:
        text = uploaded_file.read().decode("utf-8")
    st.session_state["uploaded_file_text"] = text  # Save extracted text
    st.success("✅ Text extracted from uploaded file.")

# Convert uploaded image to base64 string (for models that accept image input)
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    st.session_state["uploaded_image_base64"] = img_b64

# Initialize chat history if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages (user and assistant)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input from chat box
if prompt := st.chat_input("Talk to LLaMA..."):
    prompt = prompt.strip()
    if prompt:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            base_messages = st.session_state.messages.copy()

            # Inject uploaded file content into prompt as system message
            file_context = st.session_state.get("uploaded_file_text", "")
            if file_context:
                system_message = {
                    "role": "system",
                    "content": f"Use the following file content as context for the user's queries:\n{file_context}"
                }
                base_messages = [system_message] + base_messages

            # If model supports vision and image is uploaded, include image
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

                # Stream and display assistant's response
                full_response = ""
                response_box = st.empty()
                for chunk in stream:
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if "content" in delta:
                        full_response += delta["content"]
                        response_box.markdown(full_response)

                # Save assistant's response
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            else:
                # Handle normal text-only chat
                stream = openai.ChatCompletion.create(
                    model=selected_model_id,
                    messages=base_messages,
                    stream=True,
                )

                full_response = ""
                response_box = st.empty()
                for chunk in stream:
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if "content" in delta:
                        full_response += delta["content"]
                        response_box.markdown(full_response)

                # Save assistant's response
                st.session_state.messages.append({"role": "assistant", "content": full_response})
