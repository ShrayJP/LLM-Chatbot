import streamlit as st
import openai

# ✅ Configure Streamlit page
st.set_page_config(page_title="LLaMA Chatbot", layout="centered")
st.title("AI-Powered ChatBot 🤖")

# ✅ Inject custom CSS for sidebar shading
st.markdown("""
    <style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #6d6c6c;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

# ✅ Set OpenRouter API credentials
openai.api_key = st.secrets["OPENROUTER_API_KEY"]
openai.api_base = "https://openrouter.ai/api/v1"

# ------------------ Model Selector ------------------ #
available_models = {
    "LLaMA 3 (8B)": "meta-llama/llama-3-8b-instruct",
    "Mistral 7B": "mistralai/mistral-7b-instruct",
    "Mixtral 8x7B": "mistralai/mixtral-8x7b-instruct"
}

with st.sidebar:
    selected_model_label = st.selectbox("Choose LLM model", list(available_models.keys()))
    selected_model_id = available_models[selected_model_label]

    clear_button = st.markdown("""
    <style>
    .custom-button {
        display: inline-block;
        padding: 0.5rem 1rem;
        background-color: #dc3545;
        color: white;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
        cursor: pointer;
        margin-top: 10px;
        transition: background-color 0.3s ease;
    }
    .custom-button:hover {
        background-color: #e60000;
    }
    </style>
    <div class="custom-button" onclick="document.querySelector('button[kind=secondary]').click()">🧹 Clear Chat History</div>
""", unsafe_allow_html=True)

    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []

# ✅ Set selected model
st.session_state["llm_model"] = selected_model_id

# ------------------ Chat History ------------------ #
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Render previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ------------------ Streaming Generator ------------------ #
def stream_generator(response_stream):
    for chunk in response_stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# ------------------ Chat Input ------------------ #
if prompt := st.chat_input("Talk to LLaMA..."):
    # Save user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Query OpenRouter via OpenAI-compatible API
    with st.chat_message("assistant"):
        stream = openai.ChatCompletion.create(
            model=st.session_state["llm_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream_generator(stream))

    # Save assistant's response
    st.session_state.messages.append({"role": "assistant", "content": response})
