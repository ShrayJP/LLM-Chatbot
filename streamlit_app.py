import streamlit as st
import openai  # Used for OpenRouter-compatible models

st.set_page_config(page_title="LLaMA Chatbot", layout="centered")
st.title("General Purpose LLaMA Chatbot 🦙")

# Configure OpenRouter API
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

    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []

# Set selected model in session
st.session_state["llm_model"] = selected_model_id

# ------------------ Chat History ------------------ #
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ------------------ Stream Response Function ------------------ #
def stream_generator(stream):
    for chunk in stream:
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        if "content" in delta:
            yield delta["content"]

# ------------------ Chat Input ------------------ #
if prompt := st.chat_input("Talk to LLaMA..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

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

    st.session_state.messages.append({"role": "assistant", "content": response})
