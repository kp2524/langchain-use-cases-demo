---
title: LangChain Use Cases Demo
emoji: 🤖
colorFrom: blue
colorTo: purple
---

# LangChain Use Cases Demo

Interactive demonstration of 8 LangChain use cases powered by Groq AI.

## 🎯 Features

1. **💬 AI-Powered Chatbots** - Context-aware conversations with memory
2. **📄 Document Question Answering** - Ask questions about PDF, DOCX, and TXT files
3. **🔍 RAG (Retrieval-Augmented Generation)** - Knowledge base queries with citations
4. **📝 Document Summarization** - Automatic text summarization (short/medium/long)
5. **📊 Data Extraction** - Extract structured JSON from unstructured text
6. **✍️ Content Generation** - Generate context-aware content (emails, blogs, social media)
7. **⚙️ Workflow Automation** - Multi-step AI workflow demonstrations
8. **🛠️ Custom AI Tools** - Calculator, code generator, and text analyzer

## 🚀 Quick Start

### Local Testing

1. **Clone and setup:**
   ```bash
   git clone https://github.com/kp2524/langchain-use-cases-demo.git
   cd langchain-use-cases-demo
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Get a free Groq API key:**
   - Visit [console.groq.com](https://console.groq.com)
   - Sign up (free)
   - Create API key

3. **Set API key in code:**
   - Open `app.py`
   - Line 17: Replace with your API key or set as environment variable

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

### Deploy to Streamlit Community Cloud

1. **Push code to GitHub** (already done if you cloned from here)

2. **Deploy on Streamlit:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select repository: `kp2524/langchain-use-cases-demo`
   - Main file: `app.py`
   - Branch: `main`

3. **Add API key as secret:**
   - In app settings, go to "Secrets"
   - Add in TOML format:
     ```toml
     GROQ_API_KEY = "your-groq-api-key-here"
     ```
   - Save and deploy

4. **Your app will be live at:**
   ```
   https://YOUR_APP_NAME.streamlit.app
   ```

## 🛠️ Technology Stack

- **LangChain** - AI agent framework
- **Groq** - Ultra-fast AI inference (free tier available)
- **Hugging Face** - Embeddings and models
- **ChromaDB** - Vector database
- **Streamlit** - Interactive UI

## 📋 Requirements

- Python 3.11+
- Groq API key (free at [console.groq.com](https://console.groq.com))
- Internet connection (for downloading embeddings models)

## 📝 File Structure

```
langchain-use-cases-demo/
├── app.py              # Main Streamlit application
├── requirements.txt   # Python dependencies
├── README.md          # This file
├── .gitignore         # Git ignore rules
└── run.sh             # Quick start script (optional)
```

## 🔑 API Key Setup

The app reads the API key from environment variable `GROQ_API_KEY`. 

**For local development:**
- Set in code (line 17 of app.py) or
- Export: `export GROQ_API_KEY="your-key"`

**For Streamlit Cloud:**
- Add as secret in TOML format (see deployment steps above)

## 🎓 Use Cases Explained

Each use case demonstrates a different LangChain capability:
- **Chatbot**: Conversation memory and context management
- **Document QA**: Vector stores and retrieval chains
- **RAG**: Knowledge base construction and querying
- **Summarization**: Text processing and LLM chains
- **Data Extraction**: Structured output parsing
- **Content Generation**: Prompt templates and context injection
- **Workflow Automation**: Multi-step agent workflows
- **Custom Tools**: Tool creation and agent integration

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Credits

Built with LangChain, Groq, and Streamlit.
