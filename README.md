---
title: LangChain Use Cases Demo
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# LangChain Use Cases Demo

Interactive demonstration of 8 LangChain use cases powered by Groq AI:

1. **💬 AI-Powered Chatbots** - Context-aware conversations
2. **📄 Document Question Answering** - Ask questions about documents
3. **🔍 RAG (Retrieval-Augmented Generation)** - Knowledge base queries
4. **📝 Document Summarization** - Automatic text summarization
5. **📊 Data Extraction** - Extract structured data from text
6. **✍️ Content Generation** - Generate context-aware content
7. **⚙️ Workflow Automation** - Multi-step AI workflows
8. **🛠️ Custom AI Tools** - Specialized AI-powered tools

## 🚀 Quick Start

### Local Testing

1. **Install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Get a free Groq API key:**
   - Visit [console.groq.com](https://console.groq.com)
   - Sign up (free)
   - Create API key

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

4. **Enter your API key in the sidebar and start exploring!**

### Deploy to Hugging Face Spaces (Docker)

1. **Push this code to a GitHub repository:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin YOUR_GITHUB_REPO_URL
   git push -u origin main
   ```

2. **Create Hugging Face Space:**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Space name: `langchain-use-cases-demo` (or your choice)
   - SDK: **Docker** (select Docker, not Streamlit)
   - Visibility: Public or Private
   - Click "Create Space"

3. **Connect GitHub repository:**
   - In Space Settings → Repository
   - Click "Import from GitHub" or manually upload files
   - Make sure `Dockerfile`, `app.py`, `requirements.txt`, and `README.md` are included

4. **Add API key as secret:**
   - Go to Space Settings → Secrets
   - Add new secret:
     - Key: `GROQ_API_KEY`
     - Value: Your Groq API key
   - Save

5. **Deploy:**
   - The Space will automatically build and deploy
   - Check build logs if there are any issues
   - Your app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME`

**Note:** The Dockerfile is included in this repo and will automatically be used for deployment.

## 🛠️ Technology

- **LangChain** - AI agent framework
- **Groq** - Ultra-fast AI inference (free tier available)
- **Hugging Face** - Embeddings and models
- **ChromaDB** - Vector database
- **Streamlit** - Interactive UI

## 📝 Note

This app requires a Groq API key (free to get). The key is stored only in your session and never saved.

