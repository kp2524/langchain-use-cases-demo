import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA, ConversationChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
import os
import tempfile

# ============================================
# CONFIGURATION - Set your API key here
# ============================================
# Get API key from environment variable (set in Hugging Face Spaces secrets)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Model: Llama 3.1 8B (Faster) - already configured
GROQ_MODEL = "llama-3.1-8b-instant"
# ============================================

# Page config
st.set_page_config(
    page_title="LangChain Use Cases Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory()
if "documents" not in st.session_state:
    st.session_state.documents = []
if "rag_kb" not in st.session_state:
    st.session_state.rag_kb = None

# Initialize models
@st.cache_resource
def get_llm():
    # Check if API key is not set or is the placeholder
    if not GROQ_API_KEY or GROQ_API_KEY.strip() == "" or GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        return None
    
    try:
        return ChatGroq(
            model=GROQ_MODEL,
            temperature=0.7,
            groq_api_key=GROQ_API_KEY
        )
    except Exception as e:
        return None

@st.cache_resource
def get_embeddings():
    # This works without API key - downloads model locally
    # Add timeout and retry settings
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        # Fallback: return None and handle in the code
        st.warning(f"⚠️ Could not load embeddings model: {str(e)}. Some features may not work.")
        return None

# Use Case Descriptions
USE_CASES = {
    1: {
        "title": "💬 AI-Powered Chatbot",
        "description": "Chat with an AI that remembers conversation context",
        "instructions": "Type your message and have a conversation. The AI remembers what you said earlier in the session.",
        "icon": "💬"
    },
    2: {
        "title": "📄 Document Question Answering",
        "description": "Upload a document and ask questions about it",
        "instructions": "1. Upload a text file or paste text\n2. Ask questions about the content\n3. Get answers based on the document",
        "icon": "📄"
    },
    3: {
        "title": "🔍 Retrieval-Augmented Generation (RAG)",
        "description": "Build a knowledge base and query it with AI",
        "instructions": "1. Add multiple documents to build a knowledge base\n2. Ask questions\n3. Get answers with source citations",
        "icon": "🔍"
    },
    4: {
        "title": "📝 Document Summarization",
        "description": "Automatically summarize long documents",
        "instructions": "1. Paste or upload a long document\n2. Choose summary length\n3. Get a concise summary",
        "icon": "📝"
    },
    5: {
        "title": "📊 Data Extraction",
        "description": "Extract structured data from unstructured text",
        "instructions": "1. Paste text containing information\n2. Specify what to extract (e.g., names, dates, amounts)\n3. Get structured JSON output",
        "icon": "📊"
    },
    6: {
        "title": "✍️ Content Generation",
        "description": "Generate content based on context",
        "instructions": "1. Provide context (product info, topic, etc.)\n2. Choose content type (email, blog, social media)\n3. Get generated content",
        "icon": "✍️"
    },
    7: {
        "title": "⚙️ Workflow Automation",
        "description": "Automate multi-step workflows",
        "instructions": "Select a workflow and see how AI automates multiple steps in sequence",
        "icon": "⚙️"
    },
    8: {
        "title": "🛠️ Custom AI Tools",
        "description": "Use AI-powered tools for specific tasks",
        "instructions": "Choose a tool (calculator, code generator, text analyzer) and use it",
        "icon": "🛠️"
    }
}

# Helper function to safely call LLM
def call_llm(prompt):
    llm = get_llm()
    if llm is None:
        return "⚠️ Please set your GROQ_API_KEY in the code (app.py line 15) or as an environment variable."
    try:
        from langchain_core.messages import HumanMessage
        # ChatGroq expects HumanMessage objects
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        # ChatGroq returns a message object, extract content
        if hasattr(response, 'content'):
            return response.content
        return str(response)
    except Exception as e:
        return f"Error: {str(e)}\n\n💡 Please check your API key in the code."

# Main App
def main():
    st.title("🤖 LangChain Use Cases Demo")
    st.markdown("### Explore 8 Powerful AI Agent Capabilities")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Select Use Case")
        selected = st.radio(
            "Choose a use case:",
            options=list(USE_CASES.keys()),
            format_func=lambda x: USE_CASES[x]["title"],
            label_visibility="visible"
        )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    # Right column content first to reduce spacing
    with col2:
        # Show use case information in the same format for all use cases
        st.markdown(f"### {USE_CASES[selected]['icon']} {USE_CASES[selected]['title']}")
        st.markdown(f"*{USE_CASES[selected]['description']}*")
        st.markdown("")
        st.markdown("**How it works:**")
        st.markdown(USE_CASES[selected]["instructions"])
        
        st.markdown("---")
        st.markdown("### 💡 Real-World Applications")
        
        applications = {
            1: "• Customer support chatbots\n• Virtual assistants\n• Help desk automation",
            2: "• Legal document review\n• Research paper Q&A\n• Technical manual search",
            3: "• Enterprise knowledge bases\n• Internal company wikis\n• Customer support systems",
            4: "• News article summaries\n• Research paper abstracts\n• Meeting notes compression",
            5: "• Invoice processing\n• Form data extraction\n• Automated data entry",
            6: "• Marketing campaigns\n• Content creation\n• Email generation",
            7: "• Business process automation\n• Data pipeline orchestration\n• Multi-step workflows",
            8: "• Developer productivity tools\n• Utility applications\n• Task-specific AI tools"
        }
        
        st.info(applications[selected])
    
    with col1:
        # Get LLM instance
        llm = get_llm()
        if not llm and selected not in [7, 8]:  # Workflow and Tools work without API
            st.warning("⚠️ Please set your GROQ_API_KEY in the code (app.py line 15) or as an environment variable.")
        
        # Use Case 1: Chatbot
        if selected == 1:
            st.info("💡 This chatbot maintains conversation context throughout your session.")
            
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            # Display all chat history
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            
            # Chat input at the bottom
            user_input = st.chat_input("Type your message here...")
            
            if user_input and llm:
                # Add user message to history first
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                # Get assistant response
                try:
                    chain = ConversationChain(
                        llm=llm,
                        memory=st.session_state.chat_memory,
                        verbose=False
                    )
                    response = chain.predict(input=user_input)
                    # Add assistant response to history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.session_state.chat_history.append({"role": "assistant", "content": f"❌ {error_msg}"})
                
                # Rerun to update the display with new messages
                st.rerun()
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🗑️ Clear Conversation"):
                    st.session_state.chat_memory.clear()
                    st.session_state.chat_history = []
                    st.rerun()
        
        # Use Case 2: Document QA
        elif selected == 2:
            st.info("💡 Upload a document or paste text, then ask questions about it.")
            
            doc_input_method = st.radio("Input method:", ["Paste Text", "Upload File"])
            
            doc_text = ""
            if doc_input_method == "Upload File":
                uploaded_file = st.file_uploader(
                    "Upload a document", 
                    type=["txt", "pdf", "docx"],
                    help="Supported formats: TXT, PDF, DOCX"
                )
                if uploaded_file:
                    with st.spinner("Reading document..."):
                        try:
                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                                tmp_file.write(uploaded_file.read())
                                tmp_path = tmp_file.name
                            
                            # Extract text based on file type
                            file_extension = uploaded_file.name.split('.')[-1].lower()
                            
                            if file_extension == "pdf":
                                loader = PyPDFLoader(tmp_path)
                                documents = loader.load()
                                doc_text = "\n\n".join([doc.page_content for doc in documents])
                            elif file_extension == "docx":
                                loader = Docx2txtLoader(tmp_path)
                                documents = loader.load()
                                doc_text = "\n\n".join([doc.page_content for doc in documents])
                            else:  # txt
                                with open(tmp_path, "r", encoding="utf-8") as f:
                                    doc_text = f.read()
                            
                            # Clean up temp file
                            os.unlink(tmp_path)
                            
                        except Exception as e:
                            st.error(f"Error reading file: {str(e)}")
                            doc_text = ""
            else:
                doc_text = st.text_area(
                    "Paste your document text here:", 
                    height=200,
                    placeholder="Paste your document content here..."
                )
            
            if doc_text:
                st.success(f"✅ Document loaded ({len(doc_text)} characters)")
                
                if st.button("🔍 Process Document"):
                    with st.spinner("Processing document..."):
                        try:
                            # Get embeddings with error handling
                            embeddings = get_embeddings()
                            if embeddings is None:
                                st.error("⚠️ Could not load embeddings model. Please check your internet connection and try again.")
                            else:
                                documents = [Document(page_content=doc_text)]
                                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                                splits = text_splitter.split_documents(documents)
                                
                                vectorstore = Chroma.from_documents(
                                    documents=splits,
                                    embedding=embeddings,
                                    collection_name="doc_qa"
                                )
                                st.session_state.doc_vectorstore = vectorstore
                                st.success(f"✅ Document processed into {len(splits)} chunks")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            st.info("💡 If you see timeout errors, check your internet connection. The embeddings model needs to be downloaded first.")
                
                if "doc_vectorstore" in st.session_state and llm:
                    question = st.text_input("❓ Ask a question about the document:")
                    if question:
                        with st.spinner("🔍 Finding answer..."):
                            try:
                                qa_chain = RetrievalQA.from_chain_type(
                                    llm=llm,
                                    chain_type="stuff",
                                    retriever=st.session_state.doc_vectorstore.as_retriever(search_kwargs={"k": 3})
                                )
                                answer = qa_chain.invoke({"query": question})
                                st.success("💡 Answer:")
                                st.write(answer.get("result", "No answer found"))
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                                st.info("💡 Make sure you have set your GROQ_API_KEY in the code.")
                elif "doc_vectorstore" in st.session_state and not llm:
                    st.warning("⚠️ Please set your GROQ_API_KEY in the code (app.py line 15) to ask questions.")
        
        # Use Case 3: RAG
        elif selected == 3:
            st.info("💡 Build a knowledge base from multiple documents and query it.")
            
            kb_text = st.text_area(
                "Add text to knowledge base:", 
                height=150,
                placeholder="Add information to your knowledge base..."
            )
            
            col_add, col_clear = st.columns(2)
            with col_add:
                if st.button("➕ Add to Knowledge Base"):
                    if kb_text:
                        st.session_state.documents.append(kb_text)
                        st.success(f"✅ Added! Total documents: {len(st.session_state.documents)}")
                        st.rerun()
            
            if st.session_state.documents:
                st.write(f"**📚 Knowledge Base:** {len(st.session_state.documents)} documents")
                
                if st.button("🏗️ Build Knowledge Base"):
                    with st.spinner("Building knowledge base..."):
                        try:
                            embeddings = get_embeddings()
                            if embeddings is None:
                                st.error("⚠️ Could not load embeddings model. Please check your internet connection.")
                            else:
                                docs = [Document(page_content=text) for text in st.session_state.documents]
                                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                                splits = text_splitter.split_documents(docs)
                                
                                st.session_state.rag_kb = Chroma.from_documents(
                                    documents=splits,
                                    embedding=embeddings,
                                    collection_name="rag_kb"
                                )
                                st.success(f"✅ Knowledge base built with {len(splits)} chunks")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            st.info("💡 If you see timeout errors, check your internet connection.")
                
                with col_clear:
                    if st.button("🗑️ Clear All"):
                        st.session_state.documents = []
                        st.session_state.rag_kb = None
                        st.rerun()
            
            if st.session_state.rag_kb and llm:
                query = st.text_input("🔍 Query the knowledge base:")
                if query:
                    with st.spinner("Searching knowledge base..."):
                        try:
                            retriever = st.session_state.rag_kb.as_retriever(search_kwargs={"k": 3})
                            
                            # Use invoke() for newer LangChain versions - this returns a list of Document objects
                            docs = retriever.invoke(query)
                            
                            # Ensure docs is a list
                            if not isinstance(docs, list):
                                docs = [docs] if docs else []
                            
                            qa_chain = RetrievalQA.from_chain_type(
                                llm=llm,
                                chain_type="stuff",
                                retriever=retriever
                            )
                            answer = qa_chain.invoke({"query": query})
                            
                            st.success("💡 Answer:")
                            st.write(answer.get("result", "No answer found"))
                            
                            # Get sources separately using the vectorstore directly
                            with st.expander("📄 View Sources"):
                                if docs:
                                    for i, doc in enumerate(docs, 1):
                                        st.markdown(f"**Source {i}:**")
                                        try:
                                            # Try to get page_content from Document object
                                            if hasattr(doc, 'page_content'):
                                                content = doc.page_content
                                            elif isinstance(doc, dict):
                                                content = doc.get('page_content', str(doc))
                                            else:
                                                content = str(doc)
                                            st.write(content[:300] + "..." if len(content) > 300 else content)
                                        except Exception as e:
                                            st.write(f"Error displaying source: {str(e)}")
                                        st.markdown("---")
                                else:
                                    st.info("No sources found.")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        # Use Case 4: Summarization
        elif selected == 4:
            st.info("💡 Automatically summarize long documents into concise versions.")
            
            summary_text = st.text_area("Paste text to summarize:", height=300)
            summary_length = st.select_slider(
                "Summary Length", 
                ["Short", "Medium", "Long"], 
                value="Medium"
            )
            
            if st.button("📝 Generate Summary"):
                if summary_text:
                    with st.spinner("Generating summary..."):
                        length_prompts = {
                            "Short": "Summarize in 2-3 sentences:",
                            "Medium": "Summarize in 1-2 paragraphs:",
                            "Long": "Provide a detailed summary covering all key points:"
                        }
                        
                        prompt = f"{length_prompts[summary_length]}\n\n{summary_text}"
                        summary = call_llm(prompt)
                        
                        st.success("📄 Summary:")
                        st.write(summary)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Original", f"{len(summary_text)} chars")
                        with col2:
                            st.metric("Summary", f"{len(summary)} chars")
                else:
                    st.warning("Please enter text to summarize.")
        
        # Use Case 5: Data Extraction
        elif selected == 5:
            st.info("💡 Extract structured information from unstructured text.")
            
            extract_text = st.text_area(
                "Paste text to extract data from:", 
                height=200,
                placeholder="John Doe, email: john@example.com, phone: 555-1234, date: 2024-01-15"
            )
            extract_fields = st.text_input(
                "What to extract?", 
                value="name, email, phone, date",
                help="Comma-separated list of fields"
            )
            
            if st.button("📊 Extract Data"):
                if extract_text and extract_fields:
                    with st.spinner("Extracting data..."):
                        prompt = f"""Extract the following information from the text and format as JSON:
Fields to extract: {extract_fields}

Text:
{extract_text}

Return only valid JSON with the extracted fields. Do not include explanations."""
                        
                        result = call_llm(prompt)
                        st.success("✅ Extracted Data:")
                        st.code(result, language="json")
                else:
                    st.warning("Please enter both text and fields to extract.")
        
        # Use Case 6: Content Generation
        elif selected == 6:
            st.info("💡 Generate content based on context and requirements.")
            
            context = st.text_area(
                "Provide context:", 
                height=150,
                placeholder="Product: AI Assistant, Features: Voice commands, Smart scheduling, Price: $99"
            )
            
            col_type, col_tone = st.columns(2)
            with col_type:
                content_type = st.selectbox("Content Type", ["Email", "Blog Post", "Social Media Post", "Report"])
            with col_tone:
                tone = st.selectbox("Tone", ["Professional", "Casual", "Friendly", "Formal"])
            
            if st.button("✍️ Generate Content"):
                if context:
                    with st.spinner("Generating content..."):
                        prompt = f"""Write a {tone.lower()} {content_type.lower()} based on this context:

{context}

{content_type}:"""
                        
                        content = call_llm(prompt)
                        st.success(f"📝 Generated {content_type}:")
                        st.write(content)
                else:
                    st.warning("Please provide context for content generation.")
        
        # Use Case 7: Workflow Automation
        elif selected == 7:
            st.info("💡 See how AI automates multi-step workflows.")
            
            workflow_type = st.selectbox(
                "Select Workflow", 
                ["Email to Calendar", "Data Analysis Pipeline", "Content Publishing"]
            )
            
            if workflow_type == "Email to Calendar":
                email_content = st.text_area(
                    "Email Content:", 
                    placeholder="Meeting request: Discuss project on Dec 25, 2024 at 2 PM",
                    height=100
                )
                if st.button("▶️ Execute Workflow"):
                    with st.spinner("Executing workflow..."):
                        steps = [
                            ("📧 Step 1: Reading email...", "✅ Email processed"),
                            ("📅 Step 2: Scheduling meeting...", "✅ Meeting scheduled for Dec 25, 2024 at 2 PM"),
                            ("✉️ Step 3: Sending confirmation...", "✅ Confirmation email sent")
                        ]
                        for step, result in steps:
                            st.write(f"**{step}**")
                            st.success(result)
                            st.write("")
                        st.balloons()
            
            elif workflow_type == "Data Analysis Pipeline":
                if st.button("▶️ Execute Workflow"):
                    with st.spinner("Executing workflow..."):
                        steps = [
                            ("📊 Step 1: Gathering data...", "✅ Data collected from sources"),
                            ("🔍 Step 2: Analyzing data...", "✅ Analysis completed"),
                            ("📝 Step 3: Generating report...", "✅ Report generated and saved")
                        ]
                        for step, result in steps:
                            st.write(f"**{step}**")
                            st.success(result)
                            st.write("")
                        st.balloons()
            
            elif workflow_type == "Content Publishing":
                if st.button("▶️ Execute Workflow"):
                    with st.spinner("Executing workflow..."):
                        steps = [
                            ("✍️ Step 1: Generating content...", "✅ Content created"),
                            ("🖼️ Step 2: Adding images...", "✅ Images optimized"),
                            ("📤 Step 3: Publishing...", "✅ Published to all channels")
                        ]
                        for step, result in steps:
                            st.write(f"**{step}**")
                            st.success(result)
                            st.write("")
                        st.balloons()
        
        # Use Case 8: Custom Tools
        elif selected == 8:
            st.info("💡 Use AI-powered tools for specific tasks.")
            
            tool = st.selectbox("Select Tool", ["Calculator", "Code Generator", "Text Analyzer"])
            
            if tool == "Calculator":
                expression = st.text_input("Enter expression (e.g., 25 * 4 + 100):")
                if st.button("🔢 Calculate"):
                    if expression:
                        try:
                            result = eval(expression)
                            st.success(f"✅ Result: **{result}**")
                        except:
                            st.error("❌ Invalid expression. Please use valid Python math expressions.")
                    else:
                        st.warning("Please enter an expression.")
            
            elif tool == "Code Generator":
                description = st.text_area(
                    "Describe what code you need:", 
                    height=100,
                    placeholder="Python function to calculate factorial"
                )
                if st.button("💻 Generate Code"):
                    if description and llm:
                        with st.spinner("Generating code..."):
                            prompt = f"Generate clean, working Python code for: {description}. Return only the code, no explanations or markdown."
                            code = call_llm(prompt)
                            st.success("✅ Generated Code:")
                            st.code(code, language="python")
                    elif not description:
                        st.warning("Please describe what code you need.")
            
            elif tool == "Text Analyzer":
                text = st.text_area("Enter text to analyze:", height=150)
                if st.button("📊 Analyze"):
                    if text:
                        word_count = len(text.split())
                        char_count = len(text)
                        sentences = text.count(".") + text.count("!") + text.count("?")
                        paragraphs = text.count("\n\n") + 1
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Words", word_count)
                        with col2:
                            st.metric("Characters", char_count)
                        with col3:
                            st.metric("Sentences", sentences)
                        with col4:
                            st.metric("Paragraphs", paragraphs)
                    else:
                        st.warning("Please enter text to analyze.")

if __name__ == "__main__":
    main()

