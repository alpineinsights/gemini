import os
import streamlit as st
import pandas as pd
import os
import tempfile
import uuid
from dotenv import load_dotenv
import google.generativeai as genai
import time
from utils import QuartrAPI, GCSHandler, TranscriptProcessor
import aiohttp
import asyncio
from typing import List, Dict, Tuple
import json
from company_data import COMPANY_DATA, get_company_names, get_isin_by_name
import PyPDF2
from google.cloud import storage

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Financial Insights Chat",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "file_uploads" not in st.session_state:
    st.session_state.file_uploads = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "current_company" not in st.session_state:
    st.session_state.current_company = None
if "company_data" not in st.session_state:
    st.session_state.company_data = None
if "documents_fetched" not in st.session_state:
    st.session_state.documents_fetched = False

# Get GCS bucket name from Streamlit secrets
if hasattr(st, 'secrets') and 'other_secrets' in st.secrets and 'GCS_BUCKET_NAME' in st.secrets['other_secrets']:
    GCS_BUCKET_NAME = st.secrets['other_secrets']['GCS_BUCKET_NAME']
else:
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "financial-insights-docs")

# Get Gemini API key from Streamlit secrets
if hasattr(st, 'secrets') and 'other_secrets' in st.secrets and 'GEMINI_API_KEY' in st.secrets['other_secrets']:
    GEMINI_API_KEY = st.secrets['other_secrets']['GEMINI_API_KEY']
else:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Get Quartr API key from Streamlit secrets
if hasattr(st, 'secrets') and 'other_secrets' in st.secrets and 'QUARTR_API_KEY' in st.secrets['other_secrets']:
    QUARTR_API_KEY = st.secrets['other_secrets']['QUARTR_API_KEY']
else:
    QUARTR_API_KEY = os.getenv("QUARTR_API_KEY")

# Replace the existing create_gcs_connection function with this one:
def create_gcs_connection():
    """Create GCS connection using Streamlit secrets or environment variables"""
    return GCSHandler()  # Return handler directly instead of using FilesConnection

# Initialize GCS handler
gcs_handler = create_gcs_connection()

# Function to extract text from PDFs
def extract_pdf_text(pdf_path: str) -> str:
    """Reads the PDF from pdf_path and returns its full text."""
    full_text = []
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
    return "\n".join(full_text)

# Load company data from the pre-defined list
@st.cache_data
def load_company_data():
    """Load pre-defined company data"""
    return pd.DataFrame(COMPANY_DATA)

# Initialize Gemini model
def initialize_gemini():
    """Initialize Gemini model with API key from secrets or environment variables"""
    if hasattr(st, 'secrets') and 'other_secrets' in st.secrets and 'GEMINI_API_KEY' in st.secrets['other_secrets']:
        api_key = st.secrets['other_secrets']['GEMINI_API_KEY']
    elif os.getenv("GEMINI_API_KEY"):
        api_key = os.getenv("GEMINI_API_KEY")
    else:
        st.error("Gemini API key not found in environment variables or secrets")
        return None
    
    try:
        # Configure the Gemini API with your API key
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error initializing Gemini: {str(e)}")
        return None

# Function to process company documents
async def process_company_documents(isin: str) -> List[Dict]:
    """Process company documents and return list of file information"""
    try:
        async with aiohttp.ClientSession() as session:
            # Initialize API and handlers
            quartr_api = QuartrAPI(api_key=QUARTR_API_KEY)
            gcs_handler = GCSHandler()
            transcript_processor = TranscriptProcessor()
            
            # Get company data from Quartr API
            company_data = await quartr_api.get_company_events(isin, session)
            if not company_data:
                st.error(f"Could not find company with ISIN {isin} in Quartr database.")
                return []
            
            company_name = company_data.get('displayName', 'Unknown Company')
            events = company_data.get('events', [])
            
            if not events:
                st.warning(f"No events found for {company_name}.")
                return []
            
            # Sort events by date (descending) and take the 6 most recent
            events.sort(key=lambda x: x.get('eventDate', ''), reverse=True)
            recent_events = events[:6]
            
            processed_files = []
            
            for event in recent_events:
                event_date = event.get('eventDate', '').split('T')[0]
                event_title = event.get('eventTitle', 'Unknown Event')
                
                # Check for different document types
                doc_types = [
                    ('transcript', event.get('transcriptUrl')),
                    ('pdf', event.get('pdfUrl')),
                    ('report', event.get('reportUrl'))
                ]
                
                for doc_type, url in doc_types:
                    if not url:
                        continue
                    
                    try:
                        if doc_type == 'transcript':
                            # Process transcript differently
                            transcripts = event.get('transcripts', {})
                            if not transcripts:
                                # If the transcripts object is empty, check for liveTranscripts
                                transcripts = event.get('liveTranscripts', {})
                            
                            transcript_text = await transcript_processor.process_transcript(
                                url, transcripts, session
                            )
                            
                            if transcript_text:
                                pdf_data = transcript_processor.create_pdf(
                                    company_name, event_title, event_date, transcript_text
                                )
                                
                                filename = gcs_handler.create_filename(
                                    company_name, event_date, event_title, 'transcript', 'transcript.pdf'
                                )
                                
                                # Assuming GCSHandler has a similar upload_file method
                                success = await gcs_handler.upload_file(
                                    pdf_data, filename, GCS_BUCKET_NAME, 'application/pdf'
                                )
                                
                                if success:
                                    processed_files.append({
                                        'filename': filename,
                                        'type': 'transcript',
                                        'event_date': event_date,
                                        'event_title': event_title,
                                        'gcs_url': f"gs://{GCS_BUCKET_NAME}/{filename}"
                                    })
                        else:
                            # Process other document types
                            async with session.get(url) as response:
                                if response.status == 200:
                                    content = await response.read()
                                    original_filename = url.split('/')[-1]
                                    
                                    filename = gcs_handler.create_filename(
                                        company_name, event_date, event_title, doc_type, original_filename
                                    )
                                    
                                    success = await gcs_handler.upload_file(
                                        content, filename, GCS_BUCKET_NAME, 
                                        response.headers.get('content-type', 'application/pdf')
                                    )
                                    
                                    if success:
                                        processed_files.append({
                                            'filename': filename,
                                            'type': doc_type,
                                            'event_date': event_date,
                                            'event_title': event_title,
                                            'gcs_url': f"gs://{GCS_BUCKET_NAME}/{filename}"
                                        })
                    except Exception as e:
                        st.error(f"Error processing {doc_type} for {event_title}: {str(e)}")
            
            return processed_files
    except Exception as e:
        st.error(f"Error processing company documents: {str(e)}")
        return []

# Update the download_files_from_gcs function to use gcs_handler directly:
def download_files_from_gcs(file_infos: List[Dict]) -> List[str]:
    """Download files from GCS to temporary location and return local paths"""
    
    temp_dir = tempfile.mkdtemp()
    local_files = []
    
    for file_info in file_infos:
        try:
            gcs_path = file_info['gcs_url'].replace('gs://', '')
            bucket_name, blob_name = gcs_path.split('/', 1)
            
            local_path = os.path.join(temp_dir, file_info['filename'])
            
            # Download the file using GCSHandler directly
            bucket = gcs_handler.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)
                
            local_files.append(local_path)
        except Exception as e:
            st.error(f"Error downloading file from GCS: {str(e)}")
    
    return local_files

# Function to query Gemini with file context
def query_gemini(query: str, file_paths: List[str]) -> str:
    """Query Gemini model with context from extracted PDF text"""
    try:
        # Make sure Gemini is initialized
        if not initialize_gemini():
            return "Error initializing Gemini client"
        
        # Extract text from PDFs
        context_snippets = []
        for file_path in file_paths:
            try:
                pdf_text = extract_pdf_text(file_path)
                if pdf_text.strip():
                    context_snippets.append(pdf_text)
            except Exception as e:
                st.error(f"Error extracting text from PDF: {str(e)}")
        
        if not context_snippets:
            return "No text was extracted from the provided files"
        
        # Create a model instance
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "temperature": 0.3,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )
        
        # Create the prompt with context
        prompt = f"You are a senior financial analyst. Review the attached documents and provide a detailed and structured answer to the user's query. User's query: '{query}'"
        
        # Generate content with context snippets
        response = model.generate_content(
            [prompt, *context_snippets]
        )
        
        return response.text
    except Exception as e:
        st.error(f"Error querying Gemini: {str(e)}")
        return f"An error occurred while processing your query: {str(e)}"

# Add this after initializing your GCSHandler
def debug_google_auth():
    try:
        gcs_handler = GCSHandler()
        st.sidebar.success("✅ GCS Authentication successful")
        # Test bucket access
        buckets = list(gcs_handler.storage_client.list_buckets())
        st.sidebar.write(f"Found {len(buckets)} buckets")
        
        # Check which bucket we're using
        bucket_name = st.secrets['other_secrets']['GCS_BUCKET_NAME'] if hasattr(st, 'secrets') else os.getenv("GCS_BUCKET_NAME")
        st.sidebar.write(f"Target bucket: {bucket_name}")
        
        # Test if the bucket exists
        bucket = gcs_handler.storage_client.bucket(bucket_name)
        if bucket.exists():
            st.sidebar.success(f"✅ Bucket '{bucket_name}' exists")
        else:
            st.sidebar.error(f"❌ Bucket '{bucket_name}' does not exist")
    except Exception as e:
        st.sidebar.error(f"❌ GCS Authentication error: {str(e)}")

# Main UI components
def main():
    st.title("Financial Insights Chat")
    
    # Load company data
    company_data = load_company_data()
    if company_data is None:
        st.error("Failed to load company data. Please check the company data module.")
        return
    
    # Sidebar with company selection
    with st.sidebar:
        st.header("Select Company")
        company_names = get_company_names()
        selected_company = st.selectbox(
            "Choose a company:",
            options=company_names,
            index=0 if company_names else None
        )
        
        if selected_company:
            isin = get_isin_by_name(selected_company)
            
            # Check if company changed
            if st.session_state.current_company != selected_company:
                st.session_state.current_company = selected_company
                st.session_state.company_data = {
                    'name': selected_company,
                    'isin': isin
                }
                
                # Clear previous conversation when company changes
                st.session_state.chat_history = []
                st.session_state.processed_files = []
                st.session_state.documents_fetched = False
    
    # Main chat area
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if query := st.chat_input("Ask about the company..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # Generate response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("Thinking...")
            
            # Check if we have a selected company
            if not st.session_state.company_data:
                response = "Please select a company from the sidebar first."
                response_placeholder.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                return
            
            # Fetch documents if not already fetched
            if not st.session_state.documents_fetched:
                with st.spinner(f"Fetching documents for {st.session_state.company_data['name']}..."):
                    isin = st.session_state.company_data['isin']
                    processed_files = asyncio.run(process_company_documents(isin))
                    st.session_state.processed_files = processed_files
                    st.session_state.documents_fetched = True
                    
                    if not processed_files:
                        response = "No documents found for this company. Please try another company or check your Quartr API key."
                        response_placeholder.markdown(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        return
            
            # Process the user query with the fetched documents
            if st.session_state.processed_files:
                with st.spinner("Processing your query with Gemini..."):
                    # Download files from GCS
                    local_files = download_files_from_gcs(st.session_state.processed_files)
                    
                    if not local_files:
                        response = "Error downloading files from GCS. Please check your Google Cloud credentials."
                        response_placeholder.markdown(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        return
                    
                    # Query Gemini with file context
                    response = query_gemini(query, local_files)
                    
                    # Add used sources
                    sources = "\n\n**Sources:**\n" + "\n".join([
                        f"- {file['event_date']} - {file['event_title']} ({file['type']})"
                        for file in st.session_state.processed_files
                    ])
                    
                    full_response = response + sources
                    response_placeholder.markdown(full_response)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
            else:
                response = "No documents are available for this company. Please try another company."
                response_placeholder.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Call this in your main function
    debug_google_auth()

if __name__ == "__main__":
    main()