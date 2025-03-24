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
import logging
import traceback

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
    start_time = time.time()
    logger.info(f"Starting document processing for ISIN: {isin}")
    
    try:
        async with aiohttp.ClientSession() as session:
            # Initialize API and handlers
            logger.info("Initializing API and handlers")
            quartr_api = QuartrAPI(api_key=QUARTR_API_KEY)
            gcs_handler = GCSHandler()
            transcript_processor = TranscriptProcessor()
            
            # Get company data from Quartr API
            logger.info(f"Requesting company data for ISIN: {isin}")
            api_start_time = time.time()
            company_data = await quartr_api.get_company_events(isin, session)
            api_time = time.time() - api_start_time
            logger.info(f"Company data request completed in {api_time:.2f}s")
            
            if not company_data:
                logger.warning(f"No company data returned for ISIN: {isin}")
                st.error(f"""
                Company with ISIN {isin} not found in Quartr database.
                
                This could be because:
                - The company might not be covered by Quartr
                - The ISIN might be incorrect
                
                Try searching for a well-known large company (e.g., Apple - US0378331005, Microsoft - US5949181045).
                """)
                return []
            
            company_name = company_data.get('displayName', 'Unknown Company')
            events = company_data.get('events', [])
            logger.info(f"Processing for company: {company_name}, found {len(events)} events")
            
            if not events:
                logger.warning(f"No events found for {company_name}")
                st.warning(f"No events found for {company_name}.")
                return []
            
            # Sort events by date (descending) and take the most recent events first
            events.sort(key=lambda x: x.get('eventDate', ''), reverse=True)
            logger.info(f"Sorted {len(events)} events by date (most recent first)")
            
            processed_files = []
            transcript_count = 0
            report_count = 0
            slides_count = 0
            
            logger.info("Starting to process individual event documents")
            
            # Only process up to 6 documents in total (2 of each type)
            for i, event in enumerate(events):
                # Stop processing if we have enough documents (2 of each type)
                if transcript_count >= 2 and report_count >= 2 and slides_count >= 2:
                    logger.info("Reached target document counts (2 of each type), stopping further processing")
                    break
                    
                event_date = event.get('eventDate', '').split('T')[0]
                event_title = event.get('eventTitle', 'Unknown Event')
                logger.info(f"Processing event {i+1}/{len(events)}: {event_date} - {event_title}")
                
                # Only process the document types we need
                if transcript_count < 2 and event.get('transcriptUrl'):
                    # Process transcript
                    transcript_url = event.get('transcriptUrl')
                    logger.info(f"Processing transcript: {transcript_url}")
                    try:
                        transcript_start = time.time()
                        transcripts = event.get('transcripts', {})
                        if not transcripts:
                            # If the transcripts object is empty, check for liveTranscripts
                            transcripts = event.get('liveTranscripts', {})
                            logger.info("Using liveTranscripts instead of transcripts")
                        
                        logger.info("Calling transcript processor")
                        transcript_text = await transcript_processor.process_transcript(
                            transcript_url, transcripts, session
                        )
                        
                        if transcript_text:
                            logger.info(f"Successfully processed transcript text ({len(transcript_text)} chars)")
                            logger.info("Creating PDF from transcript text")
                            pdf_data = transcript_processor.create_pdf(
                                company_name, event_title, event_date, transcript_text
                            )
                            
                            filename = gcs_handler.create_filename(
                                company_name, event_date, event_title, 'transcript', 'transcript.pdf'
                            )
                            logger.info(f"Generated filename: {filename}")
                            
                            logger.info(f"Uploading transcript PDF to GCS bucket: {GCS_BUCKET_NAME}")
                            upload_start = time.time()
                            success = await gcs_handler.upload_file(
                                pdf_data, filename, GCS_BUCKET_NAME, 'application/pdf'
                            )
                            upload_time = time.time() - upload_start
                            
                            if success:
                                logger.info(f"Successfully uploaded transcript to GCS in {upload_time:.2f}s")
                                processed_files.append({
                                    'filename': filename,
                                    'type': 'transcript',
                                    'event_date': event_date,
                                    'event_title': event_title,
                                    'gcs_url': f"gs://{GCS_BUCKET_NAME}/{filename}"
                                })
                                transcript_count += 1
                                logger.info(f"Transcript count now: {transcript_count}/2")
                            else:
                                logger.error(f"Failed to upload transcript to GCS after {upload_time:.2f}s")
                        else:
                            logger.warning("No transcript text was extracted")
                            
                        transcript_time = time.time() - transcript_start
                        logger.info(f"Transcript processing completed in {transcript_time:.2f}s")
                    except Exception as e:
                        logger.error(f"Error processing transcript for {event_title}: {str(e)}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        st.error(f"Error processing transcript for {event_title}: {str(e)}")
                
                # Process report (if we need more)
                if report_count < 2 and event.get('reportUrl'):
                    report_url = event.get('reportUrl')
                    logger.info(f"Processing report: {report_url}")
                    try:
                        report_start = time.time()
                        async with session.get(report_url) as response:
                            response_time = time.time() - report_start
                            logger.info(f"Got report response in {response_time:.2f}s with status: {response.status}")
                            
                            if response.status == 200:
                                content = await response.read()
                                content_size = len(content) / 1024  # KB
                                logger.info(f"Downloaded report content: {content_size:.2f} KB")
                                
                                original_filename = report_url.split('/')[-1]
                                logger.info(f"Original report filename: {original_filename}")
                                
                                filename = gcs_handler.create_filename(
                                    company_name, event_date, event_title, 'report', original_filename
                                )
                                logger.info(f"Generated filename: {filename}")
                                
                                logger.info(f"Uploading report to GCS bucket: {GCS_BUCKET_NAME}")
                                upload_start = time.time()
                                success = await gcs_handler.upload_file(
                                    content, filename, GCS_BUCKET_NAME, 
                                    response.headers.get('content-type', 'application/pdf')
                                )
                                upload_time = time.time() - upload_start
                                
                                if success:
                                    logger.info(f"Successfully uploaded report to GCS in {upload_time:.2f}s")
                                    processed_files.append({
                                        'filename': filename,
                                        'type': 'report',
                                        'event_date': event_date,
                                        'event_title': event_title,
                                        'gcs_url': f"gs://{GCS_BUCKET_NAME}/{filename}"
                                    })
                                    report_count += 1
                                    logger.info(f"Report count now: {report_count}/2")
                                else:
                                    logger.error(f"Failed to upload report to GCS after {upload_time:.2f}s")
                            else:
                                logger.warning(f"Could not download report, status code: {response.status}")
                                
                        report_time = time.time() - report_start
                        logger.info(f"Report processing completed in {report_time:.2f}s")
                    except Exception as e:
                        logger.error(f"Error processing report for {event_title}: {str(e)}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        st.error(f"Error processing report for {event_title}: {str(e)}")
                
                # Process slides/PDF (if we need more)
                if slides_count < 2 and event.get('pdfUrl'):
                    pdf_url = event.get('pdfUrl')
                    logger.info(f"Processing slides PDF: {pdf_url}")
                    try:
                        pdf_start = time.time()
                        async with session.get(pdf_url) as response:
                            response_time = time.time() - pdf_start
                            logger.info(f"Got PDF response in {response_time:.2f}s with status: {response.status}")
                            
                            if response.status == 200:
                                content = await response.read()
                                content_size = len(content) / 1024  # KB
                                logger.info(f"Downloaded PDF content: {content_size:.2f} KB")
                                
                                original_filename = pdf_url.split('/')[-1]
                                logger.info(f"Original PDF filename: {original_filename}")
                                
                                filename = gcs_handler.create_filename(
                                    company_name, event_date, event_title, 'slides', original_filename
                                )
                                logger.info(f"Generated filename: {filename}")
                                
                                logger.info(f"Uploading slides PDF to GCS bucket: {GCS_BUCKET_NAME}")
                                upload_start = time.time()
                                success = await gcs_handler.upload_file(
                                    content, filename, GCS_BUCKET_NAME, 
                                    response.headers.get('content-type', 'application/pdf')
                                )
                                upload_time = time.time() - upload_start
                                
                                if success:
                                    logger.info(f"Successfully uploaded slides PDF to GCS in {upload_time:.2f}s")
                                    processed_files.append({
                                        'filename': filename,
                                        'type': 'slides',
                                        'event_date': event_date,
                                        'event_title': event_title,
                                        'gcs_url': f"gs://{GCS_BUCKET_NAME}/{filename}"
                                    })
                                    slides_count += 1
                                    logger.info(f"Slides count now: {slides_count}/2")
                                else:
                                    logger.error(f"Failed to upload slides PDF to GCS after {upload_time:.2f}s")
                            else:
                                logger.warning(f"Could not download PDF, status code: {response.status}")
                                
                        pdf_time = time.time() - pdf_start
                        logger.info(f"Slides PDF processing completed in {pdf_time:.2f}s")
                    except Exception as e:
                        logger.error(f"Error processing slides for {event_title}: {str(e)}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        st.error(f"Error processing slides for {event_title}: {str(e)}")
            
            total_time = time.time() - start_time
            logger.info(f"Document processing completed in {total_time:.2f}s. Processed {len(processed_files)} files.")
            logger.info(f"Final counts - Transcripts: {transcript_count}, Reports: {report_count}, Slides: {slides_count}")
            return processed_files
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Error in document processing after {total_time:.2f}s: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        st.error(f"Error processing company documents: {str(e)}")
        return []

# Function to download files from GCS to temporary location
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
    """Query Gemini model with context from files"""
    try:
        # Make sure Gemini is initialized
        if not initialize_gemini():
            return "Error initializing Gemini client"
        
        # Upload files to Gemini
        files = []
        for file_path in file_paths:
            try:
                file = genai.upload_file(file_path)
                files.append(file)
            except Exception as e:
                st.error(f"Error uploading file to Gemini: {str(e)}")
        
        if not files:
            return "No files were successfully uploaded to Gemini"
        
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
        
        # Generate content with files as context
        response = model.generate_content(
            [prompt, *files]
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