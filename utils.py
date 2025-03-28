import aiohttp
import io
import json
import logging
import os
from dotenv import load_dotenv
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from typing import Dict, Optional, List
from google.cloud import storage
import aiofiles
import tempfile
import datetime
import streamlit as st  # Make sure to import streamlit
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get environment variables for GCS
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

class QuartrAPI:
    def __init__(self, api_key=None):
        """Initialize QuartrAPI with API key from parameters, secrets, or environment variables"""
        self.base_url = "https://api.quartr.com/public/v1"
        
        logger.info("Initializing QuartrAPI client")
        
        # Get API key from parameter, secrets, or environment variable
        if api_key:
            self.api_key = api_key
            logger.info("Using provided API key")
        elif hasattr(st, 'secrets') and 'other_secrets' in st.secrets and 'QUARTR_API_KEY' in st.secrets['other_secrets']:
            self.api_key = st.secrets['other_secrets']['QUARTR_API_KEY']
            logger.info("Using API key from Streamlit secrets")
        else:
            self.api_key = os.getenv("QUARTR_API_KEY")
            logger.info("Using API key from environment variables")
            
        if not self.api_key:
            logger.error("Quartr API key not found in any source")
            raise ValueError("Quartr API key not found")
            
        # Mask the API key for logging (show only the first 4 and last 4 characters)
        masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}" if len(self.api_key) > 8 else "****"
        logger.info(f"API key configured: {masked_key}")
            
        # Update headers to use X-Api-Key as in the documentation
        self.headers = {
            "Content-Type": "application/json",
            "X-Api-Key": self.api_key
        }
        logger.info("QuartrAPI initialized successfully")

    async def get_company_events(self, isin: str, session: aiohttp.ClientSession) -> Dict:
        """Get company events from Quartr API using only ISIN"""
        start_time = time.time()
        logger.info(f"Starting company events lookup for ISIN: {isin}")
        
        try:
            # Only use direct ISIN lookup
            url = f"{self.base_url}/companies/isin/{isin}"
            logger.info(f"Making API request to: {url}")
            
            # Log the request headers (without the actual API key)
            safe_headers = {k: '****' if k == 'X-Api-Key' else v for k, v in self.headers.items()}
            logger.info(f"Request headers: {safe_headers}")
            
            async with session.get(url, headers=self.headers) as response:
                response_time = time.time() - start_time
                logger.info(f"Received response in {response_time:.2f}s with status code: {response.status}")
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error looking up company with ISIN {isin}: Status {response.status}")
                    logger.error(f"Response body: {error_text}")
                    return None
                
                company = await response.json()
                logger.info(f"Successfully retrieved company data for {isin}")
                
                # Log some company details
                company_id = company.get('id')
                company_name = company.get('name')
                logger.info(f"Company details - ID: {company_id}, Name: {company_name}")
                
                if not company_id:
                    logger.warning(f"Company found but no ID for ISIN {isin}")
                    return None
                
                # Get the company events using the company ID
                events_url = f"{self.base_url}/companies/{company_id}/events"
                logger.info(f"Requesting company events from: {events_url}")
                
                events_start_time = time.time()
                async with session.get(events_url, headers=self.headers) as response:
                    events_response_time = time.time() - events_start_time
                    logger.info(f"Received events response in {events_response_time:.2f}s with status code: {response.status}")
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Error getting events for company {company_id}: Status {response.status}")
                        logger.error(f"Response body: {error_text}")
                        return None
                    
                    events = await response.json()
                    event_count = len(events)
                    logger.info(f"Retrieved {event_count} events for company {company_name}")
                    
                    # Log information about the events
                    if event_count > 0:
                        earliest_date = min([e.get('eventDate', '9999-12-31') for e in events if e.get('eventDate')])
                        latest_date = max([e.get('eventDate', '0000-01-01') for e in events if e.get('eventDate')])
                        logger.info(f"Events date range: {earliest_date} to {latest_date}")
                        
                        # Count document types
                        transcript_count = sum(1 for e in events if e.get('transcriptUrl'))
                        pdf_count = sum(1 for e in events if e.get('pdfUrl'))
                        report_count = sum(1 for e in events if e.get('reportUrl'))
                        logger.info(f"Available documents - Transcripts: {transcript_count}, PDFs: {pdf_count}, Reports: {report_count}")
                    
                    # Combine company info with events
                    result = {
                        "id": company.get('id'),
                        "displayName": company.get('name'),
                        "ticker": company.get('ticker'),
                        "country": company.get('country'),
                        "events": events
                    }
                    
                    total_time = time.time() - start_time
                    logger.info(f"Completed company events lookup in {total_time:.2f}s")
                    return result
                    
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Exception in Quartr API after {total_time:.2f}s: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    async def get_document(self, doc_url: str, session: aiohttp.ClientSession):
        """Get document from URL"""
        try:
            async with session.get(doc_url) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.error(f"Failed to fetch document from {doc_url}: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting document from {doc_url}: {str(e)}")
            return None

class TranscriptProcessor:
    @staticmethod
    async def process_transcript(transcript_url: str, transcripts: Dict, session: aiohttp.ClientSession) -> str:
        """Process transcript JSON into clean text"""
        try:
            # First try to get the raw transcript URL from the transcripts object
            raw_transcript_url = None
            
            # Check for different transcript types in Quartr
            if 'transcriptUrl' in transcripts and transcripts['transcriptUrl']:
                raw_transcript_url = transcripts['transcriptUrl']
            elif 'finishedLiveTranscriptUrl' in transcripts.get('liveTranscripts', {}) and transcripts['liveTranscripts']['finishedLiveTranscriptUrl']:
                raw_transcript_url = transcripts['liveTranscripts']['finishedLiveTranscriptUrl']
            
            # If no raw transcript URL is found, try the app transcript URL
            if not raw_transcript_url and transcript_url and 'app.quartr.com' in transcript_url:
                # Convert app URL to API URL if possible
                document_id = transcript_url.split('/')[-2]
                if document_id.isdigit():
                    raw_transcript_url = f"https://api.quartr.com/public/v1/transcripts/document/{document_id}"
                    headers = {"X-Api-Key": self.api_key}
                    async with session.get(raw_transcript_url, headers=headers) as response:
                        if response.status == 200:
                            transcript_data = await response.json()
                            if transcript_data and 'transcript' in transcript_data:
                                text = transcript_data['transcript'].get('text', '')
                                if text:
                                    # Format the text with proper line breaks and cleanup
                                    formatted_text = TranscriptProcessor.format_transcript_text(text)
                                    logger.info(f"Successfully processed transcript from API, length: {len(formatted_text)}")
                                    return formatted_text
            
            # If we have a raw transcript URL, fetch and process it
            if raw_transcript_url:
                logger.info(f"Fetching transcript from: {raw_transcript_url}")
                
                try:
                    headers = {"X-Api-Key": self.api_key} if 'api.quartr.com' in raw_transcript_url else {}
                    async with session.get(raw_transcript_url, headers=headers) as response:
                        if response.status == 200:
                            # Try processing as JSON first
                            try:
                                transcript_data = await response.json()
                                # Handle different JSON formats
                                if 'transcript' in transcript_data:
                                    text = transcript_data['transcript'].get('text', '')
                                    if text:
                                        formatted_text = TranscriptProcessor.format_transcript_text(text)
                                        logger.info(f"Successfully processed JSON transcript, length: {len(formatted_text)}")
                                        return formatted_text
                                elif 'text' in transcript_data:
                                    formatted_text = TranscriptProcessor.format_transcript_text(transcript_data['text'])
                                    logger.info(f"Successfully processed simple JSON transcript, length: {len(formatted_text)}")
                                    return formatted_text
                            except json.JSONDecodeError:
                                # Not a JSON, try processing as text
                                text = await response.text()
                                if text:
                                    formatted_text = TranscriptProcessor.format_transcript_text(text)
                                    logger.info(f"Successfully processed text transcript, length: {len(formatted_text)}")
                                    return formatted_text
                        else:
                            logger.error(f"Failed to fetch transcript: {response.status}")
                except Exception as e:
                    logger.error(f"Error processing raw transcript: {str(e)}")
            
            logger.warning(f"No transcript found or could be processed for URL: {transcript_url}")
            return ''
        except Exception as e:
            logger.error(f"Error processing transcript: {str(e)}")
            return ''
    
    @staticmethod
    def format_transcript_text(text: str) -> str:
        """Format transcript text for better readability"""
        # Replace JSON line feed representations with actual line feeds
        text = text.replace('\\n', '\n')
        
        # Clean up extra whitespace
        text = ' '.join(text.split())
        
        # Format into paragraphs - break at sentence boundaries for better readability
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        formatted_text = '.\n\n'.join(sentences) + '.'
        
        return formatted_text

    @staticmethod
    def create_pdf(company_name: str, event_title: str, event_date: str, transcript_text: str) -> bytes:
        """Create a PDF from transcript text"""
        if not transcript_text:
            logger.error("Cannot create PDF: Empty transcript text")
            return b''
            
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        styles = getSampleStyleSheet()
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading1'],
            fontSize=14,
            spaceAfter=30,
            textColor=colors.HexColor('#1a472a'),
            alignment=1
        )
        
        text_style = ParagraphStyle(
            'CustomText',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            spaceBefore=6,
            fontName='Helvetica'
        )

        story = []
        
        # Create header with proper XML escaping
        header_text = f"""
            <para alignment="center">
            <b>{company_name.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')}</b><br/>
            <br/>
            Event: {event_title.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')}<br/>
            Date: {event_date}
            </para>
        """
        story.append(Paragraph(header_text, header_style))
        story.append(Spacer(1, 30))

        # Process transcript text
        paragraphs = transcript_text.split('\n\n')
        for para in paragraphs:
            if para.strip():
                # Clean and escape the text for PDF
                clean_para = para.strip().replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                try:
                    story.append(Paragraph(clean_para, text_style))
                    story.append(Spacer(1, 6))
                except Exception as e:
                    logger.error(f"Error adding paragraph to PDF: {str(e)}")
                    continue

        try:
            doc.build(story)
            pdf_data = buffer.getvalue()
            logger.info(f"Successfully created PDF, size: {len(pdf_data)} bytes")
            return pdf_data
        except Exception as e:
            logger.error(f"Error building PDF: {str(e)}")
            return b''

    def create_filename(self, company_name: str, event_date: str, event_title: str,
                       doc_type: str, original_filename: str) -> str:
        """Create standardized filename for GCS"""
        clean_company = company_name.replace(" ", "_").replace("/", "_").lower()
        clean_event = event_title.replace(" ", "_").lower()
        clean_date = event_date.split("T")[0]
        # Always use PDF extension for transcripts
        file_extension = "pdf" if doc_type == "transcript" else original_filename.split(".")[-1].lower()
        return f"{clean_company}_{clean_date}_{clean_event}_{doc_type}.{file_extension}"

class GCSHandler:
    def __init__(self):
        try:
            if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
                # Use service account info directly from Streamlit secrets
                service_account_info = st.secrets['gcp_service_account']
                
                # Create client with explicit credentials from secrets
                self.storage_client = storage.Client.from_service_account_info(service_account_info)
                logger.info("GCS client initialized from Streamlit secrets")
            elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                # If running locally with application credentials
                self.storage_client = storage.Client()
                logger.info("GCS client initialized from environment variable")
            else:
                raise ValueError("Google Cloud credentials not found in Streamlit secrets or environment variables")
        except Exception as e:
            logger.error(f"Error initializing GCS client: {str(e)}")
            raise

    async def upload_file(self, file_data: bytes, filename: str, bucket_name: str, 
                         content_type: str = 'application/pdf') -> bool:
        """Upload file to GCS bucket"""
        try:
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(filename)
            
            # Set the content type
            blob.content_type = content_type
            
            # Write the data to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(file_data)
            temp_file.close()
            
            # Upload from the temporary file asynchronously
            try:
                async with aiofiles.open(temp_file.name, 'rb') as f:
                    content = await f.read()
                    blob.upload_from_string(content, content_type=content_type)
                os.unlink(temp_file.name)
                logger.info(f"Successfully uploaded {filename} to GCS bucket {bucket_name}")
                return True
            except Exception as e:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                raise e
                
        except Exception as e:
            logger.error(f"Error uploading to GCS: {str(e)}")
            return False

    def create_filename(self, company_name: str, event_date: str, event_title: str,
                       doc_type: str, original_filename: str) -> str:
        """Create standardized filename for GCS"""
        clean_company = company_name.replace(" ", "_").replace("/", "_").lower()
        clean_event = event_title.replace(" ", "_").lower()
        clean_date = event_date.split("T")[0]
        # Always use PDF extension for transcripts
        file_extension = "pdf" if doc_type == "transcript" else original_filename.split(".")[-1].lower()
        return f"{clean_company}_{clean_date}_{clean_event}_{doc_type}.{file_extension}"
    
    def create_signed_url(self, bucket_name: str, blob_name: str, expiration: int = 3600) -> Optional[str]:
        """Create a signed URL for a GCS object"""
        try:
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            url = blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(seconds=expiration),
                method="GET"
            )
            return url
        except Exception as e:
            logger.error(f"Error creating signed URL: {str(e)}")
            return None
