Financial Insights Chat App
A Streamlit application that allows users to chat with financial documents using Gemini AI and the Quartr API. The app fetches company financial documents, processes them, and uses AI to answer user queries about the selected company.
Features

Select a company from a pre-loaded list of European companies
Fetch company financial documents from Quartr API (transcripts, reports, presentations)
Convert transcript data to well-formatted PDFs
Upload documents to Amazon S3
Process user queries against the documents using Google's Gemini 2.0 Flash model
Display AI-generated responses with source information

Architecture
The application consists of several components:

Streamlit Web Interface: Provides the user interface for company selection and chat
Company Data Module: Pre-loaded list of MSCI Europe companies with ISIN codes
Quartr API Integration: Fetches financial documents for the selected company
Document Processing: Converts and standardizes documents (especially transcripts)
S3 Storage: Stores processed documents for retrieval
Gemini AI Integration: Analyzes documents and responds to user queries

Prerequisites

Python 3.9+
AWS account with S3 access
Quartr API key
Google Gemini API key

Setup

Clone the repository
Install dependencies
Copierpip install -r requirements.txt

Create environment variables
Copy the .env-template file to .env and fill in your credentials:
Copiercp .env-template .env

Create an S3 bucket
Create an S3 bucket to store the documents and ensure your AWS credentials have permission to write to it.

Project Structure
Copierfinancial-insights-chat/
├── app.py                 # Main Streamlit application
├── company_data.py        # Pre-loaded company data
├── utils.py               # Utility classes for API, S3, and document processing
├── requirements.txt       # Python dependencies
├── .env-template          # Template for environment variables
└── README.md              # Project documentation
Running the Application
Copierstreamlit run app.py
How It Works

The user selects a company from the dropdown in the sidebar
The app fetches the latest financial documents from Quartr API
Documents are processed and uploaded to S3
When the user asks a question, the app:

Downloads the relevant documents from S3
Sends them to Gemini AI along with the user's query
Displays the AI-generated response with source information



Customizing the Company List
The list of companies is pre-loaded in company_data.py. You can modify this file to add, remove, or update companies as needed.
