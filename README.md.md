# Financial Insights Chat App

A Streamlit application that allows users to chat with financial documents using Gemini AI and the Quartr API.

## Features

- Select a company from a dropdown list loaded from an Excel file
- Fetch company financial documents from Quartr API
- Convert transcript data to well-formatted PDFs
- Upload documents to Amazon S3
- Process user queries against the documents using Google's Gemini 2.0 Flash model
- Display AI-generated responses with source information

## Prerequisites

- Python 3.9+
- AWS account with S3 access
- Quartr API key
- Google Gemini API key
- Excel file with company information (MSCI Europe universe.xlsx)

## Setup

1. **Clone the repository**

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Create environment variables**
   
   Copy the `.env-template` file to `.env` and fill in your credentials:
   ```
   cp .env-template .env
   ```

4. **Prepare the data file**
   
   Ensure you have the "MSCI Europe universe.xlsx" file in the project root directory. The file should contain at least two columns: "Name" and "ISIN".

5. **Create an S3 bucket**
   
   Create an S3 bucket to store the documents and ensure your AWS credentials have permission to write to it.

## Running the Application

```
streamlit run app.