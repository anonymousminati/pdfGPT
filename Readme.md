# PDFGPT: PDF-based Q&A with Google Gemini AI

PDFGPT is a Streamlit-based application that allows users to upload PDFs, process their content, and ask questions based on the context of the uploaded documents. The application integrates a vector database for efficient text retrieval and uses Google Gemini AI for generating answers.

---

## Workflow

1. **PDF Upload**:
   - Users upload one or more PDF files through the Streamlit interface.
   - The text is extracted from the PDFs using the `extract_text_from_pdf` function.

2. **Text Chunking and Embedding**:
   - The extracted text is split into semantically meaningful chunks using the `semantic_chunking` function.
   - Each chunk is converted into a vector embedding using the `SentenceTransformer` model.

3. **Vector Database**:
   - The embeddings and their corresponding text chunks are stored in a FAISS-based vector database for efficient similarity search.
   - The database is persistent, meaning it can be saved and loaded across sessions.

4. **Question Answering**:
   - When a user asks a question, the application retrieves the most relevant text chunks from the vector database using cosine similarity.
   - The retrieved chunks are passed as context to Google Gemini AI, which generates an answer.

5. **Response Display**:
   - The question and answer are displayed in a chat-like interface, with a history of interactions.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/PDFGPT.git
   cd PDFGPT
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK data:
   ```bash
   python -m nltk.downloader punkt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## Function Explanations

### `vector_db.py`

#### `VectorDB.__init__`
- Initializes the vector database.
- Loads the FAISS index and associated data if they exist; otherwise, creates a new index.

#### `VectorDB._load`
- Loads the FAISS index and text chunks from disk.
- If the files are missing or corrupted, initializes a new index and saves it.

#### `VectorDB._save`
- Saves the FAISS index and text chunks to disk for persistence.

#### `VectorDB.semantic_chunking`
- Splits text into semantically meaningful chunks based on sentence boundaries.
- Adds overlap between chunks to preserve context across boundaries.

#### `VectorDB.store_embeddings`
- Converts text chunks into vector embeddings using the `SentenceTransformer` model.
- Stores the embeddings and chunks in the FAISS index and saves them to disk.

#### `VectorDB.query`
- Retrieves the top `k` most relevant text chunks for a given question using cosine similarity.

---

### `pdf_processor.py`

#### `extract_text_from_pdf`
- Extracts text from a PDF file using the PyPDF2 library.
- Iterates through all pages of the PDF and concatenates the extracted text.

---

## Algorithm Steps

### PDF Processing
1. Upload PDF(s) via the Streamlit interface.
2. Extract text using `extract_text_from_pdf`.

### Text Chunking and Embedding
1. Split the extracted text into chunks using `semantic_chunking`.
2. Generate vector embeddings for each chunk using `SentenceTransformer`.

### Storing in Vector Database
1. Add the embeddings and chunks to the FAISS index.
2. Save the index and associated data to disk.

### Question Answering
1. Encode the user's question into a vector embedding.
2. Retrieve the top `k` most similar text chunks from the FAISS index.
3. Pass the retrieved chunks as context to Google Gemini AI.
4. Display the generated answer along with the question in the chat interface.

---

## Example Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Upload one or more PDF files.

3. Ask a question in the input field.

4. View the answer generated by Google Gemini AI, along with the context retrieved from the uploaded PDFs.

---

## Dependencies

- `streamlit`: For building the user interface.
- `PyPDF2`: For extracting text from PDF files.
- `sentence-transformers`: For generating vector embeddings.
- `faiss-cpu`: For efficient similarity search in the vector database.
- `nltk`: For semantic text chunking.

---

## Future Enhancements

- Add support for more advanced vector databases like Pinecone or Weaviate.
- Implement a progress bar for PDF processing.
- Display the source of retrieved text chunks (e.g., filename, page number).
- Add a "Clear Knowledge Base" button to reset the vector database.

---
