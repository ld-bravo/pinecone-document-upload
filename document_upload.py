import os
import PyPDF2
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

class DocumentManager:
    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )

        # Initialize Pinecone
        pc = Pinecone(
          api_key=os.environ.get("PINECONE_API_KEY")
        )

        # Create or connect to Pinecone index
        index_name = os.environ.get("PINECONE_INDEX_NAME")
        indexes = pc.list_indexes().names()

        if index_name not in indexes:
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        self.index = pc.Index(index_name)

    def extract_pdf_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.

        :param pdf_path: Path to the PDF file
        :return: Extracted text from the PDF
        """
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
        return text

    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Split text into chunks for embedding and indexing.

        :param text: Input text to chunk
        :param chunk_size: Size of each text chunk
        :return: List of text chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(word)
            current_length += len(word) + 1

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for given texts using OpenAI.

        :param texts: List of text chunks
        :return: List of embedding vectors
        """
        embeddings = self.openai_client.embeddings.create(
            input=texts,
            model="text-embedding-ada-002"
        ).data

        return [embedding.embedding for embedding in embeddings]

    def upload_to_pinecone(self, pdf_path: str):
        """
        Upload PDF content to Pinecone vector database.

        :param pdf_path: Path to the PDF file
        """
        # Extract text from PDF
        text = self.extract_pdf_text(pdf_path)

        # Chunk the text
        text_chunks = self.chunk_text(text)

        # Generate embeddings
        embeddings = self.generate_embeddings(text_chunks)

        # Upsert vectors to Pinecone
        vectors = [
            (str(i), embedding, {"text": chunk})
            for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings))
        ]

        self.index.upsert(vectors)
        print(f"Uploaded {len(vectors)} chunks to Pinecone")

# Example usage


def main():
    # Initialize the PDF Vector Search
    document_manager = DocumentManager()

    # Upload a PDF (replace with your PDF path)
    document_manager.upload_to_pinecone('document.pdf')

if __name__ == "__main__":
    main()
