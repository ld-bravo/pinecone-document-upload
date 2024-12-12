import os
import PyPDF2
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()


class PDFVectorSearch:
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
        index_name = 'pdf-search-index'
        indexes = pc.list_indexes()
        indexes_names = []

        for index in indexes:
            indexes_names.append(index.name)

        if index_name not in indexes_names:
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

    def query_pdf_content(self, query: str, top_k: int = 5) -> List[str]:
        """
        Query the indexed PDF content.

        :param query: Search query
        :param top_k: Number of top results to retrieve
        :return: List of most relevant text chunks
        """
        # Generate embedding for query
        query_embedding = self.generate_embeddings([query])[0]

        # Query Pinecone
        query_results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # Extract and return relevant text chunks
        return [
            match['metadata']['text']
            for match in query_results['matches']
        ]

    def answer_query(self, query: str) -> str:
        """
        Generate an answer using retrieved context and OpenAI.

        :param query: User's query
        :return: Generated answer
        """
        # Retrieve relevant context
        context_chunks = self.query_pdf_content(query)

        # Prepare prompt with context
        prompt = f"""
        Context: {' '.join(context_chunks)}

        Question: {query}

        Based on the provided context, answer the question thoroughly and precisely.
        If the answer is not in the context, state that you cannot find the information.
        """

        # Generate answer
        response = self.openai_client.chat.completions.create(
            # model="gpt-3.5-turbo",
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on given context."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

# Example usage


def main():
    # Initialize the PDF Vector Search
    pdf_search = PDFVectorSearch()

    # Upload a PDF (replace with your PDF path)
    # pdf_search.upload_to_pinecone('sample-document.pdf')

    # Query the uploaded PDF
    query = "Can you tell what are the health problems of Teresa?"
    # results = pdf_search.query_pdf_content(query)
    # print("Relevant Chunks:", results)

    # Get an AI-generated answer
    answer = pdf_search.answer_query(query)
    print("AI Answer:", answer)


if __name__ == "__main__":
    main()
