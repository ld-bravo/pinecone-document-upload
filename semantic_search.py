import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from typing import List, Dict

# LangChain Imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

class SemanticSearchManager:
    def __init__(self):
        # Retrieve API keys and configurations from environment variables
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME")

        # Validate environment variables
        if not all([
            self.openai_api_key,
            self.pinecone_api_key,
            self.index_name
        ]):
            raise ValueError("Missing required environment variables")

        # Initialize Pinecone
        self.pc = Pinecone(
            api_key=self.pinecone_api_key
        )

    def langchain_semantic_search(self, query: str) -> Dict:
        """
        Semantic search using LangChain
        """
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
        vectorstore = LangchainPinecone.from_existing_index(
            self.index_name, embeddings)

        # Perform semantic search to find most relevant documents
        similar_docs = vectorstore.similarity_search(query, k=3)

        # Prepare context from similar documents
        context = "\n\n".join([doc.page_content for doc in similar_docs])

        # Initialize ChatOpenAI model
        llm = ChatOpenAI(
            api_key=self.openai_api_key,
            model="gpt-4o",
            temperature=0.3
        )

        # Construct full prompt
        full_prompt = f"""
        Given the following context documents:
        {context}

        Respond to the following query as comprehensively as possible:
        {query}

        Your response should be:
        1. Directly answer the query
        2. Utilize information from the context documents
        3. Be clear and concise
        """

        # Generate response
        response = llm.invoke(full_prompt)

        # Return results
        return {
            "query": query,
            "similar_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in similar_docs
            ],
            "response": response.content
        }

    def native_semantic_search(self, query: str) -> Dict:
        """
        Semantic search using native Pinecone and OpenAI libraries
        """
        # Initialize OpenAI client
        # openai.api_key = self.openai_api_key
        openai_client = OpenAI(
            api_key=self.openai_api_key
        )

        # Generate embedding for the query
        def get_embedding(text: str) -> List[float]:
            response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            # return response['data'][0]['embedding']
            return response.data[0].embedding

        # Connect to Pinecone index
        index = self.pc.Index(self.index_name)

        # Get query embedding
        query_embedding = get_embedding(query)

        # Perform similarity search
        results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )

        # Prepare context from similar documents
        context = "\n\n".join([
            match.get('metadata', {}).get('text', '')
            for match in results['matches']
        ])

        # Generate response using OpenAI Chat API
        # response = openai.ChatCompletion.create(
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that responds based on given context."},
                {"role": "user", "content": f"""
                Given the following context documents:
                {context}

                Respond to the following query as comprehensively as possible:
                {query}

                Your response should be:
                1. Directly answer the query
                2. Utilize information from the context documents
                3. Be clear and concise
                """}
            ],
            temperature=0.3
        )

        # Return results
        return {
            "query": query,
            "similar_documents": [
                {
                    "content": match.get('metadata', {}).get('text', ''),
                    "metadata": match.get('metadata', {}),
                    "score": match.get('score')
                } for match in results['matches']
            ],
            "response": response.choices[0].message.content
        }


# Example usage
if __name__ == "__main__":
    # Initialize the semantic search manager
    search_manager = SemanticSearchManager()

    # Example query
    query = "What is the name of the pacient?"

    # LangChain Semantic Search
    print("=== LangChain Semantic Search ===")
    langchain_result = search_manager.langchain_semantic_search(query)

    print("Query:", langchain_result['query'])
    # print("\nSimilar Documents:")
    # for doc in langchain_result['similar_documents']:
    #     print("- Content:", doc['content'])
    #     print("  Metadata:", doc['metadata'])
    #     print()

    print("\nGenerated Response:")
    print(langchain_result['response'])

    print("\n" + "="*50 + "\n")

    # Native Semantic Search
    print("=== Native Semantic Search ===")
    native_result = search_manager.native_semantic_search(query)

    print("Query:", native_result['query'])
    # print("\nSimilar Documents:")
    # for doc in native_result['similar_documents']:
    #     print("- Content:", doc['content'])
    #     print("  Metadata:", doc['metadata'])
    #     print("  Similarity Score:", doc.get('score'))
    #     print()

    print("\nGenerated Response:")
    print(native_result['response'])
