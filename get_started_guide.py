import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

INDEX_NAME = "quickstart"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

print("Pinecone API key: ", PINECONE_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)


def create_pinecone_index():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,  # Replace with your model dimensions
        metric="cosine",  # Replace with your model metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

def create_vector_embeddings():
    data = [
        {"id": "vec1", "text": "Apple is a popular fruit known for its sweetness and crisp texture."},
        {"id": "vec2", "text": "The tech company Apple is known for its innovative products like the iPhone."},
        {"id": "vec3", "text": "Many people enjoy eating apples as a healthy snack."},
        {"id": "vec4", "text": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces."},
        {"id": "vec5", "text": "An apple a day keeps the doctor away, as the saying goes."},
        {"id": "vec6", "text": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership."}
    ]

    embeddings = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[d['text'] for d in data],
        parameters={"input_type": "passage", "truncate": "END"}
    )

    print(embeddings[0])

    return {
        "data": data,
        "embeddings": embeddings
    }

def upsert_data(data, embeddings):
    # Wait for the index to be ready
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)

    index = pc.Index(INDEX_NAME)

    vectors = []
    for d, e in zip(data, embeddings):
        vectors.append({
            "id": d['id'],
            "values": e['values'],
            "metadata": {'text': d['text']}
        })

    index.upsert(
        vectors=vectors,
        namespace="ns1"
    )

    print(index.describe_index_stats())


def query_vector():
    query = "Tell me about the tech company known as Apple."
    index = pc.Index(INDEX_NAME)

    embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={
            "input_type": "query"
        }
    )

    results = index.query(
        namespace="ns1",
        vector=embedding[0].values,
        top_k=3,
        include_values=False,
        include_metadata=True
    )

    print(results)

if __name__ == "__main__":
    try:
        # print("Creating Pinecone index...")
        # create_pinecone_index()
        # print("Pinecone index created.")
        # print("Creating vector embeddings...")
        # response = create_vector_embeddings()
        # print("Created vector embeddings.")
        # print("Upsert data...")
        # upsert_data(response["data"], response["embeddings"])
        # print("Upsert data done.")
        query_vector()
    except Exception as e:
        print("Error: ", e)
