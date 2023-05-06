import streamlit as st
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from tqdm.auto import tqdm
from uuid import uuid4

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)

# Initialize embeddings
embed = OpenAIEmbeddings(document_model_name='text-embedding-ada-002', query_model_name='text-embedding-ada-002')

# Get Pinecone API key from Streamlit secrets
api_key = st.secrets["PINECONE_API_KEY"]

# Initialize Pinecone
pinecone.init(api_key=api_key)
index_name = 'langchain-retrieval-augmentation'
pinecone.create_index(name=index_name, metric='dotproduct', dimension=1536)
index = pinecone.GRPCIndex(index_name)

def main():
    # Create a file uploader for the JSON
    uploaded_file = st.file_uploader("Upload JSON File", type=['json'])

    if uploaded_file is not None:
        # Load JSON data
        data = json.load(uploaded_file)

        # Process JSON data
        batch_limit = 100
        texts = []
        metadatas = []

        for i, record in enumerate(tqdm(data)):
            # Assume record['text'] is the text field in the JSON record
            record_texts = text_splitter.split_text(record['text'])
            record_metadatas = [{'chunk': j, 'text': text} for j, text in enumerate(record_texts)]
        
            texts.extend(record_texts)
            metadatas.extend(record_metadatas)

            if len(texts) >= batch_limit:
                ids = [str(uuid4()) for _ in range(len(texts))]
                embeds = embed.embed_documents(texts)
                index.upsert(vectors=zip(ids, embeds, metadatas))
                texts = []
                metadatas = []

if __name__ == '__main__':
    main()
