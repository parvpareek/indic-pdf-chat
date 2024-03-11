import os
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core import (VectorStoreIndex, StorageContext, PromptTemplate, load_index_from_storage, Settings)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import FlatReader
from transformers import pipeline


def load_data():
    
    reader = FlatReader()
    docs = reader.load_data("./data")
    return docs

def load_embedding_model():
    embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5") 
    print("embedding model loaded")
    return embedding_model
    
def fetch_transfomer():
    # Use a pipeline as a high-level helper

    pipe = pipeline("translation", model="ai4bharat/indictrans2-indic-en-dist-200M", trust_remote_code=True)
    
def main():
    PERSIST_DIR = "./basic/storage"
    
    choice = input("Enter 1 to use OPEN API enter 0 to use loally setup llama2 model using Ollama: ")
    if not os.path.exists(PERSIST_DIR):
        documents = load_data()
        try:
            if choice == '1':
                print("Open API is being used")
                embedding_model = OpenAIEmbedding()
                index = VectorStoreIndex.from_documents(documents)
            else:
                print("Ollama is being used")
                embedding_model = load_embedding_model()
                Settings.embed_model = embedding_model
                
                index = VectorStoreIndex.from_documents(
                    documents,
                    embed_model=embedding_model
                    )
        except Exception as  e:
            print(e)
            exit()
        print("Documents Indexed")


    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        print("Already indexed data loaded")
        
    llama = Ollama(model="llama2", request_timeout=200.0)
    Settings.llm = llama
    query_engine = index.as_query_engine(llm=llama)
    qa_prompt_tmpl_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
            "Query: {query_str}\n"
            "Answer: "
            )

    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})
    print("Press ctr + c to exit")
    while True:
        query = input("Enter your query: ")
        response = query_engine.query(query)
        print(response)

    
if __name__ == "__main__":
    main()
