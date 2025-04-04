from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("./data/processed_recipes.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = './db'
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            # What we will be vectorizing and what wwe will be looking up
            page_content = row["Title"] + " " + row["Instructions"] + " ",
            metadata ={"Ingredients": row["Ingredients"], "Cleaned Ingredients": row["Cleaned_Ingredients"] },
            i = str(i)
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="recipes",
    persist_directory=db_location,
    embedding_function=embeddings,
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)


# Making the vector usable by the LLM
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)