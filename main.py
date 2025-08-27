import os
import langchain
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get("OPEN_API_KEY")
from langchain_openai import OpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain

st.title("New Research Tool")

st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(label=f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()
llm = OpenAI(api_key=api_key, temperature=0.9, max_tokens=500)
embeddings = OpenAIEmbeddings(api_key=api_key)
if process_url_clicked:
    loaders = UnstructuredURLLoader(urls=urls)
    data = loaders.load()
    main_placeholder.text("Data Loading...Started...✅✅✅")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)


    vector_index = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    vector_index.save_local("faiss_index_store")


vector_index = FAISS.load_local(
        "faiss_index_store",
        embeddings,
        allow_dangerous_deserialization=True
    )
query = main_placeholder.text_input("Question: ")
if query:
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_index.as_retriever())
    r = chain.invoke({"question": query}, return_only_outputs=True)
    st.header("Answer")
    st.subheader(r['answer'])

    sources = r.get("sources", "")
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n")  # Split the sources by newline
        for source in sources_list:
            st.write(source)




