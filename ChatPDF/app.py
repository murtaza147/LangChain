import streamlit as st
from dotenv import load_dotenv
import pypdfium2 as pdfium
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.llms.huggingface_hub import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template

def get_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = pdfium.PdfDocument(pdf)
        for i in range(len(pdf_reader)):
            page = pdf_reader.get_page(i)
            textpage = page.get_textpage()
            text += textpage.get_text_range() + "\n"
    return text

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history',
        return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory)
    return conversation_chain

def clear_prompt(ss):
    ss.user_question = ss.prompt_bar
    ss.prompt_bar = ""

def handle_user_input(question):
    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat-PDF", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""
    if "docs_processed" not in st.session_state:
        st.session_state.docs_processed = False
    if "prompt_bar" not in st.session_state:
        st.session_state.prompt_bar = ""

    st.header("Chat-PDF :books:")
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload PDFs here and click \"Process\"", accept_multiple_files=True, type="pdf")
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing"):
                raw_text = get_text(pdf_docs) # get pdf text
                text_chunks = get_chunks(raw_text) # get text chunks
                vector_store = get_vector_store(text_chunks) # create vector store
                st.session_state.conversation = get_conversation_chain(vector_store) # create conversation chain
                st.session_state.docs_processed = True

        if st.session_state.docs_processed:
            st.text("Documents processed")
    
    st.text_input("Type your question here:", key="prompt_bar", on_change=clear_prompt(st.session_state))

    if st.session_state.user_question:
        handle_user_input(st.session_state.user_question)

    if st.session_state.docs_processed:
        if st.session_state.chat_history:
            if st.button("Forget conversation"):  # adding a button
                st.session_state.chat_history.clear()  # clears the ConversationBufferMemory
                


if __name__ == '__main__':
    main()