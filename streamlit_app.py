import streamlit as st
import tiktoken
from loguru import logger
import os
import tempfile


from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

# from streamlit_chat import message
from langchain_community.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory


def main():
    st.set_page_config(
        page_title="RAG Chat")

    st.title("mySUNI RAG Chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        model_selection = st.selectbox(
            "Choose the language model",
            ("gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4o"),
            key="model_selection"
        )
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'pptx'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")
    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key, st.session_state.model_selection)

        st.session_state.processComplete = True


    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                         "content": "안녕하세요! mysuni RAG chatbot 입니다. 주어진 문서에 대해 궁금한 점을 물어보세요."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("Message to chatbot"):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    for doc in source_documents:
                        st.markdown(doc.metadata['source'], help=doc.page_content)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


def tiktoken_len(text):
    """
    주어진 텍스트에 대한 토큰 길이를 계산합니다.

    Parameters:
    - text: str, 토큰 길이를 계산할 텍스트입니다.

    Returns:
    - int, 계산된 토큰 길이입니다.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


def load_document(doc):
    """
    업로드된 문서 파일을 로드하고, 해당 포맷에 맞는 로더를 사용하여 문서를 분할합니다.

    지원되는 파일 유형에 따라 적절한 문서 로더(PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader)를 사용하여
    문서 내용을 로드하고 분할합니다. 지원되지 않는 파일 유형은 빈 리스트를 반환합니다.

    Parameters:
    - doc (UploadedFile): Streamlit을 통해 업로드된 파일 객체입니다.

    Returns:
    - List[Document]: 로드 및 분할된 문서 객체의 리스트입니다. 지원되지 않는 파일 유형의 경우 빈 리스트를 반환합니다.
    """
    # 임시 디렉토리에 파일 저장
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, doc.name)

    # 파일 쓰기
    with open(file_path, "wb") as file:
        file.write(doc.getbuffer())  # 파일 내용을 임시 파일에 쓴다

    # 파일 유형에 따라 적절한 로더를 사용하여 문서 로드 및 분할
    try:
        if file_path.endswith('.pdf'):
            loaded_docs = PyPDFLoader(file_path).load_and_split()
        elif file_path.endswith('.docx'):
            loaded_docs = Docx2txtLoader(file_path).load_and_split()
        elif file_path.endswith('.pptx'):
            loaded_docs = UnstructuredPowerPointLoader(file_path).load_and_split()
        else:
            loaded_docs = []  # 지원되지 않는 파일 유형
    finally:
        os.remove(file_path)  # 작업 완료 후 임시 파일 삭제

    return loaded_docs

def get_text(docs):
    doc_list = []
    for doc in docs:
        doc_list.extend(load_document(doc))
    return doc_list


def get_text_chunks(text):
    """
    주어진 텍스트 목록을 특정 크기의 청크로 분할합니다.

    이 함수는 'RecursiveCharacterTextSplitter'를 사용하여 텍스트를 청크로 분할합니다. 각 청크의 크기는
    `chunk_size`에 의해 결정되며, 청크 간의 겹침은 `chunk_overlap`으로 조절됩니다. `length_function`은
    청크의 실제 길이를 계산하는 데 사용되는 함수입니다. 이 경우, `tiktoken_len` 함수가 사용되어 각 청크의
    토큰 길이를 계산합니다.

    Parameters:
    - text (List[str]): 분할할 텍스트 목록입니다.

    Returns:
    - List[str]: 분할된 텍스트 청크의 리스트입니다.

    사용 예시:
    텍스트 목록이 주어졌을 때, 이 함수를 호출하여 각 텍스트를 지정된 크기의 청크로 분할할 수 있습니다.
    이렇게 분할된 청크들은 텍스트 분석, 임베딩 생성, 또는 기계 학습 모델의 입력으로 사용될 수 있습니다.


    주의:
    `chunk_size`와 `chunk_overlap`은 분할의 세밀함과 처리할 텍스트의 양에 따라 조절할 수 있습니다.
    너무 작은 `chunk_size`는 처리할 청크의 수를 불필요하게 증가시킬 수 있고, 너무 큰 `chunk_size`는
    메모리 문제를 일으킬 수 있습니다. 적절한 값을 실험을 통해 결정하는 것이 좋습니다.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    """
    주어진 텍스트 청크 리스트로부터 벡터 저장소를 생성합니다.

    이 함수는 Hugging Face의 'jhgan/ko-sroberta-multitask' 모델을 사용하여 각 텍스트 청크의 임베딩을 계산하고,
    이 임베딩들을 FAISS 인덱스에 저장하여 벡터 검색을 위한 저장소를 생성합니다. 이 저장소는 텍스트 청크들 간의
    유사도 검색 등에 사용될 수 있습니다.

    Parameters:
    - text_chunks (List[str]): 임베딩을 생성할 텍스트 청크의 리스트입니다.

    Returns:
    - vectordb (FAISS): 생성된 임베딩들을 저장하고 있는 FAISS 벡터 저장소입니다.

    모델 설명:
    'jhgan/ko-sroberta-multitask'는 문장과 문단을 768차원의 밀집 벡터 공간으로 매핑하는 sentence-transformers 모델입니다.
    클러스터링이나 의미 검색 같은 작업에 사용될 수 있습니다. KorSTS, KorNLI 학습 데이터셋으로 멀티 태스크 학습을 진행한 후,
    KorSTS 평가 데이터셋으로 평가한 결과, Cosine Pearson 점수는 84.77, Cosine Spearman 점수는 85.60 등을 기록했습니다.
"""
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb


def get_conversation_chain(vetorestore, openai_api_key, model_selection):
    """
    대화형 검색 체인을 초기화하고 반환합니다.

    이 함수는 주어진 벡터 저장소, OpenAI API 키, 모델 선택을 기반으로 대화형 검색 체인을 생성합니다.
    이 체인은 사용자의 질문에 대한 답변을 생성하는 데 필요한 여러 컴포넌트를 통합합니다.

    Parameters:
    - vetorestore: 검색을 수행할 벡터 저장소입니다. 이는 문서 또는 데이터를 검색하는 데 사용됩니다.
    - openai_api_key (str): OpenAI API를 사용하기 위한 API 키입니다.
    - model_selection (str): 대화 생성에 사용될 언어 모델을 선택합니다. 예: 'gpt-3.5-turbo', 'gpt-4-turbo-preview'.

    Returns:
    - ConversationalRetrievalChain: 초기화된 대화형 검색 체인입니다.

    함수는 다음과 같은 작업을 수행합니다:
    1. ChatOpenAI 클래스를 사용하여 선택된 모델에 대한 언어 모델(LLM) 인스턴스를 생성합니다.
    2. ConversationalRetrievalChain.from_llm 메소드를 사용하여 대화형 검색 체인을 구성합니다. 이 과정에서,
       - 검색(retrieval) 단계에서 사용될 벡터 저장소와 검색 방식
       - 대화 이력을 관리할 메모리 컴포넌트
       - 대화 이력에서 새로운 질문을 생성하는 방법
       - 검색된 문서를 반환할지 여부 등을 설정합니다.
    3. 생성된 대화형 검색 체인을 반환합니다.
    """
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_selection, temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vetorestore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )

    return conversation_chain



if __name__ == '__main__':
    main()