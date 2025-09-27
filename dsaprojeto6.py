# Projeto 6 - RAG Baseado em Agentes de IA Para Sistema Customizado de IA Generativa

# Importa funcionalidades do sistema operacional
import os

# Importa módulo para copiar e mover arquivos e pastas
import shutil

# Importa função para carregar variáveis de ambiente de um arquivo .env
from dotenv import load_dotenv

# Importa anotações de tipo para listas, dicionários tipados e sequências
from typing import List, TypedDict, Annotated, Sequence

# Importa operações de alta ordem como add, mul, etc.
import operator

# Importa carregador de documentos PDF do LangChain Community
from langchain_community.document_loaders import PyPDFLoader

# Importa utilitário para dividir textos em fragmentos de tamanho definido
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Importa implementação de store vetorial FAISS do LangChain Community
from langchain_community.vectorstores import FAISS

# Importa embeddings da HuggingFace para gerar vetores de texto
from langchain_huggingface import HuggingFaceEmbeddings

# Importa cliente ChatGroq para geração de texto via API Groq
from langchain_groq import ChatGroq  

# Importa template de prompt de chat genérico
from langchain_core.prompts import ChatPromptTemplate

# Importa parser para converter saída em string
from langchain_core.output_parsers import StrOutputParser

# Importa runnable que passa dados sem transformação
from langchain_core.runnables import RunnablePassthrough

# Importa base de mensagens para fluxos de chat
from langchain_core.messages import BaseMessage

# Importa grafo de estado para orquestrar o fluxo do agente
from langgraph.graph import StateGraph, END

# Garante que tokenizers paralelos sejam permitidos
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'True'

# Carrega variáveis definidas em arquivo .env para o ambiente
load_dotenv()

# Verifica se a chave da API Groq está configurada; caso contrário, lança erro
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("Variável de ambiente GROQ_API_KEY não configurada.")

# Define caminho da pasta onde estão os documentos PDF
PDF_FOLDER_PATH = "documentos"

# Define caminho para salvar ou carregar o store vetorial
VECTOR_STORE_PATH = "dsavectordb"

# Modelo de Embeddings 
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Cliente Groq para geração de texto
llm = ChatGroq(api_key = os.getenv("GROQ_API_KEY"),
               model_name = "llama-3.3-70b-versatile",
               temperature = 0)

# Define função para carregar e dividir PDFs em fragmentos
def dsa_carrega_pdfs(folder_path: str) -> List[dict]:

    # Exibe a pasta de origem dos PDFs
    print(f"Carregando PDFs de: {folder_path}")

    # Inicializa lista para armazenar documentos carregados
    documents = []

    # Percorre todos os arquivos da pasta
    for filename in os.listdir(folder_path):

        # Verifica se o arquivo tem extensão .pdf (ignorando maiúsculas/minúsculas)
        if filename.lower().endswith(".pdf"):

            # Obtém o caminho completo do arquivo
            file_path = os.path.join(folder_path, filename)

            try:

                # Cria carregador de PDF para o caminho informado
                loader = PyPDFLoader(file_path)

                # Carrega as páginas do PDF como documentos
                loaded_docs = loader.load()

                # Anexa nome do arquivo aos metadados de cada documento
                for doc in loaded_docs:
                    doc.metadata['source'] = filename

                # Adiciona todos os documentos carregados à lista principal
                documents.extend(loaded_docs)

                # Exibe mensagem de sucesso para o arquivo carregado
                print(f" - {filename} carregado")
            except Exception as e:
                # Em caso de erro, exibe mensagem com detalhes
                print(f"   - Erro ao carregar {filename}: {e}")

    # Se nenhum documento foi carregado, informa usuário e encerra função
    if not documents:
        print("Nenhum documento PDF encontrado ou carregado.")
        return []

    # Informa quantas páginas serão divididas em fragmentos
    print(f"\nDividindo {len(documents)} páginas de documentos em fragmentos...")

    # Cria instância do divididor de texto com tamanho de fragmento e sobreposição
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 200)

    # Divide os documentos em fragmentos de texto
    split_docs = text_splitter.split_documents(documents)

    # Exibe quantos fragmentos de texto foram gerados
    print(f"Criados {len(split_docs)} fragmentos de texto.")

    # Retorna lista de fragmentos gerados
    return split_docs

# Define função para criar ou carregar o store vetorial a partir de documentos
def dsa_cria_carrega_vectordb(documents: List[dict], embeddings, store_path: str) -> FAISS:

    # Verifica se já existe um diretório para o store vetorial
    if os.path.exists(store_path):

        # Informa que o store existente será carregado
        print(f"\nCarregando store vetorial existente de: {store_path}")

        # Carrega o store vetorial localmente, permitindo desserialização
        vector_store = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)

        # Informa que o store vetorial foi carregado
        print("Store vetorial carregado.")

    else:

        # Se não houver documentos para criar o store, lança um erro
        if not documents:
            raise ValueError("Nenhum documento fornecido para criar um novo store vetorial.")

        # Informa que um novo store será criado no caminho especificado
        print(f"\nCriando novo store vetorial em: {store_path}")

        # Cria o store vetorial a partir dos documentos e embeddings
        vector_store = FAISS.from_documents(documents, embeddings)

        # Salva o store vetorial no disco
        vector_store.save_local(store_path)

        # Informa que o store vetorial foi criado e salvo
        print("Store vetorial criado e salvo.")

    # Retorna a instância do store vetorial
    return vector_store

# Define função para formatar documentos em uma única string
def dsa_formata_docs_metadados(docs: List[dict]) -> str:

    # Une fragmentos de texto separados por linha de divisão entre eles
    return "\n\n---\n\n".join(
        
        # Para cada documento na lista, cria uma string com fonte, página e conteúdo
        f"Fonte: {doc.metadata.get('source', 'Desconhecida')} (Página: {doc.metadata.get('page', 'N/D')})\n\n{doc.page_content}"
        
        # Itera sobre todos os documentos fornecidos
        for doc in docs
    )

# Template de prompt
RAG_PROMPT_TEMPLATE = """Você é um assistente de IA especializado em analisar contratos legais.
Use o contexto recuperado dos documentos de contrato abaixo para responder à pergunta.
Se você não souber a resposta com base no contexto, entregue a melhor resposta possível com base no seu conhecimento.
Mantenha a resposta concisa e responda diretamente à pergunta com base no contexto fornecido.
Cite o(s) documento(s) fonte, se possível, com base nos metadados.

CONTEXTO:
{context}

PERGUNTA:
{question}

RESPOSTA:
"""

# Cria o prompt template
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# Cria a classe para o estado do Agente
class AgentState(TypedDict):
    question: str
    documents: Sequence[dict]
    context: str
    answer: str

# Define função para recuperar documentos com base no estado atual do agente
def dsa_recupera_documentos(state: AgentState) -> AgentState:
    
    # Exibe no console que o nó de recuperação de documentos foi iniciado
    print("--- Nó: Recuperando Documentos ---")
    
    # Extrai a pergunta armazenada no estado
    question = state["question"]
    
    # Cria um retriever usando o vetor store, solicitando até 5 resultados
    retriever = vector_store.as_retriever(search_kwargs={'k': 5})
    
    # Exibe no console qual pergunta está sendo processada
    print(f"Recuperando para a pergunta: {question}")
    
    # Executa a busca e obtém os documentos relevantes
    documents = retriever.invoke(question)
    
    # Exibe quantos documentos foram recuperados
    print(f"{len(documents)} documentos recuperados.")
    
    # Retorna novo estado contendo os documentos recuperados e a pergunta original
    return {"documents": documents, "question": question}

# Define função para formatar o contexto a partir do estado do agente
def dsa_formata_contexto(state: AgentState) -> AgentState:
    
    # Indica o início do nó de formatação de contexto
    print("--- Nó: Formatando Contexto ---")
    
    # Recupera a lista de documentos do estado atual
    documents = state["documents"]
    
    # Converte os documentos em uma única string formatada
    context = dsa_formata_docs_metadados(documents)
    
    # Retorna novo estado contendo apenas o contexto formatado
    return {"context": context}

# Define função para gerar a resposta usando RAG
def dsa_gera_resposta(state: AgentState) -> AgentState:

    # Indica o início do nó de geração de resposta
    print("--- Nó: Gerando Resposta ---")
    
    # Extrai a pergunta do estado
    question = state["question"]
    
    # Extrai o contexto formatado do estado
    context = state["context"]
    
    # Constrói a cadeia RAG: mapeia contexto e pergunta, aplica prompt, modelo e parser de saída
    rag_chain = (
        {"context": lambda x: x['context'], "question": lambda x: x['question']}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    # Informa que a cadeia RAG está sendo invocada
    print("Invocando cadeia RAG...")
    
    # Executa a cadeia com o contexto e a pergunta, gerando a resposta
    answer = rag_chain.invoke({"context": context, "question": question})
    
    # Confirma que a resposta foi gerada
    print("Resposta gerada.")
    
    # Retorna o novo estado contendo apenas a resposta
    return {"answer": answer}

# Cria o workflow com LangGraph
print("\nConstruindo o grafo de agente...")
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", dsa_recupera_documentos)
workflow.add_node("format_context", dsa_formata_contexto)
workflow.add_node("generate", dsa_gera_resposta)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "format_context")
workflow.add_edge("format_context", "generate")
workflow.add_edge("generate", END)
agent_app = workflow.compile()
print("Grafo de agente compilado.")

# Verifica se o módulo está sendo executado como programa principal
if __name__ == "__main__":

    # Exibe título de inicialização do agente
    print("\n--- Inicializando o Agente de Contratos ---")
    
    # Verifica se a pasta de PDFs não existe ou está vazia
    if not os.path.exists(PDF_FOLDER_PATH) or not os.listdir(PDF_FOLDER_PATH):
        
        # Exibe mensagem de erro indicando pasta ausente ou vazia
        print(f"\nErro: a pasta de PDFs '{PDF_FOLDER_PATH}' está ausente ou vazia.")
        
        # Solicita ao usuário que crie a pasta e adicione arquivos PDF
        print("Por favor, crie a pasta e adicione seus arquivos PDF de contrato.")
        
        # Encerra o programa devido à falta de documentos
        exit()

    # Inicializa lista para documentos que alimentarão o vetor de busca
    docs_for_store = []
    
    # Se o vetor store não existe, carrega e divide os PDFs
    if not os.path.exists(VECTOR_STORE_PATH):
        
        # Carrega e fragmenta os PDFs encontrados
        docs_for_store = dsa_carrega_pdfs(PDF_FOLDER_PATH)
        
        # Se nenhum fragmento foi gerado, exibe mensagem e encerra
        if not docs_for_store:
            print("\nSaindo: Nenhum documento foi processado para criar o store vetorial.")
            exit()
    else:
        
        # Se o vetor store já existe, informa que irá reutilizá‑lo
        print(f"\nStore vetorial encontrado em '{VECTOR_STORE_PATH}'. Pulando carregamento/divisão de PDFs.")
        
        # Informa como atualizar o store caso os contratos tenham mudado
        print("Se os contratos foram alterados, exclua a pasta do store vetorial e execute novamente.")

    # Tenta criar ou carregar o store vetorial com os documentos e embeddings
    try:
        
        # Cria ou carrega o vetor store a partir dos fragmentos carregados
        vector_store = dsa_cria_carrega_vectordb(docs_for_store, embeddings_model, VECTOR_STORE_PATH)
    
    except ValueError as e:
        
        # Exibe erro caso falhe ao inicializar o vetor store e encerra
        print(f"\nErro ao inicializar o store vetorial: {e}")
        exit()

    # Informa que o agente está prestes a entrar em loop de interação
    print("\n--- Executando Agente de Contratos ---")
    
    # Inicia loop para interação contínua com o usuário
    while True:
        
        # Lê pergunta do usuário ou comando de saída
        user_query = input("\nDigite sua pergunta sobre os contratos (ou digite 'sair' para encerrar): \n> ")
        
        # Se o usuário digitar 'sair', interrompe o loop
        if user_query.lower() == 'sair':
            break
        
        # Se a entrada for vazia, pula para a próxima iteração
        if not user_query:
            continue

        # Informa que o processamento da pergunta começou
        print("\nProcessando consulta...")
        
        # Prepara dicionário de entrada para o grafo de agentes
        inputs = {"question": user_query}
        
        # Executa o grafo de agentes e obtém o estado final com resposta
        final_state = agent_app.invoke(inputs)

        # Exibe seção de resposta final
        print("\n--- Resposta Final ---")
        
        # Imprime a resposta gerada ou mensagem padrão se não houver resposta
        print(final_state.get("answer", "Nenhuma resposta gerada."))
        
        # Exibe seção com as fontes dos documentos utilizados no contexto
        print("\n--- Fontes dos Documentos Recuperados (para contexto) ---")
        
        # Se houver documentos recuperados, lista suas fontes
        if final_state.get("documents"):
            sources = set(doc.metadata.get('source', 'Desconhecida') for doc in final_state["documents"])
            print(", ".join(sources))
        else:
            # Informa que nenhum documento foi recuperado para a pergunta
            print("Nenhum documento foi recuperado para esta consulta.")
            
        # Imprime linha divisória para clareza visual entre interações
        print("-" * 50)

    # Informa que o agente foi finalizado após sair do loop
    print("\nAgente finalizado.")

