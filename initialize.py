import models
from agent import AgentConfig
from gpt4free import get_g4f_chat

def initialize():
    #chat_llm = get_g4f_chat(model_name="gpt-4o-mini", temperature=0) 
    chat_llm = models.get_ollama_chat(model_name="llama3.2:1b", temperature=0)
    utility_llm = chat_llm
    embedding_llm = models.get_huggingface_embedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    config = AgentConfig(
        chat_model=chat_llm,
        utility_model=utility_llm,
        embeddings_model=embedding_llm,
        prompts_subdir = "custom",
        knowledge_subdirs=["default", "custom"],
        auto_memory_count=0,
        rate_limit_requests=30,
        max_tool_response_length=3000,
        code_exec_docker_enabled=True,
        code_exec_docker_name = "agent-zero-exe",
        code_exec_docker_image = "frdel/agent-zero-exe:latest",
        code_exec_docker_ports = { "22/tcp": 50022 }
        code_exec_docker_volumes = { 
            files.get_abs_path("work_dir"): {"bind": "/root", "mode": "rw"},
            files.get_abs_path("instruments"): {"bind": "/instruments", "mode": "rw"},
                                     },
        code_exec_ssh_enabled = True,
        code_exec_ssh_addr = "localhost",
        code_exec_ssh_port = 50022,
        code_exec_ssh_user = "Naimul",
        code_exec_ssh_pass = "N1A2I4m4@12345",
        
    )
    return config
