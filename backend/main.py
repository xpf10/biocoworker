from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
import json
import os
import logging
from typing import Optional, List, Dict, Any

from backend.analysis import (
    generate_mock_rnaseq_data,
    run_differential_expression,
    run_pca_analysis,
    run_pathway_enrichment,
    get_heatmap_data,
    # Multi-Omics modules
    generate_mock_proteomics_data,
    get_ppi_network,
    generate_mock_metabolomics_data,
    run_plsda_analysis,
    generate_mock_genomics_data,
    run_gwas_analysis
)
from backend.agent import set_cached_data, get_agent_instance

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("biocoworker-api")

app = FastAPI(title="BioCoworker Multi-Omics Backend API")

# CORS middleware for local Electron development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persistent configuration file for multiple models
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "models_config.json")

DEFAULT_PRESETS = [
    {
        "id": "qwen-plus",
        "label": "Alibaba Qwen-Plus (通义千问)",
        "provider": "dashscope",
        "model_name": "qwen-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": ""
    },
    {
        "id": "qwen-max",
        "label": "Alibaba Qwen-Max (通义千问旗舰)",
        "provider": "dashscope",
        "model_name": "qwen-max",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": ""
    },
    {
        "id": "deepseek-chat",
        "label": "DeepSeek-V3 Chat (深度求索)",
        "provider": "deepseek",
        "model_name": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": ""
    },
    {
        "id": "ollama-qwen",
        "label": "Local Ollama Qwen2.5 (本地模型)",
        "provider": "ollama",
        "model_name": "qwen2.5:7b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama"
    },
    {
        "id": "glm-4",
        "label": "Zhipu GLM-4 (智谱 AI)",
        "provider": "zhipu",
        "model_name": "glm-4",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key": ""
    }
]

def load_models_config():
    if not os.path.exists(CONFIG_PATH):
        initial_config = {
            "models": DEFAULT_PRESETS,
            "active_model_id": "qwen-plus"
        }
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(initial_config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to initialize models config: {e}")
        return initial_config
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading models config: {e}")
        return {"models": DEFAULT_PRESETS, "active_model_id": "qwen-plus"}

def save_models_config(config_data):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving models config: {e}")

# Load configuration on start
model_config = load_models_config()

# Local cached data variables
current_omics = "transcriptomics"
counts_df: Optional[pd.DataFrame] = None
design_df: Optional[pd.DataFrame] = None
de_results_df: Optional[pd.DataFrame] = None
gwas_variants: Optional[list] = None

class ModelItem(BaseModel):
    id: str
    label: str
    provider: str
    model_name: str
    base_url: str
    api_key: str

class SelectionRequest(BaseModel):
    id: str

class ChatMessage(BaseModel):
    message: str

@app.get("/api/status")
def get_status():
    global counts_df, current_omics, gwas_variants, model_config
    is_loaded = (counts_df is not None) if current_omics != "genomics" else (gwas_variants is not None)
    
    # Find active model details
    active_id = model_config.get("active_model_id", "qwen-plus")
    active_model = next((m for m in model_config.get("models", []) if m["id"] == active_id), None)
    
    return {
        "status": "healthy",
        "dataset_loaded": is_loaded,
        "omics": current_omics,
        "num_items": len(gwas_variants) if current_omics == "genomics" and gwas_variants is not None else (len(counts_df) if counts_df is not None else 0),
        "samples": list(counts_df.columns) if current_omics != "genomics" and counts_df is not None else [],
        "config": {
            "model_name": active_model["model_name"] if active_model else "qwen-plus",
            "has_api_key": len(active_model["api_key"]) > 0 if active_model else False,
            "active_model_id": active_id
        }
    }

# =====================================================================
# MULTI-MODEL CONFIGURATIONS MANAGEMENT ENDPOINTS (QwenPaw Model Importer)
# =====================================================================

@app.get("/api/models")
def list_models():
    global model_config
    # Refresh in-memory config from file
    model_config = load_models_config()
    return model_config

@app.post("/api/models/select")
def select_model(request: SelectionRequest):
    global model_config
    model_config = load_models_config()
    models = model_config.get("models", [])
    
    if not any(m["id"] == request.id for m in models):
        raise HTTPException(status_code=400, detail=f"Model config ID '{request.id}' not found.")
        
    model_config["active_model_id"] = request.id
    save_models_config(model_config)
    logger.info(f"Switched active model configuration to ID: {request.id}")
    return {"status": "success", "active_model_id": request.id}

@app.post("/api/models/add")
def add_model(model: ModelItem):
    global model_config
    model_config = load_models_config()
    models = model_config.get("models", [])
    
    # If ID already exists, overwrite it. Otherwise append.
    existing_idx = next((i for i, m in enumerate(models) if m["id"] == model.id), -1)
    new_model_data = model.dict()
    
    if existing_idx != -1:
        models[existing_idx] = new_model_data
        logger.info(f"Overwriting model configuration ID: {model.id}")
    else:
        models.append(new_model_data)
        logger.info(f"Adding new model configuration ID: {model.id}")
        
    model_config["models"] = models
    model_config["active_model_id"] = model.id # Automatically select added model as active
    save_models_config(model_config)
    return {"status": "success", "active_model_id": model.id}

@app.post("/api/models/delete")
def delete_model(request: SelectionRequest):
    global model_config
    model_config = load_models_config()
    models = model_config.get("models", [])
    
    if len(models) <= 1:
        raise HTTPException(status_code=400, detail="Cannot delete the last remaining model configuration.")
        
    # Remove from list
    new_models = [m for m in models if m["id"] != request.id]
    if len(new_models) == len(models):
        raise HTTPException(status_code=400, detail=f"Model ID '{request.id}' not found.")
        
    model_config["models"] = new_models
    
    # If the active model was deleted, switch to the first one in the remaining list
    if model_config.get("active_model_id") == request.id:
        model_config["active_model_id"] = new_models[0]["id"]
        
    save_models_config(model_config)
    logger.info(f"Deleted model configuration ID: {request.id}. Active model is now: {model_config['active_model_id']}")
    return {"status": "success", "active_model_id": model_config["active_model_id"]}

@app.post("/api/models/test-connection")
def test_model_connection(request: SelectionRequest):
    global model_config
    model_config = load_models_config()
    models = model_config.get("models", [])
    
    target_model = next((m for m in models if m["id"] == request.id), None)
    if not target_model:
        raise HTTPException(status_code=400, detail=f"Model config ID '{request.id}' not found.")
        
    # Quick connectivity test by initializing langchain chat and sending a dummy prompt
    try:
        from langchain_openai import ChatOpenAI
        key = target_model["api_key"] or "mock-key"
        
        # Test connection
        tester = ChatOpenAI(
            model=target_model["model_name"],
            openai_api_key=key,
            openai_api_base=target_model["base_url"],
            max_tokens=5,
            timeout=8.0
        )
        
        # Invoke a quick hello test
        res = tester.invoke([("user", "say OK")])
        content = res.content.strip()
        logger.info(f"Connection test passed for model {target_model['id']}: response='{content}'")
        return {"status": "success", "message": f"Connection test passed! Model replied: '{content}'"}
    except Exception as e:
        logger.error(f"Connection test failed for model {target_model['id']}: {str(e)}")
        return {
            "status": "error", 
            "message": f"Connection failed: {str(e)}. Please check your API key, Base URL, or network access."
        }

# =====================================================================
# DATA IMPORTER & MULTI-OMICS PIPELINES
# =====================================================================

@app.post("/api/load-mock")
def load_mock_data(omics: str = "transcriptomics", num_genes: int = 1500):
    global counts_df, design_df, de_results_df, current_omics, gwas_variants
    current_omics = omics
    try:
        if omics == "transcriptomics":
            counts_df, design_df = generate_mock_rnaseq_data(num_genes)
            set_cached_data(counts_df, design_df)
            de_results_df = None
            head_counts = counts_df.head(20).reset_index().rename(columns={"index": "Gene"}).to_dict(orient="records")
            design_list = design_df.reset_index().to_dict(orient="records")
            return {
                "status": "success",
                "omics": "transcriptomics",
                "counts": head_counts,
                "design": design_list,
                "total_genes": len(counts_df),
                "samples": list(counts_df.columns)
            }
            
        elif omics == "proteomics":
            counts_df, design_df = generate_mock_proteomics_data(400)
            set_cached_data(counts_df, design_df)
            de_results_df = None
            head_counts = counts_df.head(20).reset_index().rename(columns={"index": "Gene"}).to_dict(orient="records")
            design_list = design_df.reset_index().to_dict(orient="records")
            return {
                "status": "success",
                "omics": "proteomics",
                "counts": head_counts,
                "design": design_list,
                "total_genes": len(counts_df),
                "samples": list(counts_df.columns)
            }
            
        elif omics == "metabolomics":
            counts_df, design_df = generate_mock_metabolomics_data(150)
            set_cached_data(counts_df, design_df)
            de_results_df = None
            head_counts = counts_df.head(20).reset_index().rename(columns={"index": "Gene"}).to_dict(orient="records")
            design_list = design_df.reset_index().to_dict(orient="records")
            return {
                "status": "success",
                "omics": "metabolomics",
                "counts": head_counts,
                "design": design_list,
                "total_genes": len(counts_df),
                "samples": list(counts_df.columns)
            }
            
        elif omics == "genomics":
            gwas_variants = generate_mock_genomics_data(2500)
            head_variants = gwas_variants[:30]
            return {
                "status": "success",
                "omics": "genomics",
                "counts": head_variants,
                "design": [],
                "total_genes": len(gwas_variants),
                "samples": []
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported omics module type: {omics}")
            
    except Exception as e:
        logger.error(f"Error loading mock data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate mock data: {str(e)}")

@app.post("/api/upload")
def upload_files(
    counts_file: UploadFile = File(...),
    design_file: Optional[UploadFile] = File(None),
    omics: str = Form("transcriptomics")
):
    global counts_df, design_df, de_results_df, current_omics, gwas_variants
    current_omics = omics
    try:
        counts_bytes = counts_file.file.read()
        counts_str = counts_bytes.decode('utf-8')
        sep = '\t' if counts_file.filename.endswith(('.tsv', '.txt')) else ','
        
        if omics == "genomics":
            uploaded_variants = pd.read_csv(io.StringIO(counts_str), sep=sep)
            gwas_variants = uploaded_variants.to_dict(orient="records")
            return {
                "status": "success",
                "omics": "genomics",
                "total_genes": len(gwas_variants),
                "samples": [],
                "counts": gwas_variants[:30],
                "design": []
            }
            
        if not design_file:
            raise HTTPException(status_code=400, detail="Sample design file is required for expression profiling omics.")
            
        uploaded_counts = pd.read_csv(io.StringIO(counts_str), sep=sep)
        first_col = uploaded_counts.columns[0]
        uploaded_counts = uploaded_counts.set_index(first_col)
        
        design_bytes = design_file.file.read()
        design_str = design_bytes.decode('utf-8')
        sep_d = '\t' if design_file.filename.endswith(('.tsv', '.txt')) else ','
        uploaded_design = pd.read_csv(io.StringIO(design_str), sep=sep_d)
        sample_col = uploaded_design.columns[0]
        uploaded_design = uploaded_design.set_index(sample_col)
        
        counts_cols = set(uploaded_counts.columns)
        design_samples = set(uploaded_design.index)
        
        if not design_samples.issubset(counts_cols):
            missing = design_samples - counts_cols
            raise HTTPException(
                status_code=400, 
                detail=f"Samples in design metadata are missing in expression columns: {list(missing)}"
            )
            
        counts_df = uploaded_counts[list(uploaded_design.index)].astype(int)
        design_df = uploaded_design
        set_cached_data(counts_df, design_df)
        de_results_df = None
        
        head_counts = counts_df.head(20).reset_index().rename(columns={counts_df.index.name or "Gene": "Gene"}).to_dict(orient="records")
        design_list = design_df.reset_index().to_dict(orient="records")
        
        return {
            "status": "success",
            "omics": omics,
            "total_genes": len(counts_df),
            "samples": list(counts_df.columns),
            "counts": head_counts,
            "design": design_list
        }
    except Exception as e:
        logger.error(f"Error uploading files: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded files: {str(e)}")

@app.post("/api/analyze")
def run_analysis(
    p_adj_cutoff: float = Form(0.05),
    log2fc_cutoff: float = Form(1.0)
):
    global counts_df, design_df, de_results_df, current_omics, gwas_variants
    
    if current_omics == "genomics":
        if gwas_variants is None:
            raise HTTPException(status_code=400, detail="No variants loaded for Genomics.")
        try:
            gwas_results = run_gwas_analysis(gwas_variants)
            return {
                "status": "success",
                "omics": "genomics",
                "gwas": gwas_results
            }
        except Exception as e:
            logger.error(f"Genomics analysis failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Genomics analysis failed: {str(e)}")
            
    if counts_df is None or design_df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded. Please import data first.")
        
    try:
        de_results_df = run_differential_expression(counts_df, design_df)
        import backend.agent as agent_module
        agent_module._data_cache["de_results"] = de_results_df
        
        if current_omics == "metabolomics":
            pca_df, explained_var = run_plsda_analysis(counts_df, design_df)
        else:
            pca_df, explained_var = run_pca_analysis(counts_df, design_df)
            
        heatmap = get_heatmap_data(counts_df, de_results_df, top_n=50)
        
        enrichment = []
        ppi_network = {}
        if current_omics == "proteomics":
            ppi_network = get_ppi_network(de_results_df, p_adj_cutoff)
        elif current_omics in ["transcriptomics", "metabolomics"]:
            enrichment = run_pathway_enrichment(de_results_df, p_adj_cutoff, log2fc_cutoff)
            
        de_results_list = de_results_df.reset_index().rename(columns={"Gene": "Gene"}).to_dict(orient="records")
        pca_list = pca_df.reset_index().to_dict(orient="records")
        
        return {
            "status": "success",
            "omics": current_omics,
            "de_results": de_results_list,
            "pca": {
                "coordinates": pca_list,
                "explained_variance": explained_var
            },
            "enrichment": enrichment,
            "ppi": ppi_network,
            "heatmap": heatmap
        }
    except Exception as e:
        logger.error(f"Multi-omics analysis pipeline failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis pipeline crashed: {str(e)}")

@app.post("/api/chat")
def chat_with_agent(chat_input: ChatMessage):
    global counts_df, design_df, current_omics, model_config
    if current_omics != "genomics" and (counts_df is None or design_df is None):
        logger.info("Auto-loading mock data for chat session initialization.")
        counts_df, design_df = generate_mock_rnaseq_data()
        set_cached_data(counts_df, design_df)
        
    try:
        # Resolve active model configuration details
        active_id = model_config.get("active_model_id", "qwen-plus")
        active_model = next((m for m in model_config.get("models", []) if m["id"] == active_id), None)
        
        m_name = active_model["model_name"] if active_model else "qwen-plus"
        base_url = active_model["base_url"] if active_model else "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key = active_model["api_key"] if active_model else ""
        
        agent = get_agent_instance(
            model_name=m_name,
            api_key=api_key,
            base_url=base_url
        )
        
        input_data = {"messages": [("user", chat_input.message)]}
        response = agent.invoke(input_data)
        
        messages = response.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "content"):
                reply = last_message.content
            elif isinstance(last_message, dict) and "content" in last_message:
                reply = last_message["content"]
            else:
                reply = str(last_message)
        else:
            reply = "I ran the task but didn't generate any response text."
            
        return {
            "status": "success",
            "reply": reply,
            "steps": [str(m) for m in messages]
        }
    except Exception as e:
        logger.error(f"Agent execution error: {str(e)}")
        err_msg = str(e)
        active_id = model_config.get("active_model_id", "qwen-plus")
        fallback_reply = (
            f"⚠️ **LLM Model Connection Error**: DeepAgents failed to invoke the active model (`{active_id}`). \n\n"
            f"Error detail: `{err_msg}`\n\n"
            f"**Suggestions**:\n"
            f"1. Check if the API key in the **Settings** panel for this model is valid and funded.\n"
            f"2. Run the **Test Connection** button on this configuration to locate endpoint or network failures.\n"
            f"3. Switch the active model to **Local Ollama** or another provider in the settings list.\n\n"
            f"*(Note: You can still run calculations and view plots locally without LLM features!)*"
        )
        return {"status": "error", "reply": fallback_reply}
