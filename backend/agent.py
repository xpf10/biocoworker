import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
import pandas as pd
import json
import logging

logger = logging.getLogger("biocoworker-agent")

# Cache to store loaded data in memory for analysis
# In production, this could be stored in a file or database
_data_cache = {
    "counts": None,
    "design": None,
    "de_results": None,
    "pca_results": None,
    "enrichment": None
}

def set_cached_data(counts: pd.DataFrame, design: pd.DataFrame):
    _data_cache["counts"] = counts
    _data_cache["design"] = design
    # Reset downstream analyses
    _data_cache["de_results"] = None
    _data_cache["pca_results"] = None
    _data_cache["enrichment"] = None

@tool
def get_dataset_summary() -> str:
    """
    Returns a summary of the currently loaded RNA-seq counts and design matrix.
    Use this to understand what data is loaded.
    """
    counts = _data_cache.get("counts")
    design = _data_cache.get("design")
    if counts is None or design is None:
        return "No RNA-seq dataset is currently loaded. Please load a mock dataset or upload your own files."
    
    summary = {
        "num_genes": len(counts),
        "samples": list(counts.columns),
        "groups": design["Group"].value_counts().to_dict()
    }
    return f"Loaded RNA-seq dataset summary:\n{json.dumps(summary, indent=2)}"

@tool
def run_differential_expression_analysis() -> str:
    """
    Runs Differential Expression (DE) analysis on the loaded dataset.
    Returns the top 15 most significantly changed genes (lowest PAdj).
    """
    counts = _data_cache.get("counts")
    design = _data_cache.get("design")
    if counts is None or design is None:
        return "Error: No dataset loaded. Cannot run differential expression."
    
    from backend.analysis import run_differential_expression
    try:
        results = run_differential_expression(counts, design)
        _data_cache["de_results"] = results
        
        # Format top 15 genes
        top_genes = results.sort_values("PAdj").head(15)
        genes_summary = []
        for gene, row in top_genes.iterrows():
            genes_summary.append(
                f"- {gene}: Log2FC = {row['Log2FC']:.3f}, PValue = {row['PValue']:.2e}, PAdj = {row['PAdj']:.2e}"
            )
            
        summary_str = "\n".join(genes_summary)
        return (
            f"Differential expression analysis completed successfully!\n"
            f"Analyzed {len(results)} genes. Top 15 differentially expressed genes (by PAdj):\n{summary_str}"
        )
    except Exception as e:
        return f"Failed to run DE analysis: {str(e)}"

@tool
def run_enrichment_analysis(p_adj_cutoff: float = 0.05, log2fc_cutoff: float = 1.0) -> str:
    """
    Runs Pathway Enrichment Analysis (KEGG/GO) on significantly differentially expressed genes.
    Parameters:
    - p_adj_cutoff: adjusted p-value threshold (default 0.05).
    - log2fc_cutoff: absolute log2 fold change threshold (default 1.0).
    """
    de_results = _data_cache.get("de_results")
    if de_results is None:
        # Auto-run DE analysis first if not run yet
        counts = _data_cache.get("counts")
        design = _data_cache.get("design")
        if counts is None or design is None:
            return "Error: No dataset loaded. Please load a dataset first."
        from backend.analysis import run_differential_expression
        de_results = run_differential_expression(counts, design)
        _data_cache["de_results"] = de_results
        
    from backend.analysis import run_pathway_enrichment
    try:
        enrichment = run_pathway_enrichment(de_results, p_adj_cutoff, log2fc_cutoff)
        _data_cache["enrichment"] = enrichment
        
        if not enrichment:
            return "No enriched pathways found. Try loosening the thresholds (e.g. increase p_adj_cutoff or decrease log2fc_cutoff)."
            
        summary = []
        for i, item in enumerate(enrichment[:8]):
            summary.append(
                f"{i+1}. {item['Pathway']}: {item['Overlap']}/{item['Pathway_Size']} genes, P-Value = {item['PValue']:.2e}"
            )
        return "Pathway Enrichment Analysis completed! Top enriched pathways:\n" + "\n".join(summary)
    except Exception as e:
        return f"Failed to run pathway enrichment: {str(e)}"

@tool
def get_gene_details(gene_name: str) -> str:
    """
    Retrieves biological function and details for a specific gene from the loaded dataset.
    Use this to explain what a specific gene does when analyzing differential expression.
    """
    de_results = _data_cache.get("de_results")
    
    # Check if the gene exists in our expression list
    in_dataset = False
    stats_info = ""
    if de_results is not None and gene_name in de_results.index:
        in_dataset = True
        row = de_results.loc[gene_name]
        stats_info = f"\nDataset Stats:\n- Log2FC: {row['Log2FC']:.3f}\n- P-value: {row['PValue']:.2e}\n- Adjusted P-value (FDR): {row['PAdj']:.2e}"
        
    # Standard biological functions for key genes
    gene_db = {
        "TP53": "TP53 (Tumor Protein p53) is a tumor suppressor gene. It encodes a protein that regulates the cell cycle and prevents genome mutation. Often downregulated in cancers or mutated, causing cell cycle progression and evasion of apoptosis.",
        "EGFR": "EGFR (Epidermal Growth Factor Receptor) is a cell-surface receptor for extracellular protein ligands. Overexpression or overactivity is associated with a number of cancers, driving cell proliferation and survival via MAPK/PI3K pathways.",
        "MYC": "MYC is a proto-oncogene encoding a transcription factor that plays a role in cell cycle progression, apoptosis and cellular transformation. Overexpressed in many cancers, promoting ribosome biogenesis and glycolysis.",
        "TNF": "TNF (Tumor Necrosis Factor) is a multifunctional proinflammatory cytokine. It regulates immune cells, induces inflammation, and can trigger cell death (apoptosis) or cell survival depending on the receptor signaling context.",
        "IL6": "IL6 (Interleukin 6) is an interleukin that acts as both a pro-inflammatory cytokine and an anti-inflammatory myokine. Involved in immune response regulation, cell survival, and inflammation.",
        "GAPDH": "GAPDH (Glyceraldehyde-3-phosphate dehydrogenase) is a housekeeping gene involved in glycolysis. It is commonly used as an internal loading control in expression analysis.",
        "ACTB": "ACTB (Beta-actin) is a housekeeping gene encoding a major structural component of the cytoskeleton. Often used as an expression loading control.",
        "BRCA1": "BRCA1 is a caretaker gene that encodes a tumor suppressor. Involved in DNA double-strand break repair, transcription regulation, and cell cycle checkpoint control.",
        "MTOR": "MTOR (Mechanistic Target of Rapamycin kinase) regulates cell growth, cell proliferation, cell motility, cell survival, protein synthesis, autophagy, and transcription. Central hub in metabolic sensing.",
        "AKT1": "AKT1 (AKT Serine/Threonine Kinase 1) is a critical signaling node in the PI3K-Akt pathway. Regulates cellular survival, metabolism, growth, and angiogenesis."
    }
    
    info = gene_db.get(gene_name.upper(), f"Gene {gene_name} is part of the loaded dataset. You can explain its function based on standard biological literature.")
    
    if in_dataset:
        return f"Gene: {gene_name}\nDescription: {info}{stats_info}"
    else:
        return f"Gene {gene_name} was not found in the currently analyzed RNA-seq dataset. Description: {info}"

def get_agent_instance(model_name: str = "qwen-plus", api_key: str = None, base_url: str = None):
    """
    Initializes and returns a deepagent instance.
    Uses domestic models (Qwen, DeepSeek, GLM, etc.) by wrapping them in ChatOpenAI.
    """
    # Environment variable fallbacks
    key = api_key or os.environ.get("COWORKER_API_KEY") or os.environ.get("OPENAI_API_KEY") or "mock-key"
    url = base_url or os.environ.get("COWORKER_API_BASE") or os.environ.get("OPENAI_API_BASE") or "https://api.openai.com/v1"
    
    # Determine the model. If no api key is real, we can mock responses.
    if key == "mock-key" and not os.environ.get("COWORKER_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        logger.warning("No real API key found. Agent will operate with fallback mock responses.")
        
    # Configure the Chat model
    # We pass it directly to create_deep_agent
    chat_model = ChatOpenAI(
        model=model_name,
        openai_api_key=key,
        openai_api_base=url,
        temperature=0.2
    )
    
    system_prompt = (
        "You are BioCoworker, an advanced AI agent specializing in Bioinformatics and RNA-seq downstream analysis.\n"
        "You help researchers analyze their RNA-seq datasets (counts matrix and sample metadata groups).\n"
        "Your available tools allow you to:\n"
        "1. Check the loaded dataset summary.\n"
        "2. Run differential expression analysis (which computes Log2 Fold Change, P-values, and FDR).\n"
        "3. Run pathway enrichment analysis (KEGG/GO) to find biological pathways affected by the treatment.\n"
        "4. Look up functional descriptions of specific genes and relate them to the dataset statistics.\n\n"
        "Please guide the researcher through the analysis. If they ask about differential expression, explain "
        "what the tools did and identify key genes (like TP53, EGFR, MYC, TNF, etc.) and pathways "
        "(like MAPK, Cell Cycle, Apoptosis) that are affected."
    )
    
    # Create the agent using deepagents
    agent = create_deep_agent(
        model=chat_model,
        tools=[
            get_dataset_summary,
            run_differential_expression_analysis,
            run_enrichment_analysis,
            get_gene_details
        ],
        system_prompt=system_prompt
    )
    
    return agent
