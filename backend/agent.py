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

# Global state for working directory
_working_dir = None

@tool
def set_working_directory(dir_path: str) -> str:
    """
    Sets the working directory where the user's data files (CSV/TSV) are stored.
    Parameters:
    - dir_path: absolute path to the directory (e.g. 'D:/data/project').
    """
    global _working_dir
    if not os.path.exists(dir_path):
        return f"Error: Directory path '{dir_path}' does not exist on the file system."
    if not os.path.isdir(dir_path):
        return f"Error: Path '{dir_path}' is a file, not a directory."
    
    _working_dir = os.path.abspath(dir_path)
    try:
        files = os.listdir(_working_dir)
        csv_tsv_files = [f for f in files if f.endswith(('.csv', '.tsv', '.txt'))]
        files_str = ", ".join(csv_tsv_files) if csv_tsv_files else "No CSV/TSV/TXT files found."
        return f"Working directory set successfully to: {_working_dir}\nAvailable data files in directory: {files_str}"
    except Exception as e:
        return f"Working directory set to: {_working_dir}, but failed to list files: {str(e)}"

@tool
def get_working_directory() -> str:
    """
    Returns the current working directory path and lists the available CSV/TSV data files in it.
    """
    global _working_dir
    if _working_dir is None:
        return "No working directory is currently set. Use set_working_directory to specify one."
    try:
        files = os.listdir(_working_dir)
        csv_tsv_files = [f for f in files if f.endswith(('.csv', '.tsv', '.txt'))]
        files_str = ", ".join(csv_tsv_files) if csv_tsv_files else "No CSV/TSV/TXT files found."
        return f"Current working directory: {_working_dir}\nAvailable files: {files_str}"
    except Exception as e:
        return f"Current working directory: {_working_dir}, but failed to list files: {str(e)}"

@tool
def load_expression_dataset(counts_filename: str, design_filename: str) -> str:
    """
    Loads a raw expression counts file and a design grouping file from the current working directory.
    Parameters:
    - counts_filename: filename of the counts matrix (CSV/TSV).
    - design_filename: filename of the design metadata (CSV/TSV).
    """
    global _working_dir
    if _working_dir is None:
        return "Error: Working directory is not set. Please set it using set_working_directory first."
    
    counts_path = os.path.join(_working_dir, counts_filename)
    design_path = os.path.join(_working_dir, design_filename)
    
    if not os.path.exists(counts_path):
        return f"Error: Counts file '{counts_filename}' not found in working directory."
    if not os.path.exists(design_path):
        return f"Error: Design file '{design_filename}' not found in working directory."
        
    try:
        # Load counts
        sep_c = '\t' if counts_filename.endswith(('.tsv', '.txt')) else ','
        try:
            uploaded_counts = pd.read_csv(counts_path, sep=sep_c)
        except UnicodeDecodeError:
            uploaded_counts = pd.read_csv(counts_path, sep=sep_c, encoding='gbk')
            
        first_col = uploaded_counts.columns[0]
        uploaded_counts = uploaded_counts.set_index(first_col)
        
        # Load design
        sep_d = '\t' if design_filename.endswith(('.tsv', '.txt')) else ','
        try:
            uploaded_design = pd.read_csv(design_path, sep=sep_d)
        except UnicodeDecodeError:
            uploaded_design = pd.read_csv(design_path, sep=sep_d, encoding='gbk')
            
        sample_col = uploaded_design.columns[0]
        uploaded_design = uploaded_design.set_index(sample_col)
        
        # Validate
        counts_cols = set(uploaded_counts.columns)
        design_samples = set(uploaded_design.index)
        common_samples = counts_cols.intersection(design_samples)
        
        if not common_samples:
            return f"Error: No matching samples between counts file columns and design file index."
            
        # Sync
        common_list = sorted(list(common_samples))
        counts_df = uploaded_counts[common_list].astype(int)
        design_df = uploaded_design.loc[common_list]
        
        # Cache
        set_cached_data(counts_df, design_df)
        
        # Also sync with main.py globals if active
        import backend.main as main_module
        main_module.counts_df = counts_df
        main_module.design_df = design_df
        main_module.de_results_df = None
        
        return (
            f"Successfully loaded dataset from working directory!\n"
            f"- Gene Count: {len(counts_df)}\n"
            f"- Sample Count: {len(counts_df.columns)}\n"
            f"- Groups: {design_df['Group'].value_counts().to_dict()}"
        )
    except Exception as e:
        return f"Failed to load dataset: {str(e)}"

@tool
def load_differential_expression_table(filename: str, is_mouse: bool = False, database: str = "KEGG") -> str:
    """
    Loads an already analyzed differential expression table (e.g. from DESeq2 or edgeR) from the working directory.
    Parameters:
    - filename: name of the DE table file (CSV/TSV).
    - is_mouse: set to True if it is a mouse dataset (converts pathway genes to title-case).
    - database: database to use for downstream enrichment (e.g., 'KEGG', 'GO_BP', 'GO_MF', 'GO_CC').
    """
    global _working_dir
    if _working_dir is None:
        return "Error: Working directory is not set. Please set it using set_working_directory first."
        
    file_path = os.path.join(_working_dir, filename)
    if not os.path.exists(file_path):
        return f"Error: File '{filename}' not found in working directory."
        
    try:
        sep = '\t' if filename.endswith(('.tsv', '.txt')) else ','
        try:
            uploaded_df = pd.read_csv(file_path, sep=sep)
        except UnicodeDecodeError:
            uploaded_df = pd.read_csv(file_path, sep=sep, encoding='gbk')
            
        # Map the gene names
        gene_col = None
        for col in uploaded_df.columns:
            if col.lower() in ['gene', 'gene_id', 'gene_name', 'symbol', 'id', 'name']:
                gene_col = col
                break
        if gene_col:
            uploaded_df = uploaded_df.rename(columns={gene_col: 'Gene'})
        else:
            first_col = uploaded_df.columns[0]
            uploaded_df = uploaded_df.rename(columns={first_col: 'Gene'})
            
        uploaded_df = uploaded_df.set_index('Gene')
        
        # Map logFC
        fc_col = None
        for col in uploaded_df.columns:
            if col.lower() in ['log2foldchange', 'log2fc', 'logfc', 'log2_fold_change', 'log2_fc', 'log_fc']:
                fc_col = col
                break
         
        # Map PValue
        p_col = None
        for col in uploaded_df.columns:
            if col.lower() in ['pvalue', 'p.value', 'p_value', 'pval', 'p']:
                p_col = col
                break
         
        # Map PAdj / FDR
        padj_col = None
        for col in uploaded_df.columns:
            if col.lower() in ['padj', 'p.adjust', 'p_adj', 'fdr', 'qvalue', 'q-value', 'adj_p']:
                padj_col = col
                break
                 
        if not fc_col or not p_col:
            return (
                f"Error: Could not identify differential expression columns. "
                f"Available columns: {list(uploaded_df.columns)}. "
                f"Need columns similar to 'log2FoldChange' / 'logFC' and 'pvalue' / 'PValue'."
            )
            
        mapped_df = pd.DataFrame(index=uploaded_df.index)
        mapped_df['Log2FC'] = uploaded_df[fc_col].astype(float)
        mapped_df['PValue'] = uploaded_df[p_col].astype(float)
        
        if padj_col:
            mapped_df['PAdj'] = uploaded_df[padj_col].astype(float)
        else:
            from statsmodels.stats import multitest
            pvals = mapped_df['PValue'].fillna(1.0).values
            _, padj, _, _ = multitest.multipletests(pvals, alpha=0.05, method='fdr_bh')
            mapped_df['PAdj'] = padj
            
        mapped_df['Mean_Control'] = 10.0
        mapped_df['Mean_Treat'] = 10.0 + mapped_df['Log2FC']
        
        # Cache de_results
        _data_cache["de_results"] = mapped_df
        
        # Setup dummy counts and design for compatibility
        counts_df = pd.DataFrame(index=mapped_df.index)
        counts_df['Control_1'] = 100
        counts_df['Treat_1'] = 100
        design_df = pd.DataFrame({'Group': ['Control', 'Treat']}, index=['Control_1', 'Treat_1'])
        
        _data_cache["counts"] = counts_df
        _data_cache["design"] = design_df
        
        # Also sync with main.py globals
        import backend.main as main_module
        main_module.counts_df = counts_df
        main_module.design_df = design_df
        main_module.de_results_df = mapped_df
        main_module.current_organism = "mouse" if is_mouse else "human"
        main_module.current_database = database
        
        return (
            f"Successfully imported pre-analyzed differential expression table from working directory!\n"
            f"- Total genes loaded: {len(mapped_df)}\n"
            f"- Species: {'mouse (小鼠)' if is_mouse else 'human (人类)'}\n"
            f"- Database set: {database}\n"
            f"- Up-regulated genes count (FDR <= 0.05, Log2FC >= 1): {len(mapped_df[(mapped_df['PAdj'] <= 0.05) & (mapped_df['Log2FC'] >= 1)])}\n"
            f"- Down-regulated genes count (FDR <= 0.05, Log2FC <= -1): {len(mapped_df[(mapped_df['PAdj'] <= 0.05) & (mapped_df['Log2FC'] <= -1)])}"
        )
    except Exception as e:
        return f"Failed to import differential expression table: {str(e)}"

def get_agent_instance(model_name: str = "qwen-plus", api_key: str = None, base_url: str = None):
    """
    Initializes and returns a deepagent instance.
    Uses domestic models (Qwen, DeepSeek, GLM, etc.) by wrapping them in ChatOpenAI.
    """
    key = api_key or os.environ.get("COWORKER_API_KEY") or os.environ.get("OPENAI_API_KEY") or "mock-key"
    url = base_url or os.environ.get("COWORKER_API_BASE") or os.environ.get("OPENAI_API_BASE") or "https://api.openai.com/v1"
    
    if key == "mock-key" and not os.environ.get("COWORKER_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        logger.warning("No real API key found. Agent will operate with fallback mock responses.")
        
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
        "1. set_working_directory: Set the folder path where the user's raw files are stored.\n"
        "2. get_working_directory: List the files in the current working directory.\n"
        "3. load_expression_dataset: Load a raw expression counts matrix and design file from the working directory.\n"
        "4. load_differential_expression_table: Directly load a pre-analyzed differential expression table (DESeq2/edgeR) from the working directory.\n"
        "5. get_dataset_summary: Check the loaded dataset summary.\n"
        "6. run_differential_expression_analysis: Run differential expression on loaded counts.\n"
        "7. run_enrichment_analysis: Run pathway enrichment (KEGG/GO) on loaded/analyzed differential expression results.\n"
        "8. get_gene_details: Look up functional details of specific genes.\n\n"
        "Please guide the researcher through the analysis. You can set the working directory, import files, "
        "and run analysis directly. If they ask about differential expression, explain "
        "what the tools did and identify key genes (like TP53, EGFR, MYC, TNF, etc.) and pathways "
        "(like MAPK, Cell Cycle, Apoptosis) that are affected."
    )
    
    agent = create_deep_agent(
        model=chat_model,
        tools=[
            get_dataset_summary,
            run_differential_expression_analysis,
            run_enrichment_analysis,
            get_gene_details,
            set_working_directory,
            get_working_directory,
            load_expression_dataset,
            load_differential_expression_table
        ],
        system_prompt=system_prompt
    )
    
    return agent
