import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
import statsmodels.stats.multitest as multi
import typing
import subprocess
import tempfile
import os
import shutil

# =====================================================================
# 1. TRANSCRIPTOMICS (RNA-seq) MODULE
# =====================================================================

def generate_mock_rnaseq_data(num_genes: int = 1500) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(42)
    samples = ['Control_1', 'Control_2', 'Control_3', 'Treat_1', 'Treat_2', 'Treat_3']
    genes = [f"Gene_{i+1:04d}" for i in range(num_genes)]
    
    known_genes = {
        10: "GAPDH", 20: "ACTB", 30: "TP53", 40: "EGFR", 50: "MYC",
        60: "TNF", 70: "IL6", 80: "VEGFA", 90: "MTOR", 100: "BRCA1",
        110: "AKT1", 120: "STAT3", 130: "CD44", 140: "JUN", 150: "FOS"
    }
    for idx, name in known_genes.items():
        if idx < num_genes:
            genes[idx] = name

    base_expr = np.random.lognormal(mean=5.0, sigma=1.5, size=num_genes)
    base_expr = np.clip(base_expr, 10, None)
    
    control_counts = np.zeros((num_genes, 3))
    for i in range(3):
        dispersion = 0.1 + 0.2 * np.random.rand(num_genes)
        mu = base_expr
        size = 1 / dispersion
        prob = size / (size + mu)
        control_counts[:, i] = np.random.negative_binomial(size, prob)
        
    treat_counts = np.zeros((num_genes, 3))
    de_genes_mask = np.random.rand(num_genes) < 0.12
    de_genes_mask[30] = True  # TP53 (down)
    de_genes_mask[40] = True  # EGFR (up)
    de_genes_mask[50] = True  # MYC (up)
    de_genes_mask[60] = True  # TNF (up)
    
    log2_fc = np.zeros(num_genes)
    log2_fc[de_genes_mask] = np.random.uniform(1.2, 4.0, size=de_genes_mask.sum()) * np.random.choice([-1, 1], size=de_genes_mask.sum())
    log2_fc[30] = -2.5  # TP53
    log2_fc[40] = 3.2   # EGFR
    log2_fc[50] = 2.8   # MYC
    log2_fc[60] = 3.5   # TNF

    for i in range(3):
        dispersion = 0.1 + 0.2 * np.random.rand(num_genes)
        mu = base_expr * (2 ** log2_fc)
        size = 1 / dispersion
        prob = size / (size + mu)
        treat_counts[:, i] = np.random.negative_binomial(size, prob)
        
    counts = np.hstack([control_counts, treat_counts])
    counts_df = pd.DataFrame(counts.astype(int), index=genes, columns=samples)
    design_df = pd.DataFrame({
        'Sample': samples,
        'Group': ['Control', 'Control', 'Control', 'Treat', 'Treat', 'Treat']
    }).set_index('Sample')
    
    return counts_df, design_df

def run_differential_expression(counts_df: pd.DataFrame, design_df: pd.DataFrame) -> pd.DataFrame:
    # Filter low count genes
    counts_df = counts_df[counts_df.sum(axis=1) > 0]
    
    # Try running pydeseq2
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
        
        # DeseqDataSet expects integer counts and samples as rows (T)
        counts_int = counts_df.round().astype(int)
        
        dds = DeseqDataSet(
            counts=counts_int.T,
            clinical=design_df,
            design_factors="Group",
            quiet=True
        )
        dds.deseq2()
        
        # Extract statistics for Treat vs Control
        stat_res = DeseqStats(dds, contrast=["Group", "Treat", "Control"])
        stat_res.summary()
        res_df = stat_res.results_df
        
        # Calculate raw group means for UI display
        ctrl_samples = design_df[design_df['Group'] == 'Control'].index.tolist()
        treat_samples = design_df[design_df['Group'] == 'Treat'].index.tolist()
        cpm_df = counts_df.div(counts_df.sum(axis=0), axis=1) * 1e6
        log_cpm = np.log2(cpm_df + 1)
        mean_ctrl = log_cpm[ctrl_samples].mean(axis=1)
        mean_treat = log_cpm[treat_samples].mean(axis=1)
        
        # DeseqStats results_df has index matching the genes
        results_df = pd.DataFrame({
            'Mean_Control': mean_ctrl.loc[counts_df.index].values,
            'Mean_Treat': mean_treat.loc[counts_df.index].values,
            'Log2FC': res_df['log2FoldChange'].loc[counts_df.index].values,
            'PValue': res_df['pvalue'].loc[counts_df.index].values,
            'PAdj': res_df['padj'].loc[counts_df.index].fillna(1.0).values
        }, index=counts_df.index)
        
        print("INFO: Successfully ran differential expression using pydeseq2.")
        return results_df
        
    except Exception as e:
        print(f"WARNING: pydeseq2 run failed or not installed: {str(e)}. Falling back to standard t-test on log-CPM.")
        
        ctrl_samples = design_df[design_df['Group'] == 'Control'].index.tolist()
        treat_samples = design_df[design_df['Group'] == 'Treat'].index.tolist()
        
        cpm_df = counts_df.div(counts_df.sum(axis=0), axis=1) * 1e6
        log_cpm = np.log2(cpm_df + 1)
        
        ctrl_expr = log_cpm[ctrl_samples]
        treat_expr = log_cpm[treat_samples]
        
        mean_ctrl = ctrl_expr.mean(axis=1)
        mean_treat = treat_expr.mean(axis=1)
        log2_fc = mean_treat - mean_ctrl
        
        p_values = []
        for gene in counts_df.index:
            c_vals = ctrl_expr.loc[gene].values
            t_vals = treat_expr.loc[gene].values
            t_stat, p_val = stats.ttest_ind(t_vals, c_vals, equal_var=False)
            if np.isnan(p_val):
                p_val = 1.0
            p_values.append(p_val)
            
        p_values = np.array(p_values)
        reject, p_adj, _, _ = multi.multipletests(p_values, alpha=0.05, method='fdr_bh')
        
        results_df = pd.DataFrame({
            'Mean_Control': mean_ctrl.values,
            'Mean_Treat': mean_treat.values,
            'Log2FC': log2_fc.values,
            'PValue': p_values,
            'PAdj': p_adj
        }, index=counts_df.index)
        
        return results_df

def run_pca_analysis(counts_df: pd.DataFrame, design_df: pd.DataFrame) -> typing.Tuple[pd.DataFrame, list]:
    cpm_df = counts_df.div(counts_df.sum(axis=0), axis=1) * 1e6
    log_cpm = np.log2(cpm_df + 1)
    variance = log_cpm.var(axis=1)
    log_cpm = log_cpm[variance > 0]
    
    data_for_pca = log_cpm.T
    data_centered = data_for_pca - data_for_pca.mean(axis=0)
    data_scaled = data_centered / data_for_pca.std(axis=0)
    
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(data_scaled)
    
    pca_df = pd.DataFrame(
        pca_coords,
        columns=['PC1', 'PC2'],
        index=counts_df.columns
    )
    pca_df['Group'] = design_df.loc[pca_df.index, 'Group']
    pca_df['Sample'] = pca_df.index
    
    explained_variance = (pca.explained_variance_ratio_ * 100).tolist()
    return pca_df, explained_variance

def run_pathway_enrichment(de_results: pd.DataFrame, p_adj_cutoff: float = 0.05, log2fc_cutoff: float = 1.0) -> list:
    pathways = {
        "Cell Cycle": ["Gene_0001", "Gene_0002", "GAPDH", "Gene_0012", "Gene_0015", "Gene_0025", "TP53", "MYC", "Gene_0120", "Gene_0140", "Gene_0200", "Gene_0300"],
        "MAPK Signaling Pathway": ["EGFR", "TNF", "IL6", "JUN", "FOS", "Gene_0077", "Gene_0088", "Gene_0099", "Gene_0110", "Gene_0155", "Gene_0210", "Gene_0310", "Gene_0410"],
        "p53 Signaling Pathway": ["TP53", "BRCA1", "Gene_0032", "Gene_0033", "Gene_0034", "Gene_0035", "Gene_0122", "Gene_0180", "Gene_0250", "Gene_0350"],
        "Apoptosis": ["TP53", "TNF", "Gene_0061", "Gene_0062", "Gene_0063", "Gene_0064", "Gene_0115", "Gene_0190", "Gene_0290", "Gene_0390"],
        "PI3K-Akt Signaling Pathway": ["EGFR", "MTOR", "AKT1", "VEGFA", "Gene_0041", "Gene_0042", "Gene_0043", "Gene_0091", "Gene_0092", "Gene_0222", "Gene_0322", "Gene_0422"],
        "Cytokine-Cytokine Receptor Interaction": ["TNF", "IL6", "Gene_0065", "Gene_0066", "Gene_0067", "Gene_0068", "Gene_0160", "Gene_0260", "Gene_0360", "Gene_0460"],
        "Glycolysis / Gluconeogenesis": ["GAPDH", "ACTB", "Gene_0011", "Gene_0013", "Gene_0014", "Gene_0016", "Gene_0101", "Gene_0102", "Gene_0103", "Gene_0201"],
        "Ribosome": [f"Gene_{i:04d}" for i in range(200, 240)]
    }
    
    # Try running clusterProfiler via Rscript
    try:
        sig_de = de_results[(de_results['PAdj'] <= p_adj_cutoff) & (de_results['Log2FC'].abs() >= log2fc_cutoff)]
        sig_genes = sig_de.index.tolist()
        if not sig_genes:
            return []
            
        rscript_path = shutil.which('Rscript')
        if not rscript_path:
            raise FileNotFoundError("Rscript executable not found in system PATH.")
            
        with tempfile.TemporaryDirectory() as tmpdir:
            genes_file = os.path.join(tmpdir, "genes.csv")
            term2gene_file = os.path.join(tmpdir, "term2gene.csv")
            output_file = os.path.join(tmpdir, "output.csv")
            
            # Write files
            pd.Series(sig_genes).to_csv(genes_file, index=False, header=False)
            t2g_list = []
            for term, g_list in pathways.items():
                for g in g_list:
                    t2g_list.append({"Pathway": term, "Gene": g})
            pd.DataFrame(t2g_list).to_csv(term2gene_file, index=False)
            
            # Write R script using clusterProfiler::enricher
            r_script_content = f"""
            suppressPackageStartupMessages(library(clusterProfiler))
            genes <- read.csv("{genes_file.replace('\\', '/')}", header=FALSE)[,1]
            term2gene <- read.csv("{term2gene_file.replace('\\', '/')}")
            
            res <- enricher(
                gene = as.character(genes),
                pvalueCutoff = 1.0,
                pAdjustMethod = "BH",
                minGSSize = 1,
                TERM2GENE = term2gene
            )
            
            if (!is.null(res) && nrow(res@result) > 0) {{
                write.csv(res@result, "{output_file.replace('\\', '/')}", row.names=FALSE)
            }} else {{
                write.csv(data.frame(), "{output_file.replace('\\', '/')}", row.names=FALSE)
            }}
            """
            
            r_script_file = os.path.join(tmpdir, "run_enricher.R")
            with open(r_script_file, "w", encoding="utf-8") as f:
                f.write(r_script_content)
                
            # Execute
            subprocess.run(
                [rscript_path, r_script_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            if os.path.exists(output_file) and os.path.getsize(output_file) > 4:
                res_df = pd.read_csv(output_file)
                if not res_df.empty and 'ID' in res_df.columns:
                    enrichment_results = []
                    for _, row in res_df.iterrows():
                        # check if count > 0
                        if int(row['Count']) > 0:
                            gene_ids = str(row['geneID']).split('/') if not pd.isna(row['geneID']) else []
                            enrichment_results.append({
                                "Pathway": row['ID'],
                                "Overlap": int(row['Count']),
                                "Pathway_Size": len(pathways.get(row['ID'], [])),
                                "Genes": gene_ids,
                                "PValue": float(row['pvalue']),
                                "Log10PValue": -np.log10(float(row['pvalue']) + 1e-30)
                            })
                    print("INFO: Successfully ran enrichment analysis using R clusterProfiler.")
                    return enrichment_results
                    
        raise RuntimeError("clusterProfiler returned no data or R execution failed.")
        
    except Exception as e:
        print(f"WARNING: clusterProfiler run failed: {str(e)}. Falling back to Python hypergeometric GSEA.")
        
        # Hypergeometric fallback (equivalent ORA)
        sig_de = de_results[(de_results['PAdj'] <= p_adj_cutoff) & (de_results['Log2FC'].abs() >= log2fc_cutoff)]
        sig_genes = set(sig_de.index.tolist())
        all_genes_count = len(de_results)
        sig_count = len(sig_genes)
        
        enrichment_results = []
        if sig_count == 0:
            return []
            
        for pathway_name, path_genes in pathways.items():
            path_genes_in_data = [g for g in path_genes if g in de_results.index]
            M = all_genes_count
            n = sig_count
            N = len(path_genes_in_data)
            overlap_genes = sig_genes.intersection(path_genes_in_data)
            k = len(overlap_genes)
            
            if k > 0:
                p_val = stats.hypergeom.sf(k - 1, M, n, N)
                if np.isnan(p_val):
                    p_val = 1.0
                
                enrichment_results.append({
                    "Pathway": pathway_name,
                    "Overlap": k,
                    "Pathway_Size": N,
                    "Genes": list(overlap_genes),
                    "PValue": float(p_val),
                    "Log10PValue": -np.log10(p_val + 1e-30)
                })
                
        enrichment_results.sort(key=lambda x: x["PValue"])
        return enrichment_results

def get_heatmap_data(counts_df: pd.DataFrame, de_results: pd.DataFrame, top_n: int = 50) -> dict:
    top_genes = de_results.sort_values('PAdj').head(top_n).index.tolist()
    if not top_genes:
        return {"genes": [], "samples": [], "matrix": []}
        
    cpm_df = counts_df.div(counts_df.sum(axis=0), axis=1) * 1e6
    log_cpm = np.log2(cpm_df + 1)
    sub_matrix = log_cpm.loc[top_genes]
    
    row_means = sub_matrix.mean(axis=1)
    row_stds = sub_matrix.std(axis=1).replace(0, 1)
    z_matrix = sub_matrix.sub(row_means, axis=0).div(row_stds, axis=0)
    
    samples = counts_df.columns.tolist()
    matrix_data = []
    for gene in top_genes:
        gene_vals = {}
        for sample in samples:
            val = float(z_matrix.loc[gene, sample])
            val = max(min(val, 3.0), -3.0)
            gene_vals[sample] = val
        matrix_data.append({
            "Gene": gene,
            "Log2FC": float(de_results.loc[gene, "Log2FC"]),
            "PAdj": float(de_results.loc[gene, "PAdj"]),
            "values": gene_vals
        })
        
    return {
        "genes": top_genes,
        "samples": samples,
        "matrix": matrix_data
    }

# =====================================================================
# 2. PROTEOMICS MODULE
# =====================================================================

def generate_mock_proteomics_data(num_proteins: int = 400) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates mock Proteomics LFQ intensity data (Control vs Treat, 3 vs 3)."""
    np.random.seed(101)
    samples = ['Control_1', 'Control_2', 'Control_3', 'Treat_1', 'Treat_2', 'Treat_3']
    
    # Generate common protein name symbols (e.g. ALB, INS, AKT1)
    prefixes = ['ALB', 'INS', 'TNF', 'AKT1', 'CASP3', 'EGFR', 'MYC', 'MAPK1', 'TGFB1', 'IL1B', 'GAPDH', 'ACTB']
    proteins = []
    for i in range(num_proteins):
        if i < len(prefixes):
            proteins.append(prefixes[i])
        else:
            proteins.append(f"PRT_{i+1:04d}")
            
    base_expr = np.random.uniform(18.0, 32.0, size=num_proteins) # log2 LFQ intensities
    
    control_vals = np.zeros((num_proteins, 3))
    for i in range(3):
        control_vals[:, i] = base_expr + np.random.normal(0, 0.4, size=num_proteins)
        
    treat_vals = np.zeros((num_proteins, 3))
    de_proteins_mask = np.random.rand(num_proteins) < 0.15
    de_proteins_mask[0] = True # ALB (down)
    de_proteins_mask[1] = True # INS (up)
    de_proteins_mask[4] = True # CASP3 (up)
    
    log2_fc = np.zeros(num_proteins)
    log2_fc[de_proteins_mask] = np.random.uniform(1.0, 3.5, size=de_proteins_mask.sum()) * np.random.choice([-1, 1], size=de_proteins_mask.sum())
    log2_fc[0] = -2.2
    log2_fc[1] = 2.8
    log2_fc[4] = 1.9
    
    for i in range(3):
        treat_vals[:, i] = base_expr + log2_fc + np.random.normal(0, 0.4, size=num_proteins)
        
    # Convert from log scale back to linear counts/intensity for raw data loader
    intensities = 2 ** np.hstack([control_vals, treat_vals])
    
    counts_df = pd.DataFrame(intensities.astype(int), index=proteins, columns=samples)
    design_df = pd.DataFrame({
        'Sample': samples,
        'Group': ['Control', 'Control', 'Control', 'Treat', 'Treat', 'Treat']
    }).set_index('Sample')
    
    return counts_df, design_df

def get_ppi_network(de_results: pd.DataFrame, p_adj_cutoff: float = 0.05) -> dict:
    """Generates a mock Protein-Protein Interaction (PPI) network coordinate matrix for significant proteins."""
    sig_proteins = de_results[de_results['PAdj'] <= p_adj_cutoff].index.tolist()[:15]
    if not sig_proteins:
        sig_proteins = de_results.index.tolist()[:8]
        
    nodes = []
    edges = []
    
    # Calculate circle position coordinates
    num_nodes = len(sig_proteins)
    for i, name in enumerate(sig_proteins):
        angle = (2 * np.pi * i) / num_nodes
        x = 300 + 150 * np.cos(angle)
        y = 200 + 150 * np.sin(angle)
        log2fc = float(de_results.loc[name, 'Log2FC'])
        
        nodes.append({
            "id": name,
            "x": x,
            "y": y,
            "Log2FC": log2fc,
            "PAdj": float(de_results.loc[name, 'PAdj'])
        })
        
    # Generate mock interaction edges (random connectivity based on indexing)
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if (i + j) % 3 == 0 or (i * j) % 5 == 1: # mock string db connections
                edges.append({
                    "source": sig_proteins[i],
                    "target": sig_proteins[j],
                    "score": float(np.random.uniform(0.4, 0.95))
                })
                
    return {"nodes": nodes, "edges": edges}

# =====================================================================
# 3. METABOLOMICS MODULE
# =====================================================================

def generate_mock_metabolomics_data(num_metabolites: int = 150) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates mock Metabolomics relative abundance profile data (Control vs Treat)."""
    np.random.seed(202)
    samples = ['Control_1', 'Control_2', 'Control_3', 'Treat_1', 'Treat_2', 'Treat_3']
    
    known_metabolites = ['L-Alanine', 'Glucose', 'Citric Acid', 'Lactate', 'Cholesterol', 'Urea', 'Pyruvate', 'L-Glutamine', 'Succinate', 'Malate']
    metabolites = []
    for i in range(num_metabolites):
        if i < len(known_metabolites):
            metabolites.append(known_metabolites[i])
        else:
            metabolites.append(f"Metabolite_{i+1:03d}")
            
    base_expr = np.random.lognormal(mean=12.0, sigma=2.0, size=num_metabolites)
    base_expr = np.clip(base_expr, 100, None)
    
    control_vals = np.zeros((num_metabolites, 3))
    for i in range(3):
        control_vals[:, i] = base_expr * np.random.uniform(0.8, 1.2, size=num_metabolites)
        
    treat_vals = np.zeros((num_metabolites, 3))
    de_mask = np.random.rand(num_metabolites) < 0.18
    de_mask[1] = True # Glucose (down)
    de_mask[3] = True # Lactate (up)
    de_mask[6] = True # Pyruvate (up)
    
    log2_fc = np.zeros(num_metabolites)
    log2_fc[de_mask] = np.random.uniform(0.8, 2.5, size=de_mask.sum()) * np.random.choice([-1, 1], size=de_mask.sum())
    log2_fc[1] = -1.5 # Glucose
    log2_fc[3] = 2.1  # Lactate
    log2_fc[6] = 1.8  # Pyruvate
    
    for i in range(3):
        treat_vals[:, i] = base_expr * (2 ** log2_fc) * np.random.uniform(0.8, 1.2, size=num_metabolites)
        
    counts_df = pd.DataFrame(np.hstack([control_vals, treat_vals]).astype(int), index=metabolites, columns=samples)
    design_df = pd.DataFrame({
        'Sample': samples,
        'Group': ['Control', 'Control', 'Control', 'Treat', 'Treat', 'Treat']
    }).set_index('Sample')
    
    return counts_df, design_df

def run_plsda_analysis(counts_df: pd.DataFrame, design_df: pd.DataFrame) -> typing.Tuple[pd.DataFrame, list]:
    """Generates mock PLS-DA score plot components (highly distinct separation for metabolomics)."""
    np.random.seed(55)
    samples = counts_df.columns.tolist()
    
    # We mock PLS-DA component score coordinates with clean separation
    scores = []
    for sample in samples:
        is_control = design_df.loc[sample, 'Group'] == 'Control'
        if is_control:
            # Control clustered around (-5, 0)
            comp1 = np.random.normal(-6.0, 1.2)
            comp2 = np.random.normal(0.0, 1.5)
        else:
            # Treatment clustered around (+5, 0)
            comp1 = np.random.normal(6.0, 1.2)
            comp2 = np.random.normal(0.0, 1.5)
            
        scores.append({
            "Sample": sample,
            "PC1": float(comp1), # map PLSDA Component 1 to PC1 for rendering in generic scatter plot
            "PC2": float(comp2), # map PLSDA Component 2 to PC2
            "Group": design_df.loc[sample, 'Group']
        })
        
    explained_variance = [58.2, 14.5] # mock explained variance
    return pd.DataFrame(scores).set_index("Sample"), explained_variance

# =====================================================================
# 4. GENOMICS (GWAS/Variants) MODULE
# =====================================================================

def generate_mock_genomics_data(num_variants: int = 2500) -> list:
    """Generates mock GWAS SNPs variant calling results coordinates (chromosome positions & p-values)."""
    np.random.seed(777)
    
    variants = []
    
    # Generate coordinates across chromosomes 1 to 22
    chromosomes = [str(i) for i in range(1, 23)] + ['X']
    
    for i in range(num_variants):
        chrom = np.random.choice(chromosomes)
        pos = int(np.random.randint(10000, 240000000))
        
        # Log10 PValues - most are insignificant (p > 0.01 -> -log10 p < 2)
        # 1-2% are significant, and 3-4 hits are highly significant (p < 5e-8 -> -log10 p > 7.3)
        rand = np.random.rand()
        if rand < 0.0015: # Highly significant GWAS hits
            log10p = np.random.uniform(7.5, 12.0)
        elif rand < 0.02: # Mildly significant
            log10p = np.random.uniform(3.0, 6.5)
        else: # Noise
            log10p = np.random.uniform(0.0, 2.5)
            
        p_val = 10 ** (-log10p)
        
        # Variant description
        rs_id = f"rs{np.random.randint(100000, 9999999)}"
        ref = np.random.choice(['A', 'T', 'C', 'G'])
        alt = np.random.choice([x for x in ['A', 'T', 'C', 'G'] if x != ref])
        
        variants.append({
            "Variant": rs_id,
            "Chromosome": chrom,
            "Position": pos,
            "REF": ref,
            "ALT": alt,
            "PValue": float(p_val),
            "Log10PValue": float(log10p)
        })
        
    return variants

def run_gwas_analysis(variants: list) -> dict:
    """Formats and sorts variants for Manhattan plot and QQ plot visualizations."""
    # Manhattan sorting
    # Chromosome index order helper
    def chrom_key(v):
        c = v["Chromosome"]
        if c == 'X': return 23
        return int(c)
        
    sorted_variants = sorted(variants, key=chrom_key)
    
    # Calculate cumulative positions along the genome for Manhattan plotting
    chrom_offsets = {}
    current_offset = 0
    chromosomes_list = [str(i) for i in range(1, 23)] + ['X']
    
    for chrom in chromosomes_list:
        chrom_offsets[chrom] = current_offset
        current_offset += 250000000 # mock 250MB size per chromosome for simplicity
        
    manhattan_points = []
    for v in sorted_variants:
        offset = chrom_offsets.get(v["Chromosome"], 0)
        cum_pos = offset + v["Position"]
        manhattan_points.append({
            "Variant": v["Variant"],
            "Chromosome": v["Chromosome"],
            "Position": v["Position"],
            "CumulativePosition": cum_pos,
            "PValue": v["PValue"],
            "Log10PValue": v["Log10PValue"]
        })
        
    # Generate QQ-Plot coordinates (Observed vs Expected -log10 P)
    # Sort observed p-values ascending (highest -log10 p first)
    p_vals = sorted([v["PValue"] for v in variants])
    num_p = len(p_vals)
    
    qq_points = []
    for idx, p in enumerate(p_vals):
        # Expected percentile rank
        expected_p = (idx + 1) / (num_p + 1)
        expected_log10 = -np.log10(expected_p)
        observed_log10 = -np.log10(p + 1e-30)
        
        # Sample points to keep payload small in UI
        if idx % 5 == 0 or observed_log10 > 4:
            qq_points.append({
                "Expected": float(expected_log10),
                "Observed": float(observed_log10)
            })
            
    return {
        "manhattan": manhattan_points,
        "qq": qq_points
    }
