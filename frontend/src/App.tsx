import React, { useState, useEffect, useRef } from 'react';
import { Group, Panel, Separator } from 'react-resizable-panels';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { 
  ConfigProvider, 
  Button, 
  Select, 
  Input, 
  Slider as AntdSlider, 
  Card, 
  Tag, 
  Spin, 
  Alert, 
  Space, 
  theme as antdTheme
} from 'antd';
import { 
  MessageSquare, 
  Table, 
  Compass, 
  Settings as SettingsIcon, 
  Activity, 
  Play, 
  RefreshCw, 
  User, 
  Bot, 
  ChevronRight, 
  Search, 
  Sliders, 
  FileText,
  AlertTriangle,
  Maximize2,
  Minimize2,
  FolderOpen,
  Check,
  Trash2,
  Zap,
  Sun,
  Moon
} from 'lucide-react';
import './App.css';

const API_BASE = 'http://127.0.0.1:8989';

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  steps?: string[];
}

interface GeneData {
  Gene: string;
  Mean_Control: number;
  Mean_Treat: number;
  Log2FC: number;
  PValue: number;
  PAdj: number;
}

interface PCAData {
  Sample: string;
  PC1: number;
  PC2: number;
  Group: 'Control' | 'Treat';
}

interface EnrichmentData {
  Pathway: string;
  Overlap: number;
  Pathway_Size: number;
  Genes: string[];
  PValue: number;
  Log10PValue: number;
}

interface HeatmapGene {
  Gene: string;
  Log2FC: number;
  PAdj: number;
  values: { [sample: string]: number };
}

interface HeatmapData {
  genes: string[];
  samples: string[];
  matrix: HeatmapGene[];
}

interface PPINode {
  id: string;
  x: number;
  y: number;
  Log2FC: number;
  PAdj: number;
}

interface PPIEdge {
  source: string;
  target: string;
  score: number;
}

interface PPINetwork {
  nodes: PPINode[];
  edges: PPIEdge[];
}

interface GWASVariant {
  Variant: string;
  Chromosome: string;
  Position: number;
  CumulativePosition: number;
  PValue: number;
  Log10PValue: number;
}

interface QQPoint {
  Expected: number;
  Observed: number;
}

interface GWASData {
  manhattan: GWASVariant[];
  qq: QQPoint[];
}

interface ModelConfig {
  id: string;
  label: string;
  provider: string;
  model_name: string;
  base_url: string;
  api_key: string;
}

type OmicsType = 'transcriptomics' | 'proteomics' | 'metabolomics' | 'genomics';

export default function App() {
  // Theme & Layout state
  const [isDark, setIsDark] = useState<boolean>(() => {
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  const [activeOmics, setActiveOmics] = useState<OmicsType>('transcriptomics');
  const [sidebarTab, setSidebarTab] = useState<'explorer' | 'settings'>('explorer');
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);
  const [rightTab, setRightTab] = useState<string>('volcano');
  const [wideMode, setWideMode] = useState<boolean>(false);
  const [isDeTable, setIsDeTable] = useState<boolean>(false);
  
  // Multiple Models Management State
  const [modelsList, setModelsList] = useState<ModelConfig[]>([]);
  const [activeModelId, setActiveModelId] = useState<string>('qwen-plus');
  const [isTestingMap, setIsTestingMap] = useState<Record<string, boolean>>({});
  const [testResultMap, setTestResultMap] = useState<Record<string, { status: 'success' | 'error', msg: string }>>({});

  // Form states for adding custom models
  const [presetTemplate, setPresetTemplate] = useState<string>('dashscope');
  const [newModelId, setNewModelId] = useState<string>('qwen-plus-custom');
  const [newModelLabel, setNewModelLabel] = useState<string>('My Qwen Plus Model');
  const [newModelName, setNewModelName] = useState<string>('qwen-plus');
  const [newModelUrl, setNewModelUrl] = useState<string>('https://dashscope.aliyuncs.com/compatible-mode/v1');
  const [newModelKey, setNewModelKey] = useState<string>('');
  const [formSaveStatus, setFormSaveStatus] = useState<string>('');

  // Status checks
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean | null>(null);
  
  // Data state
  const [datasetInfo, setDatasetInfo] = useState<{
    total_genes: number;
    samples: string[];
    counts: any[];
    design: any[];
  } | null>(null);

  const [analysisResults, setAnalysisResults] = useState<{
    de_results: GeneData[];
    pca: {
      coordinates: PCAData[];
      explained_variance: number[];
    };
    enrichment: EnrichmentData[];
    ppi?: PPINetwork;
    gwas?: GWASData;
    heatmap: HeatmapData;
  } | null>(null);

  // Upload & Calculation states
  const [isDataLoading, setIsDataLoading] = useState<boolean>(false);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string>('');

  // DE Thresholds
  const [pAdjCutoff, setPAdjCutoff] = useState<number>(0.05);
  const [log2fcCutoff, setLog2fcCutoff] = useState<number>(1.0);
  const [volcanoSearch, setVolcanoSearch] = useState<string>('');
  const [hoveredGene, setHoveredGene] = useState<GeneData | null>(null);
  const [topHeatmapGenes, setTopHeatmapGenes] = useState<number>(25);

  // Genomics Hover state
  const [hoveredVariant, setHoveredVariant] = useState<GWASVariant | null>(null);

  // Chat State
  const [chatMessages, setChatMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: '🧬 **欢迎使用 BioCoworker 多组学 AI 协同工作台！**\n\n我是一个类似于 QwenPaw 的 **多组学下游分析工作站**。您可以通过侧边栏切换不同的分析模块：\n- 🧬 **转录组学** (Transcriptomics)\n- 🧪 **蛋白质组学** (Proteomics)\n- ⚗️ **代谢组学** (Metabolomics)\n- 🗺️ **基因组学** (Genomics/GWAS)\n\n我已支持**导入多种大语言模型**（包括通义千问、DeepSeek、本地 Ollama 等）。您可以在设置面板进行模型切换、连接测试和添加新配置。'
    }
  ]);
  const [chatInput, setChatInput] = useState<string>('');
  const [isChatLoading, setIsChatLoading] = useState<boolean>(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // File Upload input references
  const countsFileInputRef = useRef<HTMLInputElement>(null);
  const designFileInputRef = useRef<HTMLInputElement>(null);

  // Auto Scroll Chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages, isChatLoading]);

  // Check Backend Health & Fetch Models on load
  useEffect(() => {
    checkBackendHealthyAndFetchModels();
  }, []);

  // Preset Template autofill behavior
  useEffect(() => {
    switch (presetTemplate) {
      case 'dashscope':
        setNewModelId('qwen-plus-custom');
        setNewModelLabel('Alibaba Qwen-Plus (自定义)');
        setNewModelName('qwen-plus');
        setNewModelUrl('https://dashscope.aliyuncs.com/compatible-mode/v1');
        break;
      case 'deepseek':
        setNewModelId('deepseek-chat-custom');
        setNewModelLabel('DeepSeek-V3 (自定义)');
        setNewModelName('deepseek-chat');
        setNewModelUrl('https://api.deepseek.com/v1');
        break;
      case 'ollama':
        setNewModelId('ollama-local');
        setNewModelLabel('Ollama Qwen2.5 (本地)');
        setNewModelName('qwen2.5:7b');
        setNewModelUrl('http://localhost:11434/v1');
        setNewModelKey('ollama');
        break;
      case 'zhipu':
        setNewModelId('glm-4-custom');
        setNewModelLabel('Zhipu GLM-4 (自定义)');
        setNewModelName('glm-4');
        setNewModelUrl('https://open.bigmodel.cn/api/paas/v4');
        break;
      case 'openai':
        setNewModelId('gpt-4o-custom');
        setNewModelLabel('OpenAI GPT-4o (自定义)');
        setNewModelName('gpt-4o');
        setNewModelUrl('https://api.openai.com/v1');
        break;
    }
  }, [presetTemplate]);

  const checkBackendHealthyAndFetchModels = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/status`);
      if (res.ok) {
        const data = await res.json();
        setIsBackendHealthy(true);
        setActiveModelId(data.config.active_model_id);
        if (data.dataset_loaded) {
          setActiveOmics(data.omics);
          loadMockData(1500, false, data.omics);
        }
      } else {
        setIsBackendHealthy(false);
      }
      fetchModelsList();
    } catch (e) {
      setIsBackendHealthy(false);
    }
  };

  const fetchModelsList = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/models`);
      if (res.ok) {
        const data = await res.json();
        setModelsList(data.models);
        setActiveModelId(data.active_model_id);
      }
    } catch (e) {
      console.error("Failed to load models list:", e);
    }
  };

  const handleSelectModel = async (id: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/models/select`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id })
      });
      if (res.ok) {
        const data = await res.json();
        setActiveModelId(data.active_model_id);
        
        const modelLabel = modelsList.find(m => m.id === id)?.label || id;
        setChatMessages(prev => [
          ...prev,
          {
            role: 'system',
            content: `🤖 已切换当前活跃大语言模型为：**${modelLabel}** (${id})。`
          }
        ]);
      }
    } catch (e) {
      console.error("Failed to select active model:", e);
    }
  };

  const handleDeleteModel = async (id: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/models/delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id })
      });
      if (res.ok) {
        const data = await res.json();
        setActiveModelId(data.active_model_id);
        fetchModelsList();
      } else {
        const err = await res.json();
        alert(err.detail || "删除失败");
      }
    } catch (e) {
      console.error("Failed to delete model:", e);
    }
  };

  const handleTestConnection = async (id: string) => {
    setIsTestingMap(prev => ({ ...prev, [id]: true }));
    setTestResultMap(prev => {
      const copy = { ...prev };
      delete copy[id];
      return copy;
    });

    try {
      const res = await fetch(`${API_BASE}/api/models/test-connection`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id })
      });
      if (res.ok) {
        const data = await res.json();
        setTestResultMap(prev => ({
          ...prev,
          [id]: {
            status: data.status,
            msg: data.message
          }
        }));
      } else {
        setTestResultMap(prev => ({
          ...prev,
          [id]: {
            status: 'error',
            msg: '连接请求超时或服务端崩溃。'
          }
        }));
      }
    } catch (e: any) {
      setTestResultMap(prev => ({
        ...prev,
        [id]: {
          status: 'error',
          msg: e.message || '网络请求错误'
        }
      }));
    } finally {
      setIsTestingMap(prev => ({ ...prev, [id]: false }));
    }
  };

  const handleImportModelSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newModelId.trim() || !newModelLabel.trim() || !newModelName.trim() || !newModelUrl.trim()) {
      setFormSaveStatus('请填写所有必填字段。');
      return;
    }

    setFormSaveStatus('正在导入...');
    try {
      const res = await fetch(`${API_BASE}/api/models/add`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: newModelId,
          label: newModelLabel,
          provider: presetTemplate,
          model_name: newModelName,
          base_url: newModelUrl,
          api_key: newModelKey
        })
      });

      if (res.ok) {
        const data = await res.json();
        setFormSaveStatus('导入并激活成功！');
        setActiveModelId(data.active_model_id);
        fetchModelsList();
        
        setNewModelKey('');
        setTimeout(() => setFormSaveStatus(''), 3000);
        
        setChatMessages(prev => [
          ...prev,
          {
            role: 'system',
            content: `📥 成功导入大模型配置 **${newModelLabel}**，并已自动激活为当前智能体模型！`
          }
        ]);
      } else {
        setFormSaveStatus('导入模型配置失败。');
      }
    } catch (e) {
      setFormSaveStatus('接口请求发生异常。');
    }
  };

  // Handle active omics switch
  const handleOmicsChange = (omics: OmicsType) => {
    setActiveOmics(omics);
    setDatasetInfo(null);
    setAnalysisResults(null);
    setUploadError('');
    
    if (omics === 'genomics') {
      setRightTab('manhattan');
    } else {
      setRightTab('volcano');
    }

    setChatMessages(prev => [
      ...prev,
      {
        role: 'system',
        content: `🔄 已成功切换至 **${
          omics === 'transcriptomics' ? '转录组学 (Transcriptomics)' :
          omics === 'proteomics' ? '蛋白质组学 (Proteomics)' :
          omics === 'metabolomics' ? '代谢组学 (Metabolomics)' : '基因组学 (Genomics/GWAS)'
        }** 分析模块。请加载该组学的数据文件以启动分析图表。`
      }
    ]);
  };

  // Load Mock Data
  const loadMockData = async (numGenes: number = 1500, triggerAlert: boolean = true, targetOmics: OmicsType = activeOmics) => {
    setIsDataLoading(true);
    setUploadError('');
    try {
      const res = await fetch(`${API_BASE}/api/load-mock?omics=${targetOmics}&num_genes=${numGenes}`, {
        method: 'POST'
      });
      if (!res.ok) throw new Error(`后端生成模拟数据失败 (omics=${targetOmics})`);
      const data = await res.json();
      
      if (targetOmics === 'genomics') {
        setDatasetInfo({
          total_genes: data.total_genes,
          samples: [],
          counts: data.counts,
          design: []
        });
      } else {
        setDatasetInfo({
          total_genes: data.total_genes,
          samples: data.samples,
          counts: data.counts,
          design: data.design
        });
      }
      
      await triggerAnalysis(data.total_genes, triggerAlert, targetOmics);
    } catch (e: any) {
      setUploadError(e.message || "加载示例数据失败。");
    } finally {
      setIsDataLoading(false);
    }
  };

  // Trigger Backend Analysis
  const triggerAnalysis = async (totalItems: number, notifyChat: boolean = true, targetOmics: OmicsType = activeOmics) => {
    setIsAnalyzing(true);
    try {
      const formData = new FormData();
      formData.append('p_adj_cutoff', pAdjCutoff.toString());
      formData.append('log2fc_cutoff', log2fcCutoff.toString());

      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        body: formData
      });
      
      if (!res.ok) throw new Error("多组学分析管道计算失败。");
      const data = await res.json();
      
      if (targetOmics === 'genomics') {
        setAnalysisResults({
          de_results: [],
          pca: { coordinates: [], explained_variance: [] },
          heatmap: { genes: [], samples: [], matrix: [] },
          enrichment: [],
          gwas: data.gwas
        });
        
        if (notifyChat) {
          const sigHits = data.gwas.manhattan.filter((v: GWASVariant) => v.PValue <= 5e-8);
          setChatMessages(prev => [
            ...prev,
            {
              role: 'system',
              content: `📊 **GWAS 关联分析完成！**\n- **总检测 SNPs 位点**: ${totalItems} 个\n- **达到全基因组显著性阈值 (p <= 5e-8) 的位点**: ${sigHits.length} 个\n\n*右侧工作台已成功渲染 GWAS 曼哈顿图 (Manhattan Plot) 及 QQ 关联分布图。*`
            }
          ]);
        }
      } else {
        setAnalysisResults({
          de_results: data.de_results,
          pca: data.pca,
          enrichment: data.enrichment,
          ppi: data.ppi,
          heatmap: data.heatmap
        });

        if (notifyChat) {
          const sigGenes = data.de_results.filter((g: GeneData) => Math.abs(g.Log2FC) >= log2fcCutoff && g.PAdj <= pAdjCutoff);
          const upGenes = sigGenes.filter((g: GeneData) => g.Log2FC > 0);
          const downGenes = sigGenes.filter((g: GeneData) => g.Log2FC < 0);
          
          let subContent = "";
          if (targetOmics === 'proteomics') {
            subContent = `\n- **蛋白互作网络节点**: ${data.ppi.nodes?.length || 0} 个，互作边数: ${data.ppi.edges?.length || 0} 条`;
          }
          
          setChatMessages(prev => [
            ...prev,
            {
              role: 'system',
              content: `📊 **分析管道运行完毕！**\n- **分析特征项数**: ${totalItems}\n- **显著差异表达项目**: ${sigGenes.length} 个 (上调: ${upGenes.length}, 下调: ${downGenes.length})${subContent}\n\n*右侧工作台已同步渲染相应图表。*`
            }
          ]);
        }
      }
    } catch (e: any) {
      setUploadError(e.message || "运行分析管道失败。");
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Custom File upload handler
  const handleCustomUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    const countsFile = countsFileInputRef.current?.files?.[0];
    const designFile = designFileInputRef.current?.files?.[0];

    if (!countsFile || (activeOmics !== 'genomics' && !isDeTable && !designFile)) {
      setUploadError("请提供完整的数据文件！");
      return;
    }

    setIsDataLoading(true);
    setUploadError('');
    try {
      const formData = new FormData();
      formData.append('counts_file', countsFile);
      if (designFile && !isDeTable) {
        formData.append('design_file', designFile);
      }
      formData.append('omics', activeOmics);
      formData.append('is_de_table', String(isDeTable));

      const res = await fetch(`${API_BASE}/api/upload`, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errorDetail = await res.json();
        throw new Error(errorDetail.detail || "解析文件失败。");
      }

      const data = await res.json();
      setDatasetInfo({
        total_genes: data.total_genes,
        samples: data.samples,
        counts: data.counts,
        design: data.design
      });

      await triggerAnalysis(data.total_genes, true);
    } catch (e: any) {
      setUploadError(e.message || "解析自定义文件发生错误，请检查格式。");
    } finally {
      setIsDataLoading(false);
    }
  };

  // Send message to AI Agent
  const sendChatMessage = async (msgText: string) => {
    if (!msgText.trim() || isChatLoading) return;

    setChatMessages(prev => [...prev, { role: 'user', content: msgText }]);
    setChatInput('');
    setIsChatLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msgText })
      });

      if (!res.ok) throw new Error("智能体响应失败。");
      const data = await res.json();
      
      setChatMessages(prev => [
        ...prev, 
        { 
          role: data.status === 'error' ? 'system' : 'assistant', 
          content: data.reply 
        }
      ]);
    } catch (e: any) {
      setChatMessages(prev => [
        ...prev,
        { role: 'system', content: `⚠️ AI Agent 响应发生故障: ${e.message || "网络请求失败"}` }
      ]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const handleChatSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendChatMessage(chatInput);
  };

  // Sidebar toggle behavior
  const handleActivityBtnClick = (tab: 'explorer' | 'settings') => {
    if (sidebarOpen && sidebarTab === tab) {
      setSidebarOpen(false);
    } else {
      setSidebarTab(tab);
      setSidebarOpen(true);
    }
  };

  // Volcano SVG Plot drawing
  const renderVolcanoPlot = () => {
    if (!analysisResults?.de_results) return null;

    const width = 600;
    const height = 400;
    const padding = 50;

    const points = analysisResults.de_results.map(g => {
      const log10p = -Math.log10(g.PAdj + 1e-10);
      return {
        ...g,
        x: g.Log2FC,
        y: log10p
      };
    });

    const xVals = points.map(p => p.x);
    const yVals = points.map(p => p.y);
    
    const maxAbsX = Math.max(Math.max(...xVals.map(Math.abs)), 3.0);
    const xMin = -maxAbsX;
    const xMax = maxAbsX;
    
    const yMin = 0;
    const yMax = Math.max(Math.max(...yVals), 8);

    const getX = (val: number) => padding + ((val - xMin) / (xMax - xMin)) * (width - 2 * padding);
    const getY = (val: number) => height - padding - ((val - yMin) / (yMax - yMin)) * (height - 2 * padding);

    const yThresholdVal = -Math.log10(pAdjCutoff);
    const thresholdY = getY(yThresholdVal);
    const leftThresholdX = getX(-log2fcCutoff);
    const rightThresholdX = getX(log2fcCutoff);

    return (
      <div className="flex flex-col items-center">
        <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} className="overflow-visible select-none max-w-2xl">
          <rect x={padding} y={padding} width={width - 2 * padding} height={height - 2 * padding} fill="var(--bg-tertiary)" opacity="0.4" rx="6" />
          
          <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="var(--border-color)" strokeWidth="1.5" />
          <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="var(--border-color)" strokeWidth="1.5" />

          {/* Threshold guides */}
          <line x1={padding} y1={thresholdY} x2={width - padding} y2={thresholdY} stroke="var(--color-primary)" strokeWidth="1" strokeDasharray="3,3" opacity="0.6" />
          <line x1={leftThresholdX} y1={padding} x2={leftThresholdX} y2={height - padding} stroke="var(--color-primary)" strokeWidth="1" strokeDasharray="3,3" opacity="0.6" />
          <line x1={rightThresholdX} y1={padding} x2={rightThresholdX} y2={height - padding} stroke="var(--color-primary)" strokeWidth="1" strokeDasharray="3,3" opacity="0.6" />

          <line x1={getX(0)} y1={padding} x2={getX(0)} y2={height - padding} stroke="var(--border-color)" strokeWidth="1" strokeDasharray="2,2" />

          {/* Scatter points */}
          {points.map((p, idx) => {
            const isSignificant = p.PAdj <= pAdjCutoff && Math.abs(p.Log2FC) >= log2fcCutoff;
            const isUp = isSignificant && p.Log2FC > 0;
            const isDown = isSignificant && p.Log2FC < 0;
            
            let color = 'var(--text-tertiary)';
            if (isUp) color = 'var(--color-danger)';
            if (isDown) color = 'var(--color-primary)';

            const isSearchMatch = volcanoSearch && p.Gene.toLowerCase().includes(volcanoSearch.toLowerCase());
            const radius = isSearchMatch ? 8 : (hoveredGene?.Gene === p.Gene ? 8 : 4.5);
            const strokeColor = isSearchMatch ? 'var(--text-primary)' : 'none';
            const strokeWidth = isSearchMatch ? 2 : 0;
            
            return (
              <circle
                key={idx}
                cx={getX(p.x)}
                cy={getY(p.y)}
                r={radius}
                fill={color}
                stroke={strokeColor}
                strokeWidth={strokeWidth}
                opacity={isSearchMatch ? 1.0 : (hoveredGene?.Gene === p.Gene ? 1.0 : 0.65)}
                className="cursor-pointer transition-all duration-100 hover:opacity-100"
                onMouseEnter={() => setHoveredGene(p)}
                onMouseLeave={() => setHoveredGene(null)}
              />
            );
          })}

          <text x={width / 2} y={height - 12} textAnchor="middle" fill="var(--text-secondary)" fontSize="11" fontWeight="bold">
            Log2 Fold Change
          </text>
          <text x={18} y={height / 2} textAnchor="middle" transform={`rotate(-90, 18, ${height / 2})`} fill="var(--text-secondary)" fontSize="11" fontWeight="bold">
            -log10(FDR Adjusted P-Value)
          </text>

          <g transform={`translate(${padding + 12}, ${padding + 15})`}>
            <circle cx={5} cy={4.5} r={4.5} fill="var(--color-danger)" />
            <text x={15} y={8} fontSize="10" fill="var(--text-secondary)" fontWeight="bold">Upregulated ({points.filter(p => p.PAdj <= pAdjCutoff && p.Log2FC >= log2fcCutoff).length})</text>

            <circle cx={5} cy={20} r={4.5} fill="var(--color-primary)" />
            <text x={15} y={23} fontSize="10" fill="var(--text-secondary)" fontWeight="bold">Downregulated ({points.filter(p => p.PAdj <= pAdjCutoff && p.Log2FC <= -log2fcCutoff).length})</text>
          </g>
        </svg>

        {/* Hovered tooltip */}
        <div className="mt-4 p-3 bg-secondary border border-color rounded-xl shadow-sm h-18 w-full max-w-md flex items-center justify-center">
          {hoveredGene ? (
            <div className="text-xs text-secondary grid grid-cols-2 gap-x-6 w-full">
              <div><strong>基因/项目名称:</strong> <span className="text-primary font-bold">{hoveredGene.Gene}</span></div>
              <div><strong>Log2 Fold Change:</strong> <span className={hoveredGene.Log2FC > 0 ? "text-danger font-bold" : "text-primary font-bold"}>{hoveredGene.Log2FC > 0 ? '+' : ''}{hoveredGene.Log2FC.toFixed(3)}</span></div>
              <div><strong>原始 P 值:</strong> {hoveredGene.PValue.toExponential(3)}</div>
              <div><strong>FDR 值 (PAdj):</strong> {hoveredGene.PAdj.toExponential(3)}</div>
            </div>
          ) : (
            <span className="text-xs text-tertiary italic">在火山图上悬停任意圆点查看详细属性</span>
          )}
        </div>
      </div>
    );
  };

  // PCA / PLS-DA Score plot drawing
  const renderPCAPlot = () => {
    if (!analysisResults?.pca) return null;

    const width = 600;
    const height = 400;
    const padding = 60;

    const { coordinates, explained_variance } = analysisResults.pca;

    const xVals = coordinates.map(c => c.PC1);
    const yVals = coordinates.map(c => c.PC2);

    const xMin = Math.min(...xVals) * 1.3;
    const xMax = Math.max(...xVals) * 1.3;
    const yMin = Math.min(...yVals) * 1.3;
    const yMax = Math.max(...yVals) * 1.3;

    const getX = (val: number) => padding + ((val - xMin) / (xMax - xMin)) * (width - 2 * padding);
    const getY = (val: number) => height - padding - ((val - yMin) / (yMax - yMin)) * (height - 2 * padding);

    return (
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} className="overflow-visible select-none max-w-2xl mx-auto">
        <rect x={padding} y={padding} width={width - 2 * padding} height={height - 2 * padding} fill="var(--bg-tertiary)" opacity="0.4" rx="6" />
        
        <line x1={getX(0)} y1={padding} x2={getX(0)} y2={height - padding} stroke="var(--border-color)" strokeWidth="1" strokeDasharray="3,3" />
        <line x1={padding} y1={getY(0)} x2={width - padding} y2={getY(0)} stroke="var(--border-color)" strokeWidth="1" strokeDasharray="3,3" />

        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="var(--border-color)" strokeWidth="1.5" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="var(--border-color)" strokeWidth="1.5" />

        {coordinates.map((c, idx) => {
          const cx = getX(c.PC1);
          const cy = getY(c.PC2);
          const isControl = c.Group === 'Control';
          
          return (
            <g key={idx} className="cursor-pointer">
              {isControl ? (
                <circle cx={cx} cy={cy} r={8.5} fill="var(--color-primary)" opacity="0.9" stroke="var(--bg-secondary)" strokeWidth="1.5" />
              ) : (
                <rect x={cx - 7.5} y={cy - 7.5} width={15} height={15} fill="var(--color-danger)" opacity="0.9" stroke="var(--bg-secondary)" strokeWidth="1.5" rx="2" />
              )}
              <text x={cx} y={cy - 12} textAnchor="middle" fontSize="10" fontWeight="bold" fill="var(--text-secondary)">
                {c.Sample}
              </text>
            </g>
          );
        })}

        <text x={width / 2} y={height - 15} textAnchor="middle" fill="var(--text-secondary)" fontSize="11" fontWeight="bold">
          {activeOmics === 'metabolomics' ? `PLS-DA Comp 1` : `PC1`} ({explained_variance[0]?.toFixed(1)}% 解释变异度)
        </text>
        <text x={20} y={height / 2} textAnchor="middle" transform={`rotate(-90, 20, ${height / 2})`} fill="var(--text-secondary)" fontSize="11" fontWeight="bold">
          {activeOmics === 'metabolomics' ? `PLS-DA Comp 2` : `PC2`} ({explained_variance[1]?.toFixed(1)}% 解释变异度)
        </text>

        <g transform={`translate(${width - padding - 95}, ${padding + 15})`}>
          <circle cx={5} cy={5} r={5.5} fill="var(--color-primary)" />
          <text x={15} y={9} fontSize="10" fill="var(--text-secondary)" fontWeight="bold">Control (对照组)</text>

          <rect x={0} y={18} width={11} height={11} fill="var(--color-danger)" rx="1.5" />
          <text x={15} y={27} fontSize="10" fill="var(--text-secondary)" fontWeight="bold">Treated (处理组)</text>
        </g>
      </svg>
    );
  };

  // PPI Interaction network graph drawing (Proteomics specific)
  const renderPPINetwork = () => {
    if (!analysisResults?.ppi) return null;

    const width = 600;
    const height = 400;
    const { nodes, edges } = analysisResults.ppi;

    return (
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} className="overflow-visible select-none max-w-2xl mx-auto">
        <rect x={10} y={10} width={width - 20} height={height - 20} fill="var(--bg-tertiary)" opacity="0.25" rx="8" />
        
        {/* Draw Edges */}
        {edges.map((e, idx) => {
          const srcNode = nodes.find(n => n.id === e.source);
          const tgtNode = nodes.find(n => n.id === e.target);
          if (!srcNode || !tgtNode) return null;
          
          return (
            <line
              key={idx}
              x1={srcNode.x}
              y1={srcNode.y}
              x2={tgtNode.x}
              y2={tgtNode.y}
              stroke="var(--text-tertiary)"
              strokeWidth={e.score * 3}
              opacity={0.4}
            />
          );
        })}

        {/* Draw Nodes */}
        {nodes.map((n, idx) => {
          const color = n.Log2FC > 0 ? 'var(--color-danger)' : 'var(--color-primary)';
          return (
            <g key={idx} className="cursor-pointer">
              <circle
                cx={n.x}
                cy={n.y}
                r={12}
                fill={color}
                stroke="var(--bg-secondary)"
                strokeWidth={1.5}
                opacity={0.9}
              />
              <text
                x={n.x}
                y={n.y + 24}
                textAnchor="middle"
                fontSize="9"
                fontWeight="bold"
                fill="var(--text-primary)"
              >
                {n.id}
              </text>
              <text
                x={n.x}
                y={n.y + 3.5}
                textAnchor="middle"
                fontSize="8"
                fontWeight="bold"
                fill="#ffffff"
              >
                {n.Log2FC > 0 ? '+' : ''}{n.Log2FC.toFixed(0)}
              </text>
            </g>
          );
        })}
      </svg>
    );
  };

  // Genomics Manhattan GWAS Plot drawing (Publication-grade alternating chromosomes)
  const renderManhattanPlot = () => {
    if (!analysisResults?.gwas) return null;

    const width = 800;
    const height = 400;
    const padding = 55;

    const { manhattan } = analysisResults.gwas;

    // Find domains
    const xVals = manhattan.map(m => m.CumulativePosition);
    const yVals = manhattan.map(m => m.Log10PValue);

    const xMax = Math.max(...xVals, 1) * 1.02;
    const yMax = Math.max(Math.max(...yVals), 9.0);

    // Genome-wide significance line (5e-8 -> -log10 p = 7.3)
    const gwasThresholdY = getY(7.3);

    // Alternating colors for chromosomes
    const chromColors: Record<string, string> = {
      '1': '#3b82f6', '3': '#3b82f6', '5': '#3b82f6', '7': '#3b82f6', '9': '#3b82f6', '11': '#3b82f6', '13': '#3b82f6', '15': '#3b82f6', '17': '#3b82f6', '19': '#3b82f6', '21': '#3b82f6',
      '2': '#64748b', '4': '#64748b', '6': '#64748b', '8': '#64748b', '10': '#64748b', '12': '#64748b', '14': '#64748b', '16': '#64748b', '18': '#64748b', '20': '#64748b', '22': '#64748b',
      'X': '#8b5cf6'
    };

    function getX(val: number) { return padding + (val / xMax) * (width - 2 * padding); }
    function getY(val: number) { return height - padding - (val / yMax) * (height - 2 * padding); }

    return (
      <div className="flex flex-col items-center">
        <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} className="overflow-visible select-none max-w-3xl">
          <rect x={padding} y={padding} width={width - 2 * padding} height={height - 2 * padding} fill="var(--bg-tertiary)" opacity="0.3" rx="6" />
          
          {/* Grid ticks */}
          <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="var(--border-color)" strokeWidth="1.5" />
          <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="var(--border-color)" strokeWidth="1.5" />

          {/* Red Significance Line */}
          <line x1={padding} y1={gwasThresholdY} x2={width - padding} y2={gwasThresholdY} stroke="var(--color-danger)" strokeWidth="1.5" strokeDasharray="3,3" />
          <text x={width - padding - 160} y={gwasThresholdY - 6} fill="var(--color-danger)" fontSize="9" fontWeight="bold">全基因组显著性红线 (p = 5e-8)</text>

          {/* Draw scatter dots */}
          {manhattan.map((p, idx) => {
            const isSignificant = p.Log10PValue >= 7.301;
            const baseColor = chromColors[p.Chromosome] || 'var(--text-tertiary)';
            const color = isSignificant ? 'var(--color-primary)' : baseColor;

            const isHovered = hoveredVariant?.Variant === p.Variant;
            const radius = isSignificant ? 5 : (isHovered ? 6 : 2.5);
            
            return (
              <circle
                key={idx}
                cx={getX(p.CumulativePosition)}
                cy={getY(p.Log10PValue)}
                r={radius}
                fill={color}
                opacity={isSignificant ? 1.0 : (isHovered ? 1.0 : 0.6)}
                className="cursor-pointer hover:opacity-100"
                onMouseEnter={() => setHoveredVariant(p)}
                onMouseLeave={() => setHoveredVariant(null)}
              />
            );
          })}

          <text x={width / 2} y={height - 12} textAnchor="middle" fill="var(--text-secondary)" fontSize="11" fontWeight="bold">
            染色体物理分布坐标 (Chromosomes 1-22, X)
          </text>
          <text x={18} y={height / 2} textAnchor="middle" transform={`rotate(-90, 18, ${height / 2})`} fill="var(--text-secondary)" fontSize="11" fontWeight="bold">
            -log10(P-Value)
          </text>
        </svg>

        {/* Manhattan Variant details card */}
        <div className="mt-4 p-3 bg-secondary border border-color rounded-xl shadow-sm h-18 w-full max-w-md flex items-center justify-center">
          {hoveredVariant ? (
            <div className="text-xs text-secondary grid grid-cols-2 gap-x-6 w-full">
              <div><strong>SNPs 标记:</strong> <span className="text-primary font-bold">{hoveredVariant.Variant}</span></div>
              <div><strong>染色体位置:</strong> <span className="text-primary font-bold">Chr {hoveredVariant.Chromosome}: {hoveredVariant.Position.toLocaleString()}</span></div>
              <div><strong>GWAS P-value:</strong> {hoveredVariant.PValue.toExponential(4)}</div>
              <div><strong>统计显著度 (-log10p):</strong> <span className={hoveredVariant.Log10PValue >= 7.3 ? "text-accent font-bold" : ""}>{hoveredVariant.Log10PValue.toFixed(3)}</span></div>
            </div>
          ) : (
            <span className="text-xs text-tertiary italic">在曼哈顿图上悬停基因突变圆点以查看显著度数据</span>
          )}
        </div>
      </div>
    );
  };

  // Genomics QQ Plot drawing
  const renderQQPlot = () => {
    if (!analysisResults?.gwas) return null;

    const width = 500;
    const height = 400;
    const padding = 50;
    const { qq } = analysisResults.gwas;

    // Find domains
    const expectedVals = qq.map(q => q.Expected);
    const observedVals = qq.map(q => q.Observed);

    const maxVal = Math.max(Math.max(...expectedVals), Math.max(...observedVals), 6) * 1.05;

    const getCoord = (val: number) => padding + (val / maxVal) * (width - 2 * padding);
    const getX = (val: number) => getCoord(val);
    const getY = (val: number) => height - padding - (val / maxVal) * (height - 2 * padding);

    return (
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} className="overflow-visible select-none max-w-lg mx-auto">
        <rect x={padding} y={padding} width={width - 2 * padding} height={height - 2 * padding} fill="var(--bg-tertiary)" opacity="0.4" rx="6" />
        
        {/* Diagonal identity line y = x */}
        <line x1={getX(0)} y1={getY(0)} x2={getX(maxVal)} y2={getY(maxVal)} stroke="var(--text-tertiary)" strokeWidth="1.5" strokeDasharray="4,4" />

        {/* Axes */}
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="var(--border-color)" strokeWidth="1.5" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="var(--border-color)" strokeWidth="1.5" />

        {/* Points */}
        {qq.map((p, idx) => (
          <circle
            key={idx}
            cx={getX(p.Expected)}
            cy={getY(p.Observed)}
            r={3}
            fill="var(--color-primary)"
            opacity={0.8}
          />
        ))}

        <text x={width / 2} y={height - 12} textAnchor="middle" fill="var(--text-secondary)" fontSize="11" fontWeight="bold">
          期望的 -log10(P-Value)
        </text>
        <text x={18} y={height / 2} textAnchor="middle" transform={`rotate(-90, 18, ${height / 2})`} fill="var(--text-secondary)" fontSize="11" fontWeight="bold">
          观测的 -log10(P-Value)
        </text>
      </svg>
    );
  };

  const getHeatmapColor = (zScore: number) => {
    if (zScore < 0) {
      const ratio = Math.min(Math.abs(zScore) / 2.5, 1);
      return `rgba(59, 130, 246, ${ratio})`; 
    } else {
      const ratio = Math.min(zScore / 2.5, 1);
      return `rgba(239, 68, 68, ${ratio})`;
    }
  };

  return (
    <ConfigProvider
      theme={{
        algorithm: isDark ? antdTheme.darkAlgorithm : antdTheme.defaultAlgorithm,
        token: {
          colorPrimary: '#FF7F16', // QwenPaw Orange
          borderRadius: 8,
        }
      }}
    >
      <div className={`workspace-container ${wideMode ? 'wide-mode' : ''} text-sm ${isDark ? 'dark-mode' : ''}`}>
        
        {/* ========================================================= */}
        {/* NARROW LEFT ACTIVITY BAR */}
        {/* ========================================================= */}
        <div className="activity-bar">
          <button 
            className={`activity-btn ${sidebarOpen && sidebarTab === 'explorer' ? 'active' : ''}`}
            onClick={() => handleActivityBtnClick('explorer')}
            title="Explorer (工作空间文件与导入)"
          >
            <FolderOpen className="w-5 h-5" />
          </button>
          <button 
            className={`activity-btn ${sidebarOpen && sidebarTab === 'settings' ? 'active' : ''}`}
            onClick={() => handleActivityBtnClick('settings')}
            title="Settings (AI 对接参数配置)"
          >
            <SettingsIcon className="w-5 h-5" />
          </button>
          
          <div className="activity-spacer" />

          {/* Theme toggler */}
          <button 
            className="activity-btn mb-1" 
            onClick={() => setIsDark(!isDark)}
            title="Toggle Light/Dark Theme"
          >
            {isDark ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-slate-600" />}
          </button>
          
          <div 
            className={`w-3 h-3 rounded-full border border-secondary shadow-sm ${isBackendHealthy === true ? 'bg-emerald-500' : 'bg-rose-500'}`}
            title={isBackendHealthy === true ? 'API Connected' : 'API Disconnected'}
            onClick={checkBackendHealthyAndFetchModels}
            style={{ cursor: 'pointer', marginBottom: 4 }}
          />
        </div>

        <Group orientation="horizontal" style={{ flexGrow: 1, height: '100%', minWidth: 0 }}>
          {/* ========================================================= */}
          {/* PANEL 1: LEFT SIDEBAR */}
          {/* ========================================================= */}
          {sidebarOpen && (
            <>
              <Panel id="left" defaultSize="18%" className="panel-left">
            
            {sidebarTab === 'explorer' && (
              <>
                <div className="panel-header">
                  <h3>
                    <FolderOpen className="w-4 h-4 text-accent" style={{ color: 'var(--color-primary)' }} />
                    <span>生信工作空间</span>
                  </h3>
                </div>
                <div className="panel-body">
                  
                  {/* Dynamic Omics Switcher */}
                  <div className="sidebar-section">
                    <h4>组学分析模块 (Omics)</h4>
                    <Select 
                      value={activeOmics} 
                      onChange={(val) => handleOmicsChange(val as OmicsType)}
                      style={{ width: '100%' }}
                      options={[
                        { value: 'transcriptomics', label: '转录组学 (RNA-Seq)' },
                        { value: 'proteomics', label: '蛋白质组学 (Proteomics)' },
                        { value: 'metabolomics', label: '代谢组学 (Metabolomics)' },
                        { value: 'genomics', label: '基因组学 (GWAS/SNPs)' }
                      ]}
                    />
                  </div>

                  {/* File list cache */}
                  <div className="sidebar-section">
                    <h4>文件浏览器 (Explorer)</h4>
                    {datasetInfo ? (
                      <div className="flex flex-col gap-1">
                        <div className="file-card active">
                          <FileText className="w-3.5 h-3.5 icon-left" />
                          <span className="name-label">
                            {activeOmics === 'genomics' ? 'variants_gwas_table.csv' : 'expression_counts.csv'}
                          </span>
                          <Tag color="success">已载入</Tag>
                        </div>
                        {activeOmics !== 'genomics' && (
                          <div className="file-card active">
                            <FileText className="w-3.5 h-3.5 icon-left" />
                            <span className="name-label">design_group.csv</span>
                            <Tag color="success">已载入</Tag>
                          </div>
                        )}
                        {analysisResults && (
                          <div className="file-card" onClick={() => setRightTab('raw_preview')}>
                            <FileText className="w-3.5 h-3.5 icon-left" style={{ color: 'var(--color-purple)' }} />
                            <span className="name-label">
                              {activeOmics === 'genomics' ? 'gwas_gsem_results.tsv' : 'de_results.tsv'}
                            </span>
                            <Tag color="purple">分析完成</Tag>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="text-xs text-tertiary italic p-3 text-center bg-tertiary rounded border border-dashed border-color">
                        暂无活动文件，请载入数据。
                      </div>
                    )}
                  </div>

                  {/* Import Area */}
                  <div className="sidebar-section">
                    <h4>数据导入 (Importer)</h4>
                    <div className="flex flex-col gap-2">
                      <Button
                        type="primary"
                        icon={isDataLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                        onClick={() => loadMockData(activeOmics === 'genomics' ? 2500 : 1500, true)}
                        loading={isDataLoading}
                        style={{ width: '100%' }}
                      >
                        加载 {activeOmics === 'genomics' ? 'GWAS 关联' : '模拟表达'} 示例
                      </Button>

                      <form onSubmit={handleCustomUpload} className="flex flex-col gap-2.5 p-3 bg-tertiary border border-color rounded text-[11px]">
                        <div className="font-bold text-secondary">导入自定义表格:</div>
                        
                        {activeOmics !== 'genomics' && (
                          <div className="flex items-center gap-1.5 mb-1">
                            <input
                              type="checkbox"
                              id="isDeTableCheckbox"
                              checked={isDeTable}
                              onChange={(e) => setIsDeTable(e.target.checked)}
                              style={{ cursor: 'pointer' }}
                            />
                            <label htmlFor="isDeTableCheckbox" style={{ cursor: 'pointer', fontWeight: 600 }}>
                              直接导入已分析的差异表达表 (DESeq2/edgeR)
                            </label>
                          </div>
                        )}

                        <div>
                          <label className="block text-[10px] text-tertiary mb-0.5">
                            {activeOmics === 'genomics'
                              ? 'GWAS P值位置表 (CSV/TSV):'
                              : isDeTable
                              ? '已分析的差异表达结果表 (CSV/TSV):'
                              : '表达量原始 Counts 矩阵 (CSV/TSV):'}
                          </label>
                          <input type="file" ref={countsFileInputRef} accept=".csv,.tsv,.txt" className="w-full text-[10px]" />
                        </div>

                        {activeOmics !== 'genomics' && !isDeTable && (
                          <div>
                            <label className="block text-[10px] text-tertiary mb-0.5">样本分组设计表 (CSV/TSV):</label>
                            <input type="file" ref={designFileInputRef} accept=".csv,.tsv,.txt" className="w-full text-[10px]" />
                          </div>
                        )}

                        <Button
                          htmlType="submit"
                          type="dashed"
                          loading={isDataLoading}
                          style={{ width: '100%', fontSize: '11px', height: '28px', color: 'var(--color-purple)', borderColor: 'var(--color-purple)' }}
                        >
                          上传并运行分析
                        </Button>
                      </form>
                    </div>
                  </div>

                </div>
              </>
            )}

            {sidebarTab === 'settings' && (
              <>
                <div className="panel-header">
                  <h3>
                    <SettingsIcon className="w-4 h-4 text-accent" style={{ color: 'var(--color-primary)' }} />
                    <span>AI 模型管理</span>
                  </h3>
                </div>
                <div className="panel-body">
                  
                  {/* Active Model Selector */}
                  <div className="sidebar-section">
                    <h4>当前活跃模型 (Active LLM)</h4>
                    <Select
                      value={activeModelId}
                      onChange={(val) => handleSelectModel(val)}
                      style={{ width: '100%' }}
                      options={modelsList.map(m => ({ value: m.id, label: m.label }))}
                    />
                  </div>

                  {/* Loaded Models config manager */}
                  <div className="sidebar-section">
                    <h4>已导入模型提供商</h4>
                    <div className="flex flex-col gap-2 max-h-[220px] overflow-y-auto pr-1">
                      {modelsList.map((model) => {
                        const isActive = model.id === activeModelId;
                        const isTesting = isTestingMap[model.id];
                        const testRes = testResultMap[model.id];
                        
                        return (
                          <div 
                            key={model.id} 
                            className={`p-2.5 rounded border text-[11px] flex flex-col gap-1.5 transition-all ${
                              isActive ? 'border-orange-500/60 bg-accent' : 'border-color bg-tertiary'
                            }`}
                          >
                            <div className="flex justify-between items-start">
                              <div>
                                <div className="font-bold text-primary">{model.label}</div>
                                <div className="text-[10px] text-tertiary font-mono">{model.model_name}</div>
                              </div>
                              {isActive && (
                                <span className="text-[9px] bg-orange-500 text-white font-extrabold px-1 py-0.5 rounded leading-none">
                                  ACTIVE
                                </span>
                              )}
                            </div>

                            <div className="text-[9px] text-tertiary truncate" title={model.base_url}>
                              {model.base_url}
                            </div>

                            {testRes && (
                              <div className={`text-[9px] font-semibold leading-tight ${
                                testRes.status === 'success' ? 'text-emerald-500' : 'text-rose-500'
                              }`}>
                                {testRes.msg}
                              </div>
                            )}

                            <div className="flex gap-1.5 mt-1 border-t border-color/40 pt-1.5 justify-end">
                              <Button
                                size="small"
                                icon={isTesting ? <RefreshCw className="w-2.5 h-2.5 animate-spin" /> : <Zap className="w-2.5 h-2.5 text-amber-500" />}
                                onClick={() => handleTestConnection(model.id)}
                                disabled={isTesting}
                                style={{ fontSize: '10px', height: '22px' }}
                              >
                                测试连接
                              </Button>
                              {!isActive && (
                                <Button
                                  size="small"
                                  onClick={() => handleSelectModel(model.id)}
                                  style={{ fontSize: '10px', height: '22px' }}
                                >
                                  激活
                                </Button>
                              )}
                              <Button
                                danger
                                size="small"
                                disabled={modelsList.length <= 1}
                                onClick={() => handleDeleteModel(model.id)}
                                style={{ height: '22px', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '0 8px' }}
                              >
                                <Trash2 className="w-3 h-3" />
                              </Button>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Import New Model */}
                  <div className="sidebar-section">
                    <h4>导入大语言模型 (Importer)</h4>
                    <form onSubmit={handleImportModelSubmit} className="flex flex-col gap-2.5 p-3 bg-tertiary border border-color rounded text-[11px]">
                      <div className="input-field-group">
                        <label>选择模板快速填充:</label>
                        <Select
                          value={presetTemplate}
                          onChange={(val) => setPresetTemplate(val)}
                          style={{ width: '100%' }}
                          options={[
                            { value: 'dashscope', label: 'Alibaba 通义千问 (DashScope)' },
                            { value: 'deepseek', label: 'DeepSeek 官方 API' },
                            { value: 'ollama', label: 'Local Ollama 本地接口' },
                            { value: 'zhipu', label: '智谱 GLM API' },
                            { value: 'openai', label: 'OpenAI 兼容接口 (Custom)' }
                          ]}
                        />
                      </div>

                      <div className="input-field-group">
                        <label>配置唯一 ID (Config ID):</label>
                        <Input
                          size="small"
                          value={newModelId}
                          onChange={(e) => setNewModelId(e.target.value)}
                          placeholder="e.g. qwen-plus-new"
                        />
                      </div>

                      <div className="input-field-group">
                        <label>显示标签 (Model Label):</label>
                        <Input
                          size="small"
                          value={newModelLabel}
                          onChange={(e) => setNewModelLabel(e.target.value)}
                          placeholder="e.g. My Qwen Model"
                        />
                      </div>

                      <div className="input-field-group">
                        <label>模型参数名称 (Model Name):</label>
                        <Input
                          size="small"
                          value={newModelName}
                          onChange={(e) => setNewModelName(e.target.value)}
                          placeholder="e.g. qwen-plus"
                        />
                      </div>

                      <div className="input-field-group">
                        <label>API Endpoint (Base URL):</label>
                        <Input
                          size="small"
                          value={newModelUrl}
                          onChange={(e) => setNewModelUrl(e.target.value)}
                          placeholder="https://api.domain.com/v1"
                        />
                      </div>

                      <div className="input-field-group">
                        <label>API 访问密匙 (Secret Key):</label>
                        <Input.Password
                          size="small"
                          value={newModelKey}
                          onChange={(e) => setNewModelKey(e.target.value)}
                          placeholder="API authorization key"
                        />
                      </div>

                      <Button
                        type="primary"
                        htmlType="submit"
                        icon={<Check className="w-3.5 h-3.5" />}
                        style={{ width: '100%', fontSize: '11px', height: '28px', marginTop: '6px' }}
                      >
                        导入并设置为激活
                      </Button>
                      
                      {formSaveStatus && (
                        <span className="text-[10px] text-emerald-500 font-bold text-center mt-1 block">
                          {formSaveStatus}
                        </span>
                      )}
                    </form>
                  </div>

                </div>
              </>
            )}

              </Panel>
              <Separator className="resize-handle" />
            </>
          )}

          {/* ========================================================= */}
          {/* PANEL 2: MIDDLE CHAT CONSOLE */}
          {/* ========================================================= */}
          <Panel id="middle" defaultSize={30} minSize={20} className="panel-middle">
          
          <div className="panel-header">
            <h3>
              <MessageSquare className="w-4 h-4 text-accent" style={{ color: 'var(--color-primary)' }} />
              <span>AI 多组学专家助手对话</span>
            </h3>
            <Tag color="orange" style={{ fontWeight: 'bold', fontFamily: 'monospace' }}>
              {activeOmics.toUpperCase()} CHAT
            </Tag>
          </div>

          {uploadError && (
            <div className="mx-4 mt-3">
              <Alert 
                message={uploadError} 
                type="error" 
                showIcon 
                closable 
                onClose={() => setUploadError('')}
                style={{ fontSize: '12px' }}
              />
            </div>
          )}

          {/* Message Feed (shadcn-chat structure) */}
          <div className="chat-messages-scroll">
            {chatMessages.map((msg, idx) => (
              <div key={idx} className={`message-bubble ${msg.role}`}>
                <div className={`message-avatar ${msg.role}`}>
                  {msg.role === 'user' ? <User className="w-4 h-4" /> : msg.role === 'system' ? <AlertTriangle className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                </div>
                <div className="message-content-wrapper">
                  <div className="message-meta-header">
                    {msg.role === 'user' ? '生信研究员 (User)' : msg.role === 'system' ? '系统提示 (System)' : 'BioCoworker 智能体 (AI)'}
                  </div>
                  <div className="message-text">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                  </div>
                </div>
              </div>
            ))}
            {isChatLoading && (
              <div className="message-bubble assistant">
                <div className="message-avatar assistant">
                  <Bot className="w-4 h-4 animate-pulse" />
                </div>
                <div className="message-content-wrapper">
                  <div className="message-meta-header">BioCoworker 智能体 (AI)</div>
                  <div className="message-text">
                    <Space size="middle">
                      <Spin size="small" />
                      <span>正在分析数据集中，关联多组学知识背景...</span>
                    </Space>
                  </div>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Chips suggestions */}
          {datasetInfo && (
            <div className="suggestion-pill-container">
              {activeOmics === 'genomics' ? (
                <>
                  <button onClick={() => sendChatMessage("帮我把GWAS的分析画出来，并找出全基因组最显著关联的前五个rs号点")} className="suggestion-pill" disabled={isChatLoading}>
                    🗺️ 查看GWAS显著突变位点
                  </button>
                  <button onClick={() => sendChatMessage("解释一下GWAS曼哈顿图和QQ图的关系，这个数据的Q-Q线是否有偏差")} className="suggestion-pill" disabled={isChatLoading}>
                    📈 解析曼哈顿图与QQ线
                  </button>
                </>
              ) : (
                <>
                  <button onClick={() => sendChatMessage(`帮我运行${activeOmics === 'proteomics'?'蛋白质':'代谢'}组学差异分析，找出差异前15项`)} className="suggestion-pill" disabled={isChatLoading}>
                    📊 运行统计差异表达
                  </button>
                  <button onClick={() => sendChatMessage(`展示${activeOmics === 'proteomics'?'蛋白互作PPI网络':'通路富集分析'}并解释受调控的通路`)} className="suggestion-pill" disabled={isChatLoading}>
                    🧬 查看通路网状机制
                  </button>
                </>
              )}
            </div>
          )}

          {/* Chat input box */}
          <form onSubmit={handleChatSubmit} className="chat-input-form-bar">
            <Input
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder={datasetInfo ? "向 AI 专家提问，或指导其计算..." : "加载左侧示例后，在此向助手提问..."}
              className="chat-input-text"
              disabled={isChatLoading}
              onPressEnter={handleChatSubmit}
            />
            <Button
              type="primary"
              onClick={handleChatSubmit}
              disabled={!chatInput.trim() || isChatLoading}
              icon={<ChevronRight className="w-3.5 h-3.5" />}
              style={{ display: 'flex', alignItems: 'center' }}
            >
              发送
            </Button>
          </form>

          </Panel>

          <Separator className="resize-handle" />

          {/* ========================================================= */}
          {/* PANEL 3: PLUGGABLE MULTI-OMICS WORKSPACE */}
          {/* ========================================================= */}
          <Panel id="right" defaultSize={52} minSize={30} className="panel-right">
          
          {/* Tabs header using Ant Design 5 premium styling */}
          <div className="desk-tabs-bar">
            {activeOmics !== 'genomics' ? (
              <>
                <button className={`desk-tab-item ${rightTab === 'volcano' ? 'active' : ''}`} onClick={() => setRightTab('volcano')}>
                  <Activity className="w-3.5 h-3.5" />
                  <span>volcano_plot.svg</span>
                </button>
                {analysisResults && (
                  <>
                    <button className={`desk-tab-item ${rightTab === 'heatmap' ? 'active' : ''}`} onClick={() => setRightTab('heatmap')}>
                      <Table className="w-3.5 h-3.5" />
                      <span>abundance_heatmap.svg</span>
                    </button>
                    <button className={`desk-tab-item ${rightTab === 'pca' ? 'active' : ''}`} onClick={() => setRightTab('pca')}>
                      <Compass className="w-3.5 h-3.5" />
                      <span>{activeOmics === 'metabolomics' ? 'plsda_scores.svg' : 'pca_scores.svg'}</span>
                    </button>
                    {activeOmics === 'proteomics' ? (
                      <button className={`desk-tab-item ${rightTab === 'ppi' ? 'active' : ''}`} onClick={() => setRightTab('ppi')}>
                        <Compass className="w-3.5 h-3.5" style={{ color: 'var(--color-primary)' }} />
                        <span>ppi_interaction_network.svg</span>
                      </button>
                    ) : (
                      <button className={`desk-tab-item ${rightTab === 'enrichment' ? 'active' : ''}`} onClick={() => setRightTab('enrichment')}>
                        <Sliders className="w-3.5 h-3.5" />
                        <span>pathway_enrichment.svg</span>
                      </button>
                    )}
                    <button className={`desk-tab-item ${rightTab === 'raw_preview' ? 'active' : ''}`} onClick={() => setRightTab('raw_preview')}>
                      <FileText className="w-3.5 h-3.5" />
                      <span>abundance_stats_table.csv</span>
                    </button>
                  </>
                )}
              </>
            ) : (
              <>
                <button className={`desk-tab-item ${rightTab === 'manhattan' ? 'active' : ''}`} onClick={() => setRightTab('manhattan')}>
                  <Activity className="w-3.5 h-3.5" />
                  <span>gwas_manhattan_plot.svg</span>
                </button>
                {analysisResults && (
                  <>
                    <button className={`desk-tab-item ${rightTab === 'qqplot' ? 'active' : ''}`} onClick={() => setRightTab('qqplot')}>
                      <Compass className="w-3.5 h-3.5" />
                      <span>gwas_qq_plot.svg</span>
                    </button>
                    <button className={`desk-tab-item ${rightTab === 'raw_preview' ? 'active' : ''}`} onClick={() => setRightTab('raw_preview')}>
                      <FileText className="w-3.5 h-3.5" />
                      <span>snps_association_results.csv</span>
                    </button>
                  </>
                )}
              </>
            )}

            <div style={{ flexGrow: 1 }} />

            <button 
              onClick={() => setWideMode(!wideMode)}
              className="desk-tab-item border-l border-color border-r-0 hover:bg-tertiary"
              title="Toggle Wide Mode"
              style={{ borderRight: 0 }}
            >
              {wideMode ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
            </button>
          </div>

          {/* Pluggable Toolbar controls using Ant Design Slider and Search */}
          <div className="px-4 py-2 border-b border-color bg-secondary flex items-center justify-between text-xs flex-shrink-0 gap-4">
            {(rightTab === 'volcano' && activeOmics !== 'genomics') && (
              <>
                <div className="flex gap-8 flex-grow items-center">
                  <div className="flex items-center gap-3">
                    <span className="font-semibold text-secondary whitespace-nowrap">FDR (P-Adj) 阈值:</span>
                    <div style={{ width: 100 }}>
                      <AntdSlider
                        min={0.001} max={0.15} step={0.001} value={pAdjCutoff}
                        onChange={(val) => {
                          setPAdjCutoff(val);
                          if (datasetInfo) triggerAnalysis(datasetInfo.total_genes, false);
                        }}
                      />
                    </div>
                    <span className="font-mono text-primary font-bold">{pAdjCutoff}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="font-semibold text-secondary whitespace-nowrap">Log2FC 阈值:</span>
                    <div style={{ width: 100 }}>
                      <AntdSlider
                        min={0.5} max={2.5} step={0.1} value={log2fcCutoff}
                        onChange={(val) => {
                          setLog2fcCutoff(val);
                          if (datasetInfo) triggerAnalysis(datasetInfo.total_genes, false);
                        }}
                      />
                    </div>
                    <span className="font-mono text-primary font-bold">±{log2fcCutoff}</span>
                  </div>
                </div>
                <div className="relative w-40 flex-shrink-0">
                  <Input
                    size="small"
                    placeholder="搜索特征分子..."
                    value={volcanoSearch}
                    onChange={(e) => setVolcanoSearch(e.target.value)}
                    prefix={<Search className="w-3.5 h-3.5 text-secondary mr-1" />}
                  />
                </div>
              </>
            )}

            {rightTab === 'heatmap' && (
              <div className="flex items-center gap-2">
                <span className="font-semibold text-secondary">展示差异特征数 (Top):</span>
                <Input
                  size="small"
                  type="number"
                  min={10}
                  max={100}
                  value={topHeatmapGenes}
                  onChange={(e) => setTopHeatmapGenes(Math.max(10, parseInt(e.target.value) || 25))}
                  style={{ width: 80 }}
                />
              </div>
            )}

            {rightTab === 'pca' && (
              <div className="text-secondary italic">
                {activeOmics === 'metabolomics' ? 'PLS-DA 降维投影图：评估样品代谢谱组别分离情况' : 'PCA 降维投影图：评估样本重复变异与聚类分布'}
              </div>
            )}

            {rightTab === 'ppi' && (
              <div className="text-secondary italic">
                串联蛋白质互作网络 (STRING PPI)：基于差异蛋白质生成的拓扑互作图
              </div>
            )}

            {rightTab === 'enrichment' && (
              <div className="text-secondary italic">
                通路富集图：统计受调控的代谢物/基因的宏观通路变化
              </div>
            )}

            {rightTab === 'manhattan' && (
              <div className="text-secondary italic">
                GWAS Manhattan Plot：全基因组 SNPs 位点与表型性状的关联分析图
              </div>
            )}

            {rightTab === 'qqplot' && (
              <div className="text-secondary italic">
                GWAS QQ-Plot：检测关联统计的期望分布与观测分布是否偏离
              </div>
            )}

            {rightTab === 'raw_preview' && (
              <div className="text-secondary italic font-semibold">
                统计学分析特征项表格明细
              </div>
            )}
          </div>

          {/* Workspace Display Area */}
          <div className="panel-body flex-grow overflow-y-auto" style={{ padding: '20px' }}>
            
            {/* 1. Volcano view */}
            {rightTab === 'volcano' && (
              <Card className="visual-desk-card max-w-3xl mx-auto flex flex-col justify-center py-4">
                {!analysisResults ? (
                  <div className="h-[380px] flex flex-col gap-2 items-center justify-center text-secondary text-center p-4">
                    <Activity className="w-12 h-12 text-tertiary animate-pulse" />
                    <div className="font-bold text-sm">暂无统计差分表达分析数据</div>
                    <div className="text-xs text-tertiary max-w-sm">请点击左侧面板“加载示例数据集”按钮，系统会自动运行差分分析并渲染火山图。</div>
                  </div>
                ) : isAnalyzing ? (
                  <div className="h-[380px] flex items-center justify-center text-secondary">
                    <Spin tip="正在计算差分表达指标..." />
                  </div>
                ) : (
                  renderVolcanoPlot()
                )}
              </Card>
            )}

            {/* 2. Heatmap view */}
            {rightTab === 'heatmap' && analysisResults && (
              <Card className="visual-desk-card max-w-3xl mx-auto">
                <div className="flex flex-col gap-2 overflow-x-auto">
                  <div className="flex text-[10px] text-secondary font-bold border-b border-color pb-2 mb-2">
                    <span className="w-24 flex-shrink-0">ID (特异性分子)</span>
                    <div className="flex justify-between flex-grow">
                      {analysisResults.heatmap.samples.map(s => (
                        <span key={s} className="w-14 text-center overflow-hidden text-ellipsis whitespace-nowrap">
                          {s.split('_')[0] === 'Control' ? 'Ctrl' : 'Treat'}<br />{s.split('_')[1]}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="flex flex-col gap-1.5 max-h-[450px] overflow-y-auto pr-1">
                    {analysisResults.heatmap.matrix.slice(0, topHeatmapGenes).map((row, idx) => (
                      <div key={idx} className="flex items-center text-xs hover:bg-tertiary/40 py-0.5 rounded">
                        <span className="w-24 flex-shrink-0 font-bold text-primary font-mono">{row.Gene}</span>
                        <div className="flex justify-between flex-grow">
                          {analysisResults.heatmap.samples.map(s => {
                            const val = row.values[s];
                            return (
                              <div 
                                key={s} 
                                className="w-14 h-6.5 rounded flex items-center justify-center text-[10px] text-white font-mono font-bold"
                                style={{ 
                                  backgroundColor: getHeatmapColor(val),
                                  color: Math.abs(val) > 1.2 ? '#fff' : 'var(--text-primary)'
                                }}
                                title={`特征 ID: ${row.Gene}\n样本 ID: ${s}\nZ-Score: ${val.toFixed(3)}`}
                              >
                                {val.toFixed(2)}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </Card>
            )}

            {/* 3. PCA / PLS-DA score plot view */}
            {rightTab === 'pca' && analysisResults && (
              <Card className="visual-desk-card max-w-3xl mx-auto flex flex-col justify-center">
                {renderPCAPlot()}
              </Card>
            )}

            {/* 4. Protein interaction network view */}
            {rightTab === 'ppi' && analysisResults && (
              <Card className="visual-desk-card max-w-3xl mx-auto flex flex-col justify-center">
                {renderPPINetwork()}
              </Card>
            )}

            {/* 5. Pathway Enrichment view */}
            {rightTab === 'enrichment' && analysisResults && (
              <div className="double-panel-grid max-w-4xl mx-auto">
                <Card className="visual-desk-card flex flex-col justify-between" title="显著通路排序 (-log10 P-Value)">
                  {analysisResults.enrichment.length === 0 ? (
                    <div className="h-60 flex items-center justify-center text-tertiary italic">无显著富集通路</div>
                  ) : (
                    <div className="flex flex-col gap-4">
                      {analysisResults.enrichment.slice(0, 8).map((item, idx) => {
                        const maxVal = Math.max(...analysisResults.enrichment.map(e => e.Log10PValue), 4);
                        const pct = (item.Log10PValue / maxVal) * 100;
                        return (
                          <div key={idx} className="flex flex-col gap-0.5 text-xs">
                            <div className="flex justify-between font-semibold text-secondary font-sans">
                              <span>{item.Pathway}</span>
                              <span className="font-mono text-[10px]">{item.Overlap}/{item.Pathway_Size}</span>
                            </div>
                            <div className="h-5 w-full bg-tertiary rounded relative overflow-hidden border border-color">
                              <div 
                                className="h-full rounded animate-fade-in"
                                style={{
                                  width: `${pct}%`,
                                  backgroundImage: 'linear-gradient(90deg, var(--color-primary) 0%, var(--color-purple) 100%)'
                                }}
                              />
                              <span className="absolute right-2 top-1 text-[9px] font-mono font-bold text-primary">
                                p = {item.PValue.toExponential(2)}
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </Card>

                <Card className="visual-desk-card flex flex-col" title="通路覆盖特征项目">
                  <div className="overflow-y-auto max-h-[360px] flex flex-col gap-3">
                    {analysisResults.enrichment.map((item, idx) => (
                      <div key={idx} className="p-3 bg-tertiary rounded border border-color text-xs">
                        <div className="flex justify-between font-bold text-primary mb-2">
                          <span>{item.Pathway}</span>
                          <span className="text-[10px] text-secondary font-mono">p = {item.PValue.toExponential(2)}</span>
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {item.Genes.map(g => (
                            <span key={g} className="px-2 py-0.5 bg-secondary border border-color rounded text-[10px] font-mono text-secondary">
                              {g}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              </div>
            )}

            {/* 6. Genomics Manhattan plot view */}
            {rightTab === 'manhattan' && (
              <Card className="visual-desk-card max-w-3xl mx-auto flex flex-col justify-center py-4">
                {!analysisResults?.gwas ? (
                  <div className="h-[380px] flex flex-col gap-2 items-center justify-center text-secondary text-center p-4">
                    <Activity className="w-12 h-12 text-tertiary animate-pulse" />
                    <div className="font-bold text-sm">暂无 GWAS 分析数据</div>
                    <div className="text-xs text-tertiary max-w-sm">请点击左侧面板“加载 GWAS 示例”按钮，系统会自动载入位点坐标并绘制曼哈顿图。</div>
                  </div>
                ) : isAnalyzing ? (
                  <div className="h-[380px] flex items-center justify-center text-secondary">
                    <Spin tip="正在绘制全基因组 Manhattan 图..." />
                  </div>
                ) : (
                  renderManhattanPlot()
                )}
              </Card>
            )}

            {/* 7. Genomics QQ plot view */}
            {rightTab === 'qqplot' && analysisResults?.gwas && (
              <Card className="visual-desk-card max-w-3xl mx-auto flex flex-col justify-center">
                {renderQQPlot()}
              </Card>
            )}

            {/* 8. Raw data preview spreadsheet view */}
            {rightTab === 'raw_preview' && (
              <Card className="visual-desk-card max-w-4xl mx-auto">
                {activeOmics === 'genomics' ? (
                  analysisResults?.gwas && (
                    <div className="overflow-x-auto max-h-[460px] border border-color rounded-xl">
                      <table className="w-full text-left text-xs border-collapse">
                        <thead>
                          <tr className="bg-tertiary border-b border-color text-secondary">
                            <th className="p-3 font-bold">SNPs 标记 (Variant)</th>
                            <th className="p-3 font-bold text-center">染色体 (Chr)</th>
                            <th className="p-3 font-bold text-right">基因组物理坐标 (Pos)</th>
                            <th className="p-3 font-bold text-center">REF</th>
                            <th className="p-3 font-bold text-center">ALT</th>
                            <th className="p-3 font-bold text-right">P-Value</th>
                          </tr>
                        </thead>
                        <tbody>
                          {datasetInfo?.counts.map((row: any, idx: number) => (
                            <tr key={idx} className="border-b border-color hover:bg-tertiary/40">
                              <td className="p-3 font-bold text-primary font-mono">{row.Variant}</td>
                              <td className="p-3 text-center text-secondary font-semibold">{row.Chromosome}</td>
                              <td className="p-3 text-right text-secondary font-mono">{row.Position.toLocaleString()}</td>
                              <td className="p-3 text-center text-secondary font-mono">{row.REF}</td>
                              <td className="p-3 text-center text-secondary font-mono">{row.ALT}</td>
                              <td className={`p-3 text-right font-mono font-bold ${row.PValue <= 5e-8 ? "text-rose-500" : "text-secondary"}`}>
                                {row.PValue.toExponential(4)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )
                ) : (
                  analysisResults?.de_results && (
                    <div className="overflow-x-auto max-h-[460px] border border-color rounded-xl">
                      <table className="w-full text-left text-xs border-collapse">
                        <thead>
                          <tr className="bg-tertiary border-b border-color text-secondary">
                            <th className="p-3 font-bold">特征 ID (Feature Name)</th>
                            <th className="p-3 font-bold text-right">Log2FC</th>
                            <th className="p-3 font-bold text-right">Mean Control</th>
                            <th className="p-3 font-bold text-right">Mean Treat</th>
                            <th className="p-3 font-bold text-right">P-Value</th>
                            <th className="p-3 font-bold text-right">FDR (PAdj)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {analysisResults.de_results
                            .filter(g => Math.abs(g.Log2FC) >= log2fcCutoff && g.PAdj <= pAdjCutoff)
                            .map((row, idx) => (
                              <tr key={idx} className="border-b border-color hover:bg-tertiary/40">
                                <td className="p-3 font-bold text-primary font-mono">{row.Gene}</td>
                                <td className={`p-3 text-right font-bold ${row.Log2FC > 0 ? "text-rose-500" : "text-blue-500"}`}>
                                  {row.Log2FC > 0 ? '+' : ''}{row.Log2FC.toFixed(3)}
                                </td>
                                <td className="p-3 text-right text-secondary font-mono">{row.Mean_Control.toFixed(3)}</td>
                                <td className="p-3 text-right text-secondary font-mono">{row.Mean_Treat.toFixed(3)}</td>
                                <td className="p-3 text-right text-secondary font-mono">{row.PValue.toExponential(3)}</td>
                                <td className="p-3 text-right text-secondary font-mono">{row.PAdj.toExponential(3)}</td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  )
                )}
              </Card>
            )}

          </div>
          </Panel>
        </Group>

      </div>
    </ConfigProvider>
  );
}
