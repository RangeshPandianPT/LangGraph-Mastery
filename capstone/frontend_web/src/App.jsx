import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Play, CheckCircle, Clock, History, LayoutDashboard, GitFork, Download, RefreshCw, Activity } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { motion, AnimatePresence } from 'framer-motion';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import Mermaid from './Mermaid';
import './index.css';

const BACKEND_URL = "http://localhost:8000";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [threadId, setThreadId] = useState('react-thread-1');
  const [isPaused, setIsPaused] = useState(false);
  const [summaries, setSummaries] = useState([]);
  const [activeTab, setActiveTab] = useState('chat'); // 'chat', 'architecture', 'history'
  const [mermaidGraph, setMermaidGraph] = useState('');
  const [history, setHistory] = useState([]);
  const [checkpointId, setCheckpointId] = useState(null);
  const [finalDraft, setFinalDraft] = useState('');
  const [activeNode, setActiveNode] = useState(null);
  
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isPaused, activeTab]);

  useEffect(() => {
    fetchGraph();
  }, []);

  const fetchGraph = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/graph_mermaid`);
      if (res.ok) {
        const text = await res.text();
        setMermaidGraph(text);
      }
    } catch(e) {
      console.error("Error fetching graph", e);
    }
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/history/${threadId}`);
      if (res.ok) {
        const data = await res.json();
        setHistory(data.history || []);
      }
    } catch(e) {
      console.error(e);
    }
  };

  // Handle incoming SSE stream
  const processStream = async (payload) => {
    setIsLoading(true);
    let currentAssistantMessage = "";
    
    setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

    try {
      const response = await fetch(`${BACKEND_URL}/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          setActiveNode(null);
          break;
        }
        
        buffer += decoder.decode(value, { stream: true });
        
        const lines = buffer.split('\n');
        buffer = lines.pop(); 
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.substring(6);
            if (!dataStr) continue;
            
            try {
              const data = JSON.parse(dataStr);
              
              if (data.status === "PAUSED") {
                setIsPaused(true);
                currentAssistantMessage += "\n\n⏸️ **Paused for Human Approval.** I am ready to write the report based on the gathered research.";
                fetchState();
                setActiveNode("human_approval");
                break;
              }
              
              if (data.token) {
                if (!currentAssistantMessage.includes('### Final Report Draft')) {
                    currentAssistantMessage += `\n\n### Final Report Draft\n`;
                }
                currentAssistantMessage += data.token;
                setMessages(prev => {
                  const newMsgs = [...prev];
                  newMsgs[newMsgs.length - 1].content = currentAssistantMessage;
                  return newMsgs;
                });
                setActiveNode("writer");
                continue;
              }
              
              const nodeName = Object.keys(data)[0];
              if (nodeName && nodeName !== "status" && nodeName !== "token") {
                setActiveNode(nodeName);
              }
              
              const stateUpdate = data[nodeName];
              
              if (nodeName === "writer" && stateUpdate?.draft) {
                setFinalDraft(stateUpdate.draft);
                currentAssistantMessage += `\n\n✅ Report finalized.\n\n`;
              } else if (nodeName === "evaluator" && stateUpdate?.evaluation) {
                if (stateUpdate.evaluation === 'ACCEPT') {
                   currentAssistantMessage += `\n**[Evaluator]**: ✅ Draft Accepted!\n`;
                } else {
                   currentAssistantMessage += `\n**[Evaluator]**: 🔄 Revision Needed - ${stateUpdate.evaluation}\n`;
                }
              } else if (nodeName === "fact_checker" && stateUpdate?.fact_check_result) {
                if (stateUpdate.fact_check_result === 'ACCEPT') {
                   currentAssistantMessage += `\n**[Fact Checker]**: ✅ Facts Verified!\n`;
                } else {
                   currentAssistantMessage += `\n**[Fact Checker]**: ❌ Inaccuracies Found - ${stateUpdate.fact_check_result}\n`;
                }
              } else if (stateUpdate?.messages) {
                const msgs = stateUpdate.messages || [];
                if (msgs.length > 0) {
                  currentAssistantMessage += `\n*[{${nodeName}}]*: ${msgs[msgs.length - 1]}\n`;
                }
              }
              
              setMessages(prev => {
                const newMsgs = [...prev];
                newMsgs[newMsgs.length - 1].content = currentAssistantMessage;
                return newMsgs;
              });
              
            } catch (e) {
              // Ignore partial JSON parsing errors in stream
            }
          }
        }
      }
    } catch (e) {
      console.error("Fetch error", e);
      setMessages(prev => [...prev, { role: 'assistant', content: `❌ Error connecting to backend: ${e.message}` }]);
    } finally {
      setIsLoading(false);
      setCheckpointId(null);
      if (!isPaused) setActiveNode(null);
    }
  };

  const fetchState = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/state/${threadId}`);
      if (res.ok) {
        const data = await res.json();
        setSummaries(data.values.summaries || []);
        if (data.values.draft) setFinalDraft(data.values.draft);
        if (data.next && data.next.includes("writer")) {
          setIsPaused(true);
        } else {
          setIsPaused(false);
        }
      }
    } catch(e) {
      console.error(e);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    
    setMessages(prev => [...prev, { role: 'user', content: input }]);
    const currentInput = input;
    setInput('');
    setActiveTab('chat');
    setActiveNode("supervisor");
    
    const payload = { message: currentInput, thread_id: threadId };
    if (checkpointId) payload.checkpoint_id = checkpointId;
    
    processStream(payload);
  };

  const handleApprove = () => {
    setIsPaused(false);
    setActiveNode("writer");
    const payload = { message: "", thread_id: threadId };
    if (checkpointId) payload.checkpoint_id = checkpointId;
    processStream(payload);
  };

  const downloadDraft = () => {
    if (!finalDraft) return;
    const blob = new Blob([finalDraft], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `research_report_${threadId}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const forkResume = (cid) => {
    setCheckpointId(cid);
    setActiveTab('chat');
    setMessages(prev => [...prev, { role: 'assistant', content: `🔄 **Time Travel**: Resuming from checkpoint ${cid.substring(0,8)}...` }]);
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append("file", file);
    
    try {
      const res = await fetch(`${BACKEND_URL}/upload`, {
        method: 'POST',
        body: formData
      });
      if (res.ok) {
        alert("File uploaded and queued for ingestion successfully!");
      } else {
        alert("Failed to upload file.");
      }
    } catch(err) {
      alert("Error uploading file: " + err.message);
    }
    e.target.value = null;
  };

  return (
    <div className="app-container">
      <div className="sidebar">
        <div className="logo-section">
          <Bot size={32} className="logo-icon" />
          <h2>AI Researcher</h2>
        </div>
        
        <div className="thread-settings">
          <label>Thread ID</label>
          <input 
            type="text" 
            value={threadId} 
            onChange={(e) => setThreadId(e.target.value)} 
            placeholder="e.g., project-x"
          />
        </div>

        <nav className="nav-tabs">
          <button className={activeTab === 'chat' ? 'active' : ''} onClick={() => setActiveTab('chat')}>
            <LayoutDashboard size={18} /> Chat
          </button>
          <button className={activeTab === 'architecture' ? 'active' : ''} onClick={() => setActiveTab('architecture')}>
            <GitFork size={18} /> Architecture
          </button>
          <button className={activeTab === 'history' ? 'active' : ''} onClick={() => { setActiveTab('history'); fetchHistory(); }}>
            <History size={18} /> Time Travel
          </button>
        </nav>

        <div className="upload-section" style={{marginTop: '2rem', padding: '0 1rem'}}>
          <h4 style={{marginBottom: '0.5rem', color: '#888'}}>Knowledge Base (RAG)</h4>
          <input 
            type="file" 
            accept=".pdf,.txt" 
            onChange={handleFileUpload}
            style={{fontSize: '0.8rem', width: '100%'}}
          />
        </div>

        {finalDraft && (
          <div className="download-section">
            <button className="download-btn" onClick={downloadDraft}>
              <Download size={18} /> Download Final Report
            </button>
          </div>
        )}
      </div>

      <div className="main-content">
        <header>
          <div style={{display: 'flex', alignItems: 'center', gap: '1rem'}}>
            <h2>{activeTab === 'chat' ? 'Multi-Agent Workflow' : activeTab === 'architecture' ? 'Graph Visualization' : 'Execution History'}</h2>
            {activeNode && (
              <motion.div 
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="active-node-badge"
              >
                <Activity size={14} className="spin-icon" />
                Active: {activeNode}
              </motion.div>
            )}
          </div>
          <div className="action-buttons">
            <button onClick={fetchState} title="Sync State"><RefreshCw size={16} /> Sync</button>
          </div>
        </header>

        {activeTab === 'chat' && (
          <div className="chat-area">
            <div className="chat-container">
              {messages.length === 0 && (
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  style={{textAlign: 'center', opacity: 0.5, marginTop: '20%'}}
                >
                  <h2>What should the agents research?</h2>
                  <p>Enter a topic below to start the multi-agent workflow.</p>
                </motion.div>
              )}
              
              <AnimatePresence>
                {messages.map((msg, idx) => (
                  <motion.div 
                    key={idx} 
                    initial={{ opacity: 0, y: 15 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ type: "spring", stiffness: 200, damping: 20 }}
                    className={`message ${msg.role}`}
                  >
                    <div className="avatar">
                      {msg.role === 'user' ? <User size={20} /> : <Bot size={20} />}
                    </div>
                    <div className="message-content">
                      <ReactMarkdown
                        components={{
                          code({node, inline, className, children, ...props}) {
                            const match = /language-(\w+)/.exec(className || '')
                            return !inline && match ? (
                              <SyntaxHighlighter
                                style={vscDarkPlus}
                                language={match[1]}
                                PreTag="div"
                                className="code-block"
                                {...props}
                              >
                                {String(children).replace(/\n$/, '')}
                              </SyntaxHighlighter>
                            ) : (
                              <code className={className} {...props}>
                                {children}
                              </code>
                            )
                          }
                        }}
                      >
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
              
              {isPaused && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="message assistant"
                >
                  <div className="avatar"><CheckCircle size={20} /></div>
                  <div className="message-content approval-box">
                    <h3>Human Approval Required</h3>
                    <p>The researchers have finished gathering data. Review the summaries before approving the draft generation.</p>
                    
                    <div className="summaries-container">
                      {summaries.map((s, i) => (
                        <div key={i}><ReactMarkdown>{s}</ReactMarkdown><hr className="summary-divider"/></div>
                      ))}
                    </div>
                    
                    <button className="approval-btn" onClick={handleApprove} disabled={isLoading}>
                      <Play size={16} style={{marginRight: '0.5rem'}} /> Approve & Draft Report
                    </button>
                  </div>
                </motion.div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
            
            <form className="input-area" onSubmit={handleSubmit}>
              <input 
                type="text" 
                placeholder="Ask the agents to research a topic..." 
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={isLoading || isPaused}
              />
              <button type="submit" disabled={!input.trim() || isLoading || isPaused}>
                <Send size={20} />
              </button>
            </form>
          </div>
        )}

        {activeTab === 'architecture' && (
          <motion.div 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="tab-pane architecture-pane"
          >
            <div className="mermaid-container">
              {mermaidGraph ? <Mermaid chart={mermaidGraph} /> : <p>Loading graph...</p>}
            </div>
          </motion.div>
        )}

        {activeTab === 'history' && (
          <motion.div 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="tab-pane history-pane"
          >
            {history.length === 0 ? (
              <p style={{opacity: 0.5}}>No history found for thread '{threadId}'.</p>
            ) : (
              <div className="history-list">
                {history.map((h, i) => (
                  <div key={i} className="history-card">
                    <div className="history-header">
                      <h4>Checkpoint: {h.checkpoint_id.substring(0,8)}...</h4>
                      <button onClick={() => forkResume(h.checkpoint_id)} className="fork-btn">
                        <GitFork size={14} /> Fork & Resume
                      </button>
                    </div>
                    <div className="history-body">
                      {h.messages.map((m, j) => (
                        <div key={j} className="history-msg">{m}</div>
                      ))}
                      {h.draft && (
                        <div className="history-draft">
                          <strong>Draft Snapshot:</strong>
                          <p>{h.draft.substring(0, 100)}...</p>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </motion.div>
        )}
      </div>
    </div>
  );
}

export default App;
