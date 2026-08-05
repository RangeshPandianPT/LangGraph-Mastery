import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Play, Download, CheckCircle, Clock } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import './index.css';

const BACKEND_URL = "http://localhost:8000";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [threadId, setThreadId] = useState('react-thread-1');
  const [isPaused, setIsPaused] = useState(false);
  const [summaries, setSummaries] = useState([]);
  
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isPaused]);

  // Handle incoming SSE stream
  const processStream = async (payload) => {
    setIsLoading(true);
    let currentAssistantMessage = "";
    
    // Add a placeholder message for the assistant
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
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        
        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep the incomplete line in buffer
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.substring(6);
            if (!dataStr) continue;
            
            try {
              const data = JSON.parse(dataStr);
              
              if (data.status === "PAUSED") {
                setIsPaused(true);
                currentAssistantMessage += "\n\n⏸️ **Paused for Human Approval.** I am ready to write the report based on the gathered research.";
                fetchState(); // Fetch state to get summaries
                break;
              }
              
              const nodeName = Object.keys(data)[0];
              const stateUpdate = data[nodeName];
              
              if (nodeName === "writer" && stateUpdate.draft) {
                currentAssistantMessage += `\n\n### Final Report Draft\n${stateUpdate.draft}\n\n`;
              } else if (nodeName === "evaluator" && stateUpdate.evaluation) {
                if (stateUpdate.evaluation === 'ACCEPT') {
                   currentAssistantMessage += `\n**[Evaluator]**: ✅ Draft Accepted!\n`;
                } else {
                   currentAssistantMessage += `\n**[Evaluator]**: 🔄 Revision Needed - ${stateUpdate.evaluation}\n`;
                }
              } else if (nodeName === "fact_checker" && stateUpdate.fact_check_result) {
                if (stateUpdate.fact_check_result === 'ACCEPT') {
                   currentAssistantMessage += `\n**[Fact Checker]**: ✅ Facts Verified!\n`;
                } else {
                   currentAssistantMessage += `\n**[Fact Checker]**: ❌ Inaccuracies Found - ${stateUpdate.fact_check_result}\n`;
                }
              } else {
                const msgs = stateUpdate.messages || [];
                if (msgs.length > 0) {
                  currentAssistantMessage += `\n*[{${nodeName}}]*: ${msgs[msgs.length - 1]}\n`;
                }
              }
              
              // Update the last message
              setMessages(prev => {
                const newMsgs = [...prev];
                newMsgs[newMsgs.length - 1].content = currentAssistantMessage;
                return newMsgs;
              });
              
            } catch (e) {
              console.error("Parse error", e);
            }
          }
        }
      }
    } catch (e) {
      console.error("Fetch error", e);
      setMessages(prev => [...prev, { role: 'assistant', content: `❌ Error connecting to backend: ${e.message}` }]);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchState = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/state/${threadId}`);
      if (res.ok) {
        const data = await res.json();
        setSummaries(data.values.summaries || []);
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
    
    processStream({ message: currentInput, thread_id: threadId });
  };

  const handleApprove = () => {
    setIsPaused(false);
    processStream({ message: "", thread_id: threadId });
  };

  return (
    <div className="app-container">
      <header>
        <div className="logo-section">
          <Bot size={32} className="logo-icon" />
          <h1>Multi-Agent Researcher Pro</h1>
        </div>
        <div className="action-buttons">
          <button onClick={fetchState} title="Refresh State"><Clock size={16} /> Sync</button>
        </div>
      </header>
      
      <div className="chat-container">
        {messages.length === 0 && (
          <div style={{textAlign: 'center', opacity: 0.5, marginTop: '20%'}}>
            <h2>What should the agents research?</h2>
            <p>Enter a topic below to start the multi-agent workflow.</p>
          </div>
        )}
        
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <div className="avatar">
              {msg.role === 'user' ? <User size={20} /> : <Bot size={20} />}
            </div>
            <div className="message-content">
              <ReactMarkdown>{msg.content}</ReactMarkdown>
            </div>
          </div>
        ))}
        
        {isPaused && (
          <div className="message assistant">
            <div className="avatar"><CheckCircle size={20} /></div>
            <div className="message-content approval-box">
              <h3>Human Approval Required</h3>
              <p>The researchers have finished gathering data. Review the summaries before approving the draft generation.</p>
              
              <div style={{background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px', maxHeight: '150px', overflowY: 'auto'}}>
                {summaries.map((s, i) => (
                  <div key={i}><ReactMarkdown>{s}</ReactMarkdown><hr style={{borderColor: 'rgba(255,255,255,0.1)', margin: '0.5rem 0'}}/></div>
                ))}
              </div>
              
              <button className="approval-btn" onClick={handleApprove} disabled={isLoading}>
                <Play size={16} style={{marginRight: '0.5rem'}} /> Approve & Draft Report
              </button>
            </div>
          </div>
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
  );
}

export default App;
