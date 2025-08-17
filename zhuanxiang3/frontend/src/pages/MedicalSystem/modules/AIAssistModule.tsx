import React, { useState, useRef, useEffect } from 'react';
import { message } from 'antd';
import { apiService } from '../../../services/apiService';

interface ChatMessage {
  id: number;
  type: 'user' | 'ai';
  content: string;
  timestamp: Date;
  file?: File;
  contextMode?: string;
}

const AIAssistModule: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [contextMode, setContextMode] = useState<string>('auto');
  const [isLoading, setIsLoading] = useState(false);
  const [hasStartedChat, setHasStartedChat] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // 自动滚动到最新消息
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // 当消息变化时自动滚动
  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      console.log('文件已选择:', file.name);
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() && !selectedFile) return;

    // 如果是第一次发送消息，切换到对话模式
    if (!hasStartedChat) {
      setHasStartedChat(true);
    }

    // 添加用户消息
    const userMessage: ChatMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage || '上传了文件',
      timestamp: new Date(),
      file: selectedFile || undefined,
      contextMode: contextMode
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setSelectedFile(null);
    setIsLoading(true);

    try {
      // 根据todolist文档调用7.1 AI智能分析接口
      const formData = new FormData();
      
      // 如果有文本输入，添加query参数
      if (inputMessage.trim()) {
        formData.append('query', inputMessage.trim());
      }
      
      // 如果有文件，添加file参数
      if (selectedFile) {
        formData.append('file', selectedFile);
      }
      
      // 添加contextMode参数（根据todolist文档）
      formData.append('contextMode', contextMode);

      const response = await apiService.aiAnalyze(formData);
      
      let responseContent = '';
      if (response.success && response.data?.solution) {
        responseContent = response.data.solution;
      } else {
        responseContent = response.message || 'AI分析出现错误，请稍后重试。';
      }

      const aiMessage: ChatMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: responseContent,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, aiMessage]);
      message.success('AI分析完成');
    } catch (error) {
      console.error('AI分析请求失败:', error);
      
      // 错误时显示友好提示
      const errorMessage: ChatMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: '抱歉，AI服务暂时不可用。请检查网络连接或稍后重试。',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
      message.error(error instanceof Error ? error.message : 'AI分析失败，请重试');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setSelectedFile(null);
    setInputMessage('');
    setContextMode('auto');
    setHasStartedChat(false);
  };

  return (
    <div className="ai-assist-module">
      {!hasStartedChat ? (
        // 欢迎界面
        <div className="welcome-container">
          <div className="welcome-content">
            <div className="ai-avatar">
              <i className="fas fa-robot"></i>
            </div>
            <h1 className="welcome-title">我是 AI医疗助手，很高兴见到您！</h1>
            <p className="welcome-subtitle">我可以帮您进行疾病相关知识解答，请描述您的问题或上传相关文件。</p>
          </div>

          {/* 底部输入框 */}
          <div className="welcome-input-container">
            {/* 分析模式选择 */}
            <div className="context-mode-selector" style={{ marginBottom: '10px' }}>
              <label style={{ fontSize: '14px', color: '#666' }}>分析模式：</label>
              <select 
                value={contextMode} 
                onChange={(e) => setContextMode(e.target.value)}
                style={{ 
                  marginLeft: '8px', 
                  padding: '4px 8px', 
                  borderRadius: '4px', 
                  border: '1px solid #d9d9d9',
                  fontSize: '14px'
                }}
              >
                <option value="auto">智能模式（推荐）</option>
                <option value="simple">简单模式</option>
                <option value="comprehensive">综合模式</option>
              </select>
            </div>
            
            {selectedFile && (
              <div className="selected-file">
                <div className="file-info">
                  <i className="fas fa-file"></i>
                  <span>{selectedFile.name}</span>
                </div>
                <button 
                  className="remove-file"
                  onClick={() => setSelectedFile(null)}
                >
                  <i className="fas fa-times"></i>
                </button>
              </div>
            )}
            <div className="welcome-input-group">
              <input
                type="file"
                id="welcomeFileUpload"
                style={{ display: 'none' }}
                onChange={handleFileUpload}
                accept="image/*,.pdf,.doc,.docx,.txt"
              />
              <textarea
                className="welcome-input"
                placeholder="请输入您的问题..."
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                rows={1}
                disabled={isLoading}
              />
              <div className="welcome-input-actions">
                <button 
                  className="btn btn-outline file-btn"
                  onClick={() => document.getElementById('welcomeFileUpload')?.click()}
                  disabled={isLoading}
                  title="上传文件"
                >
                  <i className="fas fa-paperclip"></i>
                </button>
                <button 
                  className="btn btn-primary send-btn"
                  onClick={handleSendMessage}
                  disabled={isLoading || (!inputMessage.trim() && !selectedFile)}
                  title="发送消息"
                >
                  <i className="fas fa-paper-plane"></i>
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : (
        // 对话界面
        <div className="card">
          <div className="card-header">
            <div className="header-left">
              <i className="fas fa-robot"></i>
              <h3>AI智能助手</h3>
            </div>
            <button className="btn btn-outline clear-btn" onClick={clearChat}>
              <i className="fas fa-trash"></i>
              清空对话
            </button>
          </div>
          
          <div className="chat-container">
            <div className="chat-messages">
              {messages.map((message) => (
                <div key={message.id} className={`message ${message.type}`}>
                  <div className="message-avatar">
                    <i className={message.type === 'ai' ? 'fas fa-robot' : 'fas fa-user'}></i>
                  </div>
                  <div className="message-content">
                    <div className="message-text">{message.content}</div>
                    {message.file && (
                      <div className="message-file">
                        <i className="fas fa-file"></i>
                        <span>{message.file.name}</span>
                      </div>
                    )}
                    {message.type === 'user' && message.contextMode && (
                      <div className="message-context-mode" style={{ 
                        fontSize: '12px', 
                        color: '#888', 
                        marginTop: '4px' 
                      }}>
                        <i className="fas fa-cog"></i> 分析模式: {
                          message.contextMode === 'auto' ? '智能模式' :
                          message.contextMode === 'simple' ? '简单模式' :
                          message.contextMode === 'comprehensive' ? '综合模式' : message.contextMode
                        }
                      </div>
                    )}
                    <div className="message-time">
                      {message.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="message ai">
                  <div className="message-avatar">
                    <i className="fas fa-robot"></i>
                  </div>
                  <div className="message-content">
                    <div className="typing-indicator">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
            
            <div className="chat-input">
              {/* 分析模式选择 */}
              <div className="context-mode-selector" style={{ 
                marginBottom: '8px', 
                padding: '8px 12px', 
                backgroundColor: '#f8f9fa',
                borderRadius: '6px',
                border: '1px solid #e9ecef'
              }}>
                <label style={{ fontSize: '13px', color: '#666', marginRight: '8px' }}>分析模式：</label>
                <select 
                  value={contextMode} 
                  onChange={(e) => setContextMode(e.target.value)}
                  style={{ 
                    padding: '3px 6px', 
                    borderRadius: '3px', 
                    border: '1px solid #d9d9d9',
                    fontSize: '13px',
                    backgroundColor: 'white'
                  }}
                >
                  <option value="auto">智能模式（推荐）</option>
                  <option value="simple">简单模式</option>
                  <option value="comprehensive">综合模式</option>
                </select>
              </div>
              
              {selectedFile && (
                <div className="selected-file">
                  <div className="file-info">
                    <i className="fas fa-file"></i>
                    <span>{selectedFile.name}</span>
                  </div>
                  <button 
                    className="remove-file"
                    onClick={() => setSelectedFile(null)}
                  >
                    <i className="fas fa-times"></i>
                  </button>
                </div>
              )}
              <div className="input-group">
                <input
                  type="file"
                  id="fileUpload"
                  style={{ display: 'none' }}
                  onChange={handleFileUpload}
                  accept="image/*,.pdf,.doc,.docx,.txt"
                />
                <textarea
                  className="form-control"
                  placeholder="请输入您的问题..."
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  rows={2}
                  disabled={isLoading}
                />
                <div className="input-actions">
                  <button 
                    className="btn btn-outline file-btn"
                    onClick={() => document.getElementById('fileUpload')?.click()}
                    disabled={isLoading}
                    title="上传文件"
                  >
                    <i className="fas fa-paperclip"></i>
                  </button>
                  <button 
                    className="btn btn-primary send-btn"
                    onClick={handleSendMessage}
                    disabled={isLoading || (!inputMessage.trim() && !selectedFile)}
                    title="发送消息"
                  >
                    <i className="fas fa-paper-plane"></i>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AIAssistModule;
