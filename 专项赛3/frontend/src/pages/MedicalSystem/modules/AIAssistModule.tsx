import React, { useState, useRef, useEffect } from 'react';

interface ChatMessage {
  id: number;
  type: 'user' | 'ai';
  content: string;
  timestamp: Date;
  file?: File;
}

const AIAssistModule: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
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
      file: selectedFile || undefined
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setSelectedFile(null);
    setIsLoading(true);

    // 模拟AI回复
    setTimeout(() => {
      let responseContent = '';
      if (userMessage.file) {
        responseContent = `我已收到您上传的文件"${userMessage.file.name}"。`;
        if (inputMessage) {
          responseContent += `\n\n关于您的问题："${inputMessage}"，我建议您考虑以下几个方面：\n\n1. 详细记录症状的发生时间和持续时间\n2. 注意观察症状的变化规律\n3. 如果症状持续或加重，建议及时就医\n\n结合您提供的文件，建议进行更深入的分析。`;
        } else {
          responseContent += '我正在分析您的文件内容，请稍等片刻。基于文件信息，我会为您提供针对性的医疗建议。';
        }
      } else {
        responseContent = `根据您的描述"${inputMessage}"，我建议您考虑以下几个方面：\n\n1. 详细记录症状的发生时间和持续时间\n2. 注意观察症状的变化规律\n3. 如果症状持续或加重，建议及时就医\n\n您还有其他需要咨询的问题吗？`;
      }

      const aiMessage: ChatMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: responseContent,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, aiMessage]);
      setIsLoading(false);
    }, 1500);
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
