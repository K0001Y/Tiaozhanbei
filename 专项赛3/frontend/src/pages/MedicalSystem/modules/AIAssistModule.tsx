import React, { useState } from 'react';

interface ChatMessage {
  id: number;
  type: 'user' | 'ai';
  content: string;
  timestamp: Date;
}

const AIAssistModule: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 1,
      type: 'ai',
      content: '您好，我是AI医疗助手。我可以帮助您进行疾病诊断、症状分析、治疗建议等。请描述您需要咨询的问题。',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;

    // 添加用户消息
    const userMessage: ChatMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    // 模拟AI回复
    setTimeout(() => {
      const aiMessage: ChatMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: `根据您的描述"${inputMessage}"，我建议您考虑以下几个方面：\n\n1. 详细记录症状的发生时间和持续时间\n2. 注意观察症状的变化规律\n3. 如果症状持续或加重，建议及时就医\n\n您还有其他需要咨询的问题吗？`,
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
    setMessages([
      {
        id: 1,
        type: 'ai',
        content: '您好，我是AI医疗助手。我可以帮助您进行疾病诊断、症状分析、治疗建议等。请描述您需要咨询的问题。',
        timestamp: new Date()
      }
    ]);
  };

  return (
    <div className="ai-assist-module">
      <div className="card">
        <div className="card-header">
          <i className="fas fa-robot"></i>
          <h3>AI智能助手</h3>
          <button className="btn btn-secondary clear-btn" onClick={clearChat}>
            <i className="fas fa-trash"></i> 清空对话
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
          </div>
          
          <div className="chat-input">
            <div className="input-group">
              <textarea
                className="form-control"
                placeholder="请输入您的问题..."
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                rows={2}
                disabled={isLoading}
              />
              <button 
                className="btn btn-primary send-btn"
                onClick={handleSendMessage}
                disabled={isLoading || !inputMessage.trim()}
              >
                <i className="fas fa-paper-plane"></i>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AIAssistModule;
