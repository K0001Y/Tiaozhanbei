import React, { useState, useRef, useEffect } from 'react';
import { message } from 'antd';

interface SupplementMessage {
  id: number;
  type: 'user' | 'ai';
  content: string;
  timestamp: Date;
  file?: File;
}

const DiagnosisModule: React.FC = () => {
  const [imageDescription, setImageDescription] = useState('');
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [analysisResult, setAnalysisResult] = useState<string>('');
  const [showSupplement, setShowSupplement] = useState(false);
  const [supplementMessages, setSupplementMessages] = useState<SupplementMessage[]>([]);
  const [supplementInput, setSupplementInput] = useState('');
  const [supplementFile, setSupplementFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isSupplementing, setIsSupplementing] = useState(false);
  const supplementMessagesEndRef = useRef<HTMLDivElement>(null);

  // 自动滚动到最新消息
  const scrollToBottomSupplement = () => {
    supplementMessagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // 当补充消息变化时自动滚动
  useEffect(() => {
    if (showSupplement) {
      scrollToBottomSupplement();
    }
  }, [supplementMessages, isSupplementing, showSupplement]);

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setUploadedImage(file);
      console.log('图片已上传:', file.name);
    }
  };

  const handleSupplementFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSupplementFile(file);
      console.log('补充文件已上传:', file.name);
    }
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const files = event.dataTransfer.files;
    if (files.length > 0) {
      setUploadedImage(files[0]);
      console.log('图片已拖放上传:', files[0].name);
    }
  };

  const handleAnalyze = async () => {
    if (!uploadedImage) {
      message.warning('请先上传图片');
      return;
    }
    
    setIsAnalyzing(true);
    console.log('开始分析图片:', uploadedImage.name);
    console.log('图片描述:', imageDescription);
    
    // 模拟API调用
    setTimeout(() => {
      const mockResult = "根据图像分析，观察到以下特征：\n\n1. 皮肤色泽略显苍白，可能提示气血不足\n2. 舌苔偏厚，舌质偏红，提示可能存在湿热体质\n3. 眼部周围略有暗沉，可能与睡眠不足或肾功能相关\n\n建议结合问诊结果进行综合判断。";
      setAnalysisResult(mockResult);
      setShowSupplement(true);
      setIsAnalyzing(false);
    }, 2000);
  };

  const handleSupplement = async () => {
    if (!supplementInput.trim() && !supplementFile) {
      message.warning('请输入补充信息或上传文件');
      return;
    }

    // 添加用户消息
    const userMessage: SupplementMessage = {
      id: Date.now(),
      type: 'user',
      content: supplementInput || '上传了文件',
      timestamp: new Date(),
      file: supplementFile || undefined
    };

    setSupplementMessages(prev => [...prev, userMessage]);
    setSupplementInput('');
    setSupplementFile(null);
    setIsSupplementing(true);

    // 模拟AI回复
    setTimeout(() => {
      const aiMessage: SupplementMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: `感谢您的补充信息。结合新的资料，我对之前的分析进行以下更新：\n\n根据您提供的补充信息，建议关注以下几个方面：\n1. 注意观察症状的变化规律\n2. 建议适当调整作息时间\n3. 可考虑中医调理方案\n\n更新后的综合分析已保存到您的望诊结果中。`,
        timestamp: new Date()
      };
      
      setSupplementMessages(prev => [...prev, aiMessage]);
      
      // 更新分析结果
      const updatedResult = analysisResult + "\n\n【补充分析】\n根据补充信息，进一步确认了上述判断，建议结合个人体质特点制定针对性的调理方案。";
      setAnalysisResult(updatedResult);
      
      setIsSupplementing(false);
    }, 1500);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSupplement();
    }
  };

  const clearSupplement = () => {
    setSupplementMessages([]);
  };

  return (
    <div className="diagnosis-module">
      <div className="card">
        <div className="card-header">
          <i className="fas fa-eye"></i>
          <h3>辅助望诊</h3>
        </div>
        <div className="card-body">
          <div 
            className="image-upload"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => document.getElementById('file-input')?.click()}
          >
            <i className="fas fa-cloud-upload-alt"></i>
            <h4>导入图片进行望诊分析</h4>
            <p>点击或拖放图片到此处上传 (支持JPG, PNG格式)</p>
            {uploadedImage && (
              <div className="uploaded-file-info">
                <p>已上传: {uploadedImage.name}</p>
              </div>
            )}
          </div>
          
          <input
            id="file-input"
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={handleImageUpload}
          />
          
          <div className="form-group" style={{ marginTop: '25px' }}>
            <label>补充说明</label>
            <textarea 
              className="form-control" 
              rows={3} 
              placeholder="补充图片说明或者症状信息"
              value={imageDescription}
              onChange={(e) => setImageDescription(e.target.value)}
            />
          </div>
          
          <div className="btn-group">
            <button 
              className="btn btn-primary" 
              onClick={handleAnalyze}
              disabled={isAnalyzing}
            >
              {isAnalyzing ? (
                <>
                  <i className="fas fa-spinner fa-spin"></i> 分析中...
                </>
              ) : (
                <>
                  <i className="fas fa-diagnoses"></i> 开始望诊
                </>
              )}
            </button>
          </div>

          {/* 望诊结果展示 */}
          {analysisResult && (
            <div className="analysis-result" style={{ marginTop: '25px' }}>
              <div className="result-header">
                <h4><i className="fas fa-chart-line"></i> 望诊分析结果</h4>
              </div>
              <div className="result-content">
                <pre style={{ whiteSpace: 'pre-wrap', fontSize: '14px', lineHeight: '1.6' }}>
                  {analysisResult}
                </pre>
              </div>
            </div>
          )}

          {/* 补充对话框 */}
          {showSupplement && (
            <div className="supplement-section" style={{ marginTop: '25px' }}>
              <div className="supplement-header">
                <h4><i className="fas fa-comments"></i> 补充信息</h4>
                <button className="btn clear-chat-btn btn-sm" onClick={clearSupplement}>
                  <i className="fas fa-trash"></i> 清空
                </button>
              </div>
              
              {/* 对话记录 */}
              {supplementMessages.length > 0 && (
                <div className="chat-messages">
                  {supplementMessages.map((message) => (
                    <div key={message.id} className={`message ${message.type}`}>
                      <div className="message-avatar">
                        <i className={message.type === 'ai' ? 'fas fa-robot' : 'fas fa-user'}></i>
                      </div>
                      <div className="message-content">
                        <div className="message-text">
                          {message.content}
                          {message.file && (
                            <div className="message-file">
                              <i className="fas fa-file"></i> {message.file.name}
                            </div>
                          )}
                        </div>
                        <div className="message-time">
                          {message.timestamp.toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                  ))}
                  {isSupplementing && (
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
                  <div ref={supplementMessagesEndRef} />
                </div>
              )}

              {/* 补充输入框 */}
              <div className="supplement-input">
                <div className="input-group">
                  <textarea
                    className="form-control"
                    placeholder="请输入补充信息或上传相关文件..."
                    value={supplementInput}
                    onChange={(e) => setSupplementInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    rows={2}
                    disabled={isSupplementing}
                  />
                  <div className="input-actions">
                    <input
                      id="supplement-file-input"
                      type="file"
                      accept="image/*,.pdf,.doc,.docx,.txt"
                      style={{ display: 'none' }}
                      onChange={handleSupplementFileUpload}
                    />
                    <button 
                      className="btn btn-outline-secondary"
                      onClick={() => document.getElementById('supplement-file-input')?.click()}
                      disabled={isSupplementing}
                      title="上传文件"
                    >
                      <i className="fas fa-paperclip"></i>
                    </button>
                    <button 
                      className="btn btn-primary"
                      onClick={handleSupplement}
                      disabled={isSupplementing || (!supplementInput.trim() && !supplementFile)}
                    >
                      <i className="fas fa-paper-plane"></i>
                    </button>
                  </div>
                </div>
                {supplementFile && (
                  <div className="selected-file">
                    <i className="fas fa-file"></i> {supplementFile.name}
                    <button 
                      className="btn btn-sm btn-link" 
                      onClick={() => setSupplementFile(null)}
                    >
                      <i className="fas fa-times"></i>
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DiagnosisModule;
