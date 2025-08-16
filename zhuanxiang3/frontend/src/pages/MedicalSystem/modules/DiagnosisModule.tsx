import React, { useState, useRef, useEffect } from 'react';
import { message } from 'antd';
import { apiService } from '../../../services/apiService';
import { useMedicalStore } from '../../../store/medicalStore';

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

  // 使用医疗数据store
  const { 
    setDiagnosisResult, 
    addDiagnosisSupplement 
  } = useMedicalStore();

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
    
    try {
      // 准备FormData
      const formData = new FormData();
      formData.append('image', uploadedImage);
      if (imageDescription) {
        formData.append('description', imageDescription);
      }

      // 调用真实API进行图片分析
      const response = await apiService.analyzeImage(formData);
      
      if (response.success && response.data) {
        setAnalysisResult(response.data.results);
        setShowSupplement(true);
        
        // 保存望诊结果到store
        const diagnosisResult = {
          imageUrl: URL.createObjectURL(uploadedImage),
          description: imageDescription,
          analysisReport: response.data.results,
          supplements: [],
          timestamp: new Date().toISOString()
        };
        setDiagnosisResult(diagnosisResult);
        
        message.success('图片分析完成');
      } else {
        message.error(response.message || '图片分析失败');
      }
    } catch (error) {
      console.error('图片分析失败:', error);
      message.error('图片分析失败，请稍后重试');
    } finally {
      setIsAnalyzing(false);
    }
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
    const currentInput = supplementInput;
    const currentFile = supplementFile;
    setSupplementInput('');
    setSupplementFile(null);
    setIsSupplementing(true);

    try {
      // 准备FormData
      const formData = new FormData();
      formData.append('prevAnalysis', analysisResult);
      formData.append('additionalInfo', currentInput);
      if (currentFile) {
        formData.append('additionalFile', currentFile);
      }

      // 调用真实API进行补充分析
      const response = await apiService.completeWatchAnalysis(formData);
      
      if (response.success && response.data) {
        // 在对话框中显示详细的补充分析结果，而不是简单确认消息
        const aiMessage: SupplementMessage = {
          id: Date.now() + 1,
          type: 'ai',
          content: response.data.results, // 显示完整的补充分析结果
          timestamp: new Date()
        };
        
        setSupplementMessages(prev => [...prev, aiMessage]);
        
        // 保存补充分析到store
        const supplement = {
          description: currentInput,
          analysis: response.data.results,
          timestamp: new Date().toISOString()
        };
        addDiagnosisSupplement(supplement);
        
        message.success('补充分析完成');
      } else {
        message.error(response.message || '补充分析失败');
      }
    } catch (error) {
      console.error('补充分析失败:', error);
      message.error('补充分析失败，请稍后重试');
      
      // 发生错误时添加错误提示消息
      const errorMessage: SupplementMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: '抱歉，补充分析失败，请稍后重试。',
        timestamp: new Date()
      };
      setSupplementMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsSupplementing(false);
    }
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
