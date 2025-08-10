import React, { useState, useRef, useEffect } from 'react';
import { message } from 'antd';

interface InquiryData {
  symptoms: string;
  duration: string;
  severity: string;
  additionalNotes: string;
}

interface SupplementMessage {
  id: number;
  type: 'user' | 'ai';
  content: string;
  timestamp: Date;
  file?: File;
}

const InquiryModule: React.FC = () => {
  const [inquiryData, setInquiryData] = useState<InquiryData>({
    symptoms: '',
    duration: '',
    severity: '',
    additionalNotes: ''
  });
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

  const handleInputChange = (field: keyof InquiryData, value: string) => {
    setInquiryData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSupplementFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSupplementFile(file);
      console.log('补充文件已上传:', file.name);
    }
  };

  const handleStartInquiry = async () => {
    if (!inquiryData.symptoms.trim()) {
      message.warning('请描述主要症状');
      return;
    }

    setIsAnalyzing(true);
    console.log('开始问切诊断', inquiryData);
    
    // 模拟API调用
    setTimeout(() => {
      const mockResult = `根据您提供的症状信息分析：\n\n【症状概述】\n主要症状：${inquiryData.symptoms}\n持续时间：${inquiryData.duration || '未指定'}\n严重程度：${inquiryData.severity || '未指定'}\n\n【初步分析】\n1. 根据症状描述，可能涉及${inquiryData.symptoms.includes('头') ? '头部' : '身体'}相关问题\n2. 建议关注症状的发展趋势和伴随症状\n3. 如症状持续或加重，建议及时就医\n\n【中医角度分析】\n从中医角度来看，此类症状可能与气血运行、脏腑功能等相关，建议结合舌诊、脉诊等进行综合判断。`;
      
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
        content: `感谢您的补充信息。基于新的描述，我对问诊结果进行以下更新：\n\n1. 结合补充信息，症状的具体特征更加明确\n2. 建议注意观察症状与日常生活习惯的关联性\n3. 推荐适当的调理方法和注意事项\n\n更新后的分析结果已整合到您的问诊记录中，可用于后续的病历生成。`,
        timestamp: new Date()
      };
      
      setSupplementMessages(prev => [...prev, aiMessage]);
      
      // 更新分析结果
      const updatedResult = analysisResult + "\n\n【补充分析】\n结合补充信息，进一步明确了症状特点，建议采取综合性的调理方案，包括生活方式调整和必要的医疗干预。";
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
    <div className="inquiry-module">
      <div className="card">
        <div className="card-header">
          <i className="fas fa-user-md"></i>
          <h3>问切诊断</h3>
        </div>
        <div className="card-body">
          <div className="form-group">
            <label>主要症状</label>
            <textarea 
              className="form-control" 
              rows={3} 
              placeholder="请详细描述患者的主要症状"
              value={inquiryData.symptoms}
              onChange={(e) => handleInputChange('symptoms', e.target.value)}
            />
          </div>

          <div className="form-group">
            <label>症状持续时间</label>
            <select 
              className="form-control"
              value={inquiryData.duration}
              onChange={(e) => handleInputChange('duration', e.target.value)}
            >
              <option value="">请选择持续时间</option>
              <option value="1-3天">1-3天</option>
              <option value="1周内">1周内</option>
              <option value="1-4周">1-4周</option>
              <option value="1-3个月">1-3个月</option>
              <option value="3个月以上">3个月以上</option>
            </select>
          </div>

          <div className="form-group">
            <label>症状严重程度</label>
            <select 
              className="form-control"
              value={inquiryData.severity}
              onChange={(e) => handleInputChange('severity', e.target.value)}
            >
              <option value="">请选择严重程度</option>
              <option value="轻度">轻度</option>
              <option value="中度">中度</option>
              <option value="重度">重度</option>
            </select>
          </div>

          <div className="form-group">
            <label>其他补充信息</label>
            <textarea 
              className="form-control" 
              rows={4} 
              placeholder="请补充其他相关症状、既往病史、用药情况等"
              value={inquiryData.additionalNotes}
              onChange={(e) => handleInputChange('additionalNotes', e.target.value)}
            />
          </div>

          <div className="btn-group">
            <button 
              className="btn btn-primary" 
              onClick={handleStartInquiry}
              disabled={isAnalyzing}
            >
              {isAnalyzing ? (
                <>
                  <i className="fas fa-spinner fa-spin"></i> 分析中...
                </>
              ) : (
                <>
                  <i className="fas fa-stethoscope"></i> 开始问诊
                </>
              )}
            </button>
          </div>

          {/* 问诊结果展示 */}
          {analysisResult && (
            <div className="analysis-result" style={{ marginTop: '25px' }}>
              <div className="result-header">
                <h4><i className="fas fa-chart-line"></i> 问诊分析结果</h4>
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
                      id="inquiry-supplement-file-input"
                      type="file"
                      accept="image/*,.pdf,.doc,.docx,.txt"
                      style={{ display: 'none' }}
                      onChange={handleSupplementFileUpload}
                    />
                    <button 
                      className="btn btn-outline-secondary"
                      onClick={() => document.getElementById('inquiry-supplement-file-input')?.click()}
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

export default InquiryModule;
