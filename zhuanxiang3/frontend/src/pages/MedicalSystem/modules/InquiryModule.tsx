import React, { useState, useRef, useEffect } from 'react';
import { message } from 'antd';
import { apiService } from '../../../services/apiService';
import { useUserStore } from '../../../store/userStore';
import { useMedicalStore } from '../../../store/medicalStore';

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
  const { user } = useUserStore();
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

  // 使用医疗数据store
  const { 
    setInquiryResult, 
    addInquirySupplement
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

    if (!user?.age || !user?.gender) {
      message.warning('请先完善个人信息（年龄和性别）');
      return;
    }

    setIsAnalyzing(true);
    
    try {
      const symptomsText = `主要症状：${inquiryData.symptoms}\n持续时间：${inquiryData.duration || '未指定'}\n严重程度：${inquiryData.severity || '未指定'}\n其他信息：${inquiryData.additionalNotes || '无'}`;
      
      const response = await apiService.analyzeInquiry({
        age: user.age,
        gender: user.gender,
        symptoms: symptomsText
      });

      if (response.success) {
        console.log('AI问诊响应数据:', response.data);
        
        // 直接使用后端返回的 results 字段
        const analysisContent = response.data?.results || '';
        
        // 如果没有内容，显示调试信息
        if (!analysisContent) {
          console.error('AI服务返回空结果:', response.data);
          message.error('AI服务返回结果为空，请重试');
          return;
        }
        
        setAnalysisResult(analysisContent);
        setShowSupplement(true);
        
        // 保存问诊结果到store
        const inquiryResult = {
          symptoms: symptomsText,
          analysisReport: analysisContent,
          supplements: [],
          timestamp: new Date().toISOString()
        };
        setInquiryResult(inquiryResult);
        
        message.success('问诊分析完成');
      } else {
        throw new Error(response.message);
      }
    } catch (error) {
      console.error('问诊分析失败:', error);
      message.error(error instanceof Error ? error.message : '问诊分析失败，请重试');
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
      // 准备FormData，按照todolist文档要求
      const formData = new FormData();
      formData.append('prevInquiry', analysisResult); // 发送之前的分析结果
      formData.append('additionalInfo', currentInput);
      
      if (currentFile) {
        formData.append('additionalFile', currentFile);
      }

      const response = await apiService.completeInquiry(formData);

      if (response.success) {
        // 在对话框中显示详细的补充分析结果
        const aiMessage: SupplementMessage = {
          id: Date.now() + 1,
          type: 'ai',
          content: response.data?.results || '补充分析完成',
          timestamp: new Date()
        };
        
        setSupplementMessages(prev => [...prev, aiMessage]);
        
        // 保存补充分析到store，但不更新上面的主报告
        const supplement = {
          additionalInfo: currentInput,
          analysis: response.data?.results || '',
          timestamp: new Date().toISOString()
        };
        addInquirySupplement(supplement);
        
        message.success('补充信息已处理');
      } else {
        throw new Error(response.message);
      }
    } catch (error) {
      console.error('问诊补充失败:', error);
      message.error(error instanceof Error ? error.message : '补充分析失败，请重试');
      
      // 添加错误提示消息
      const errorMessage: SupplementMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: '抱歉，处理补充信息时出现错误，请稍后重试。',
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
                <div style={{ 
                  whiteSpace: 'pre-wrap', 
                  fontSize: '14px', 
                  lineHeight: '1.8',
                  padding: '15px',
                  backgroundColor: '#f8f9fa',
                  border: '1px solid #e9ecef',
                  borderRadius: '6px',
                  color: '#333'
                }}>
                  {analysisResult}
                </div>
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
