import React, { useState } from 'react';

interface InquiryData {
  symptoms: string;
  duration: string;
  severity: string;
  additionalNotes: string;
}

const InquiryModule: React.FC = () => {
  const [inquiryData, setInquiryData] = useState<InquiryData>({
    symptoms: '',
    duration: '',
    severity: '',
    additionalNotes: ''
  });

  const handleInputChange = (field: keyof InquiryData, value: string) => {
    setInquiryData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleStartInquiry = () => {
    console.log('开始问切诊断', inquiryData);
    // 这里可以添加问切诊断的逻辑
  };

  const handleAIAnalysis = () => {
    console.log('AI分析症状', inquiryData);
    // 这里可以添加AI分析的逻辑
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
            <button className="btn btn-primary" onClick={handleStartInquiry}>
              <i className="fas fa-stethoscope"></i> 开始问切
            </button>
            <button className="btn btn-secondary" onClick={handleAIAnalysis}>
              <i className="fas fa-brain"></i> AI分析
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InquiryModule;
