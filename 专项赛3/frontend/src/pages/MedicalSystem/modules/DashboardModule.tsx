import React, { useState } from 'react';

interface PatientInfo {
  name: string;
  birthDate: string;
  condition: string;
  description: string;
}

const DashboardModule: React.FC = () => {
  const [patientInfo, setPatientInfo] = useState<PatientInfo>({
    name: '',
    birthDate: '',
    condition: '',
    description: ''
  });

  const handleInputChange = (field: keyof PatientInfo, value: string) => {
    setPatientInfo(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleCardClick = (action: string) => {
    console.log(`执行操作: ${action}`);
    // 这里可以添加实际的业务逻辑
  };

  return (
    <div className="dashboard-module">
      
      <div className="card">
        <div className="card-header">
          <i className="fas fa-user-injured"></i>
          <h3>当前患者信息</h3>
        </div>
        <div className="card-body">
          <div className="form-group">
            <label>姓名</label>
            <input 
              type="text" 
              className="form-control" 
              placeholder="请输入患者姓名"
              value={patientInfo.name}
              onChange={(e) => handleInputChange('name', e.target.value)}
            />
          </div>
          
          <div className="form-group">
            <label>出生年月</label>
            <input 
              type="date" 
              className="form-control"
              value={patientInfo.birthDate}
              onChange={(e) => handleInputChange('birthDate', e.target.value)}
            />
          </div>
          
          <div className="form-group">
            <label>选择病情</label>
            <select 
              className="form-control"
              value={patientInfo.condition}
              onChange={(e) => handleInputChange('condition', e.target.value)}
            >
              <option value="">请选择病情类型</option>
              <option value="发热">发热</option>
              <option value="咳嗽">咳嗽</option>
              <option value="头痛">头痛</option>
              <option value="消化不良">消化不良</option>
              <option value="其他">其他</option>
            </select>
          </div>
          
          <div className="form-group">
            <label>补充描述</label>
            <textarea 
              className="form-control" 
              rows={3} 
              placeholder="请输入详细症状描述"
              value={patientInfo.description}
              onChange={(e) => handleInputChange('description', e.target.value)}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardModule;
