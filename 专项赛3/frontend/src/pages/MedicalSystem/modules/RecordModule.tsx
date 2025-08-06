import React, { useState } from 'react';

interface MedicalRecord {
  name: string;
  gender: string;
  age: string;
  visitDate: string;
  chiefComplaint: string;
  presentIllness: string;
  aiDiagnosis: string;
}

const RecordModule: React.FC = () => {
  const [medicalRecord] = useState<MedicalRecord>({
    name: '李某某',
    gender: '男',
    age: '42岁',
    visitDate: '2025-08-03',
    chiefComplaint: '反复咳嗽、咳痰3个月，加重伴发热1周。',
    presentIllness: '患者3个月前无明显诱因出现咳嗽，咳白色粘痰，量中等，无咯血、胸痛、呼吸困难等。1周前受凉后症状加重，咳嗽频繁，痰量增多，为黄脓痰，伴发热，体温最高38.5℃，自服"感冒药"效果不佳...',
    aiDiagnosis: '肺炎可能性大（细菌性），建议进一步行胸部CT及痰培养检查。'
  });

  const handleGenerate = () => {
    console.log('一键生成病历');
    // 这里可以添加AI生成病历的逻辑
  };

  const handleImport = () => {
    console.log('一键导入病历');
    // 这里可以添加导入病历的逻辑
  };

  const handleSave = () => {
    console.log('保存病历', medicalRecord);
    // 这里可以添加保存病历的逻辑
    alert('病历已保存');
  };

  return (
    <div className="record-module">
      <div className="card">
        <div className="card-header">
          <i className="fas fa-file-medical"></i>
          <h3>病历生成</h3>
        </div>
        <div className="card-body">
          <div className="btn-group">
            <button className="btn btn-primary" onClick={handleGenerate}>
              <i className="fas fa-magic"></i> 一键生成
            </button>
            <button className="btn btn-primary" onClick={handleImport}>
              <i className="fas fa-file-import"></i> 一键导入
            </button>
            <button className="btn btn-secondary" onClick={handleSave}>
              <i className="fas fa-save"></i> 保存病历
            </button>
          </div>
          
          <div className="medical-record">
            <h4>电子病历</h4>
            
            <div className="record-basic-info">
              <div className="info-item">
                <div className="info-label">姓名：</div>
                <div className="info-value">{medicalRecord.name}</div>
              </div>
              <div className="info-item">
                <div className="info-label">性别：</div>
                <div className="info-value">{medicalRecord.gender}</div>
              </div>
              <div className="info-item">
                <div className="info-label">年龄：</div>
                <div className="info-value">{medicalRecord.age}</div>
              </div>
              <div className="info-item">
                <div className="info-label">就诊日期：</div>
                <div className="info-value">{medicalRecord.visitDate}</div>
              </div>
            </div>
            
            <div className="record-section">
              <div className="section-label">主诉：</div>
              <div className="section-content">{medicalRecord.chiefComplaint}</div>
            </div>
            
            <div className="record-section">
              <div className="section-label">现病史：</div>
              <div className="section-content">{medicalRecord.presentIllness}</div>
            </div>
            
            <div className="record-section">
              <div className="section-label">AI辅助诊断：</div>
              <div className="section-content ai-diagnosis">{medicalRecord.aiDiagnosis}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RecordModule;
