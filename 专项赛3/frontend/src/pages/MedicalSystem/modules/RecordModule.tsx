import React, { useState, useRef } from 'react';
import html2canvas from 'html2canvas';
import { message } from 'antd';

interface MedicalRecord {
  name: string;
  gender: string;
  age: string;
  contactInfo: string;
  visitDate: string;
  diseaseCategory: string;
  symptoms: string;
  diagnosis: string;
  prescription: string;
}

const RecordModule: React.FC = () => {
  const [medicalRecord, setMedicalRecord] = useState<MedicalRecord>({
    name: '李某某',
    gender: '男',
    age: '42岁',
    contactInfo: '138****5678',
    visitDate: '2025年08月03日',
    diseaseCategory: '呼吸系统疾病',
    symptoms: '反复咳嗽、咳痰3个月，加重伴发热1周。患者自述咳嗽以干咳为主，夜间明显，伴有胸闷、气促，活动后加重。',
    diagnosis: '1. 慢性支气管炎急性发作\n2. 肺部感染\n3. 呼吸功能不全',
    prescription: '1. 阿莫西林克拉维酸钾片 0.625g bid po\n2. 氨溴索口服液 30ml tid po\n3. 布地奈德福莫特罗粉吸入剂 160/4.5μg bid 吸入\n4. 复方甘草片 3片 tid po\n\n注意事项：多饮水，避免辛辣刺激食物，戒烟限酒，注意休息。'
  });

  const [hasRecord, setHasRecord] = useState(false); // 是否已生成病历
  const recordRef = useRef<HTMLDivElement>(null); // 病历内容区域的引用

  const handleGenerate = () => {
    console.log('一键生成病历');
    
    const sampleRecord = {
      name: "张三",
      gender: "男",
      age: "45岁",
      contactInfo: "138****1234",
      visitDate: "2025年1月8日",
      diseaseCategory: "内科（消化系统疾病）",
      symptoms: "患者主诉：腹痛3天，伴有恶心、呕吐、食欲不振。患者描述疼痛位于上腹部，呈阵发性绞痛，疼痛向背部放射。无发热，大便正常。既往有胆囊炎病史。",
      diagnosis: "急性胆囊炎\n\n诊断依据：\n1. 患者有典型的右上腹疼痛，向右肩背部放射\n2. 疼痛呈阵发性绞痛，伴恶心呕吐\n3. 有胆囊炎既往史\n4. 体格检查右上腹压痛明显\n\n分析建议：\n• 建议进一步行腹部超声检查明确诊断\n• 必要时可行CT检查排除并发症\n• 监测炎症指标变化",
      prescription: "【中药处方】\n茵陈蒿汤加减：\n茵陈 20g，栀子 12g，大黄 6g\n柴胡 12g，黄芩 10g，半夏 10g\n郁金 12g，元胡 15g，木香 8g\n甘草 6g\n\n【用法用量】\n水煎服，每日一剂，分2次温服\n连续服用5-7天\n\n【注意事项】\n1. 饭后30分钟服用，避免空腹\n2. 服药期间忌食辛辣、油腻食物\n3. 如症状加重请及时就医"
    };
    
    setMedicalRecord(sampleRecord);
    setHasRecord(true); // 标记已生成病历
    message.success('AI病历诊断生成成功');
  };

  const handleImport = () => {
    console.log('一键导入病历');
    // 模拟导入病历图片生成的病历
    const importedRecord = {
      name: "王五",
      gender: "女", 
      age: "38岁",
      contactInfo: "159****9876",
      visitDate: "2025年1月8日",
      diseaseCategory: "妇科（内分泌疾病）",
      symptoms: "患者主诉：月经不调6个月，伴有情绪波动、乏力。患者描述月经周期不规律，量少色暗，有血块。伴有头晕、失眠、易怒等症状。",
      diagnosis: "月经不调（气血瘀滞证）\n\n诊断依据：\n1. 月经周期不规律，量少色暗有血块\n2. 伴有情绪症状和体力下降\n3. 舌质暗红，苔薄白，脉细涩\n\n治疗原则：\n• 活血化瘀，调经止痛\n• 疏肝理气，养血安神",
      prescription: "【中药处方】\n血府逐瘀汤加减：\n当归 15g，川芝 10g，红花 6g\n桃仁 10g，生地黄 12g，赤芍 10g\n牛膝 12g，桔梗 6g，柴胡 6g\n枳壳 6g，甘草 3g\n\n【用法用量】\n水煎服，每日一剂，分早晚2次温服\n月经前7天开始服用，连服至月经来潮\n\n【生活调理】\n1. 保持情绪稳定，避免过度紧张\n2. 规律作息，充足睡眠\n3. 适当运动，避免剧烈活动"
    };
    
    setMedicalRecord(importedRecord);
    setHasRecord(true); // 标记已生成病历
    message.success('病历导入生成成功');
  };

  const handleSave = async () => {
    if (!hasRecord) {
      message.warning('请先生成病历后再保存');
      return;
    }

    try {
      // 截图病历内容
      if (!recordRef.current) {
        message.error('无法获取病历内容区域');
        return;
      }

      console.log('开始截图病历...');
      const canvas = await html2canvas(recordRef.current, {
        backgroundColor: '#ffffff',
        scale: 2, // 提高截图质量
        useCORS: true,
        allowTaint: true
      });

      // 将canvas转换为blob
      canvas.toBlob(async (blob) => {
        if (!blob) {
          message.error('截图生成失败');
          return;
        }

        // 下载截图
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `病历_${medicalRecord.name}_${new Date().toISOString().split('T')[0]}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        console.log('病历截图下载成功');

        // 准备API请求数据
        const formData = new FormData();
        formData.append('recordImage', blob, `病历_${medicalRecord.name}.png`);
        formData.append('patientId', '1'); // 这里应该是当前用户ID
        formData.append('symptoms', medicalRecord.symptoms);
        formData.append('disease', medicalRecord.diagnosis);
        formData.append('prescription', medicalRecord.prescription);

        // 发送到后端保存
        try {
          console.log('发送病历保存请求到后端...');
          const token = localStorage.getItem('token'); // 获取存储的token
          
          if (!token) {
            message.warning('用户未登录，请先登录后再保存病历');
            return;
          }

          const response = await fetch('/api/record/save', {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${token}`
            },
            body: formData
          });

          if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
          }

          const result = await response.json();
          console.log('API响应:', result);

          if (result.success) {
            message.success(`病历保存成功！病历ID: ${result.data.recordId}，保存时间: ${new Date(result.data.createdAt).toLocaleString()}，截图已下载到本地`, 6);
          } else {
            message.error(`病历保存失败: ${result.message}`);
          }
        } catch (apiError) {
          console.error('API保存错误:', apiError);
          if (apiError instanceof Error) {
            message.warning(`病历保存到服务器失败: ${apiError.message}，但截图已成功下载到本地`, 5);
          } else {
            message.warning('病历保存到服务器失败，但截图已下载', 4);
          }
        }
      }, 'image/png', 0.95);

    } catch (error) {
      console.error('保存病历错误:', error);
      message.error('保存病历时发生错误');
    }
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
              <i className="fas fa-magic"></i> 诊断生成
            </button>
            <button className="btn btn-primary" onClick={handleImport}>
              <i className="fas fa-file-import"></i> 导入生成
            </button>
            <button 
              className={`btn ${hasRecord ? 'btn-success' : 'btn-disabled'}`} 
              onClick={handleSave}
              disabled={!hasRecord}
            >
              <i className="fas fa-save"></i> 保存病历
            </button>
          </div>
          
          {hasRecord ? (
            <div className="medical-record" ref={recordRef}>
              <div className="record-header">
                <h3>AI辅助诊疗报告</h3>
                <div className="system-info">
                  <div className="system-name">AI辅助诊疗系统</div>
                  <div className="generation-time">生成时间：{new Date().toLocaleString()}</div>
                </div>
              </div>
              
              <div className="record-basic-info">
                <div className="info-row">
                  <div className="info-item">
                    <span className="info-label">姓名：</span>
                    <span className="info-value">{medicalRecord.name}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">性别：</span>
                    <span className="info-value">{medicalRecord.gender}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">年龄：</span>
                    <span className="info-value">{medicalRecord.age}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">联系方式：</span>
                    <span className="info-value">{medicalRecord.contactInfo}</span>
                  </div>
                </div>
                <div className="info-row">
                  <div className="info-item">
                    <span className="info-label">诊疗日期：</span>
                    <span className="info-value">{medicalRecord.visitDate}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">疾病分类：</span>
                    <span className="info-value">{medicalRecord.diseaseCategory}</span>
                  </div>
                </div>
              </div>
              
              <div className="record-section">
                <div className="section-label">主要症状</div>
                <div className="section-content">{medicalRecord.symptoms}</div>
              </div>
              
              <div className="record-section">
                <div className="section-label">AI诊断结果</div>
                <div className="section-content diagnosis-content">{medicalRecord.diagnosis}</div>
              </div>
              
              <div className="record-section">
                <div className="section-label">推荐处方</div>
                <div className="section-content prescription-content">{medicalRecord.prescription}</div>
              </div>

              <div className="ai-disclaimer">
                <div className="disclaimer-header">
                  <i className="fas fa-exclamation-triangle"></i>
                  <span className="disclaimer-title">免责声明：</span>
                </div>
                <div className="disclaimer-content">
                  本报告由AI辅助诊疗系统生成，仅供医疗参考，不能替代专业医师的诊断和治疗建议。诊断结果和用药建议可能存在误差，请务必在专业医师指导下进行治疗。如有疑问或病情严重，请及时就医。
                </div>
              </div>
            </div>
          ) : (
            <div className="record-placeholder">
              <div className="placeholder-icon">
                <i className="fas fa-file-medical-alt"></i>
              </div>
              <div className="placeholder-content">
                <h4>尚未生成病历</h4>
                <p>请先进行<strong>辅助问诊</strong>或<strong>辅助望诊</strong>，然后点击<strong>"诊断生成"</strong>按钮生成AI诊疗报告</p>
                <p>或者上传病历图片，点击<strong>"导入生成"</strong>按钮导入现有病历</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RecordModule;
