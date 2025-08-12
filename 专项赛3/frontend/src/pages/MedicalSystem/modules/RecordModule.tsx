import React, { useState, useRef } from 'react';
import html2canvas from 'html2canvas';
import { message } from 'antd';
import { useMedicalStore } from '../../../store/medicalStore';
import { useUserStore } from '../../../store/userStore';
import { apiService } from '../../../services/apiService';

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
  const [medicalRecord, setMedicalRecord] = useState<MedicalRecord | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const recordRef = useRef<HTMLDivElement>(null);
  
  // 获取医疗store中的数据
  const { getCurrentMedicalRecord } = useMedicalStore();
  
  // 获取用户信息
  const { user } = useUserStore();

  const handleGenerate = async () => {
    try {
      setIsGenerating(true);
      
      // 从store获取最新的望诊和问诊结果
      const medicalData = getCurrentMedicalRecord();
      
      // 检查是否有望诊或问诊结果
      if (!medicalData.diagnosis && !medicalData.inquiry) {
        message.error('请先进行望诊或问诊分析后再生成病历');
        return;
      }
      
      console.log('从store获取的医疗数据:', medicalData);
      
      // 调用后端API生成病历
      const response = await apiService.generateRecord(
        medicalData.diagnosis, 
        medicalData.inquiry
      );
      
      if (response.success && response.data) {
        // 整合生成的数据为医疗记录格式，使用用户store中的真实信息
        const generatedRecord: MedicalRecord = {
          name: user?.name || user?.username || "患者",
          gender: user?.gender || "未知",
          age: user?.age?.toString() || "未知", 
          contactInfo: user?.phone || "未填写",
          visitDate: new Date().toLocaleDateString('zh-CN'),
          diseaseCategory: "AI辅助诊断",
          symptoms: response.data.patientInfo?.symptoms || '',
          diagnosis: response.data.diagnosis || '',
          prescription: response.data.prescription || ''
        };
        
        setMedicalRecord(generatedRecord);
        message.success('AI病历诊断生成成功');
      } else {
        message.error(response.message || '病历生成失败');
      }
      
    } catch (error) {
      console.error('生成病历失败:', error);
      message.error('生成病历失败，请稍后重试');
    } finally {
      setIsGenerating(false);
    }
  };

  // 导入病历图片
  const handleImport = async () => {
    // 创建隐藏的文件输入元素
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.style.display = 'none';
    
    input.onchange = async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (!file) return;
      
      try {
        setIsImporting(true);
        
        // 调用后端API处理病历图片
        const response = await apiService.importRecord(file);
        
        if (response.success && response.data) {
          const extractedData = response.data.structuredData;
          
          // 转换为医疗记录格式
          const importedRecord: MedicalRecord = {
            name: response.data.extractedContent?.patientName || "患者",
            gender: response.data.extractedContent?.gender || "未知",
            age: response.data.extractedContent?.age || "未知",
            contactInfo: "从病历图片导入",
            visitDate: response.data.extractedContent?.visitDate || new Date().toLocaleDateString('zh-CN'),
            diseaseCategory: "导入病历",
            symptoms: extractedData?.symptoms || response.data.extractedContent?.symptoms || '',
            diagnosis: extractedData?.diagnosis || response.data.extractedContent?.diagnosis || '',
            prescription: extractedData?.prescription || response.data.extractedContent?.prescription || ''
          };
          
          setMedicalRecord(importedRecord);
          message.success('病历图片解析成功');
        } else {
          message.error(response.message || '病历图片解析失败');
        }
        
      } catch (error) {
        console.error('导入病历失败:', error);
        message.error('导入病历失败，请稍后重试');
      } finally {
        setIsImporting(false);
      }
    };
    
    // 触发文件选择
    input.click();
  };


  const handleSave = async () => {
    if (!medicalRecord) {
      message.warning('请先生成病历后再保存');
      return;
    }

    try {
      setIsSaving(true);
      
      // 调用后端API保存病历
      const response = await apiService.saveRecord({
        symptoms: medicalRecord.symptoms,
        diagnosis: medicalRecord.diagnosis,
        prescription: medicalRecord.prescription
      });
      
      if (response.success) {
        message.success('病历保存成功');
        
        // 同时生成病历截图下载
        if (recordRef.current) {
          const canvas = await html2canvas(recordRef.current, {
            backgroundColor: '#ffffff',
            scale: 2,
            useCORS: true,
            allowTaint: true
          });

          canvas.toBlob((blob) => {
            if (blob) {
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = `病历_${medicalRecord.name}_${new Date().toISOString().split('T')[0]}.png`;
              document.body.appendChild(a);
              a.click();
              document.body.removeChild(a);
              URL.revokeObjectURL(url);
              message.success('病历截图已下载到本地');
            }
          }, 'image/png', 0.95);
        }
      } else {
        message.error(response.message || '病历保存失败');
      }
      
    } catch (error) {
      console.error('保存病历错误:', error);
      message.error('保存病历时发生错误');
    } finally {
      setIsSaving(false);
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
            <button 
              className="btn btn-primary" 
              onClick={handleGenerate}
              disabled={isGenerating}
            >
              <i className="fas fa-magic"></i> 
              {isGenerating ? '生成中...' : '诊断生成'}
            </button>
            <button 
              className="btn btn-primary" 
              onClick={handleImport}
              disabled={isImporting}
            >
              <i className="fas fa-file-import"></i> 
              {isImporting ? '处理中...' : '导入生成'}
            </button>
            <button 
              className={`btn ${medicalRecord ? 'btn-success' : 'btn-disabled'}`} 
              onClick={handleSave}
              disabled={!medicalRecord || isSaving}
            >
              <i className="fas fa-save"></i> 
              {isSaving ? '保存中...' : '保存病历'}
            </button>
          </div>
          
          {medicalRecord ? (
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
