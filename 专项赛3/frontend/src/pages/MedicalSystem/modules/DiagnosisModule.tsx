import React, { useState } from 'react';

const DiagnosisModule: React.FC = () => {
  const [imageDescription, setImageDescription] = useState('');
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setUploadedImage(file);
      console.log('图片已上传:', file.name);
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

  const handleAnalyze = () => {
    if (!uploadedImage) {
      alert('请先上传图片');
      return;
    }
    console.log('开始分析图片:', uploadedImage.name);
    console.log('图片描述:', imageDescription);
    // 这里可以添加图片分析的API调用
  };

  const handleViewHistory = () => {
    console.log('查看历史记录');
    // 这里可以添加查看历史记录的逻辑
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
            <label>请输入图片描述</label>
            <textarea 
              className="form-control" 
              rows={3} 
              placeholder="描述图片中的症状表现、部位等信息"
              value={imageDescription}
              onChange={(e) => setImageDescription(e.target.value)}
            />
          </div>
          
          <div className="btn-group">
            <button className="btn btn-primary" onClick={handleAnalyze}>
              <i className="fas fa-diagnoses"></i> 开始分析
            </button>
            <button className="btn btn-secondary" onClick={handleViewHistory}>
              <i className="fas fa-history"></i> 历史记录
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DiagnosisModule;
