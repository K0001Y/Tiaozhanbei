import React, { useState, useRef } from 'react';
import { apiService } from '../../../services/apiService';

interface WatchAnalysis {
  results: string;
  imageId: string;
  imagePath: string;
}

const WatchModule: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>('');
  const [description, setDescription] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [analysis, setAnalysis] = useState<WatchAnalysis | null>(null);
  const [additionalInfo, setAdditionalInfo] = useState('');
  const [supplementFile, setSupplementFile] = useState<File | null>(null);
  const [supplementPreview, setSupplementPreview] = useState<string>('');
  const [supplementing, setSupplementing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const supplementInputRef = useRef<HTMLInputElement>(null);

  // 处理图片选择
  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // 验证文件类型
      if (!file.type.startsWith('image/')) {
        setError('请选择图片文件');
        return;
      }

      // 验证文件大小 (10MB)
      if (file.size > 10 * 1024 * 1024) {
        setError('图片文件不能超过10MB');
        return;
      }

      setSelectedImage(file);
      setError('');

      // 生成预览
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  // 处理补充图片选择
  const handleSupplementImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('请选择图片文件');
        return;
      }

      if (file.size > 10 * 1024 * 1024) {
        setError('图片文件不能超过10MB');
        return;
      }

      setSupplementFile(file);
      setError('');

      const reader = new FileReader();
      reader.onload = (e) => {
        setSupplementPreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  // 开始分析
  const handleAnalyze = async () => {
    if (!selectedImage) {
      setError('请先选择图片');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('image', selectedImage);
      if (description.trim()) {
        formData.append('description', description);
      }

      console.log('开始图片望诊分析...');
      const response = await apiService.analyzeImage(formData);

      if (response.success && response.data) {
        setAnalysis(response.data);
        console.log('望诊分析成功');
      } else {
        setError(response.message || '分析失败');
      }
    } catch (err) {
      console.error('望诊分析失败:', err);
      setError(err instanceof Error ? err.message : '分析失败，请稍后重试');
    } finally {
      setLoading(false);
    }
  };

  // 补充分析
  const handleSupplement = async () => {
    if (!analysis || !additionalInfo.trim()) {
      setError('请提供补充信息');
      return;
    }

    setSupplementing(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('prevAnalysis', analysis.results);
      formData.append('additionalInfo', additionalInfo);
      
      if (supplementFile) {
        formData.append('additionalFile', supplementFile);
      }

      console.log('开始补充分析...');
      const response = await apiService.completeWatchAnalysis(formData);

      if (response.success && response.data) {
        // 更新分析结果
        setAnalysis(prev => prev ? {
          ...prev,
          results: response.data!.results
        } : null);
        
        // 清空补充信息
        setAdditionalInfo('');
        setSupplementFile(null);
        setSupplementPreview('');
        if (supplementInputRef.current) {
          supplementInputRef.current.value = '';
        }
        
        console.log('补充分析成功');
      } else {
        setError(response.message || '补充分析失败');
      }
    } catch (err) {
      console.error('补充分析失败:', err);
      setError(err instanceof Error ? err.message : '补充分析失败，请稍后重试');
    } finally {
      setSupplementing(false);
    }
  };

  // 重新分析
  const handleRestart = () => {
    setSelectedImage(null);
    setImagePreview('');
    setDescription('');
    setAnalysis(null);
    setAdditionalInfo('');
    setSupplementFile(null);
    setSupplementPreview('');
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    if (supplementInputRef.current) {
      supplementInputRef.current.value = '';
    }
  };

  return (
    <div className="watch-module">
      <div className="card">
        <div className="card-header">
          <i className="fas fa-eye"></i>
          <h3>辅助望诊</h3>
        </div>
        <div className="card-body">
          {error && (
            <div className="error-message" style={{ 
              color: 'var(--danger)', 
              marginBottom: '15px',
              padding: '10px',
              backgroundColor: 'rgba(var(--danger-rgb), 0.1)',
              borderRadius: '4px',
              border: '1px solid rgba(var(--danger-rgb), 0.3)'
            }}>
              {error}
            </div>
          )}

          {!analysis ? (
            // 初始分析界面
            <div className="initial-analysis">
              <div className="form-group">
                <label>上传医学图片</label>
                <div className="image-upload-area" style={{
                  border: '2px dashed var(--border)',
                  borderRadius: '8px',
                  padding: '20px',
                  textAlign: 'center',
                  marginBottom: '15px',
                  cursor: 'pointer',
                  backgroundColor: imagePreview ? 'transparent' : '#f8f9fa'
                }} onClick={() => fileInputRef.current?.click()}>
                  {imagePreview ? (
                    <div className="image-preview">
                      <img 
                        src={imagePreview} 
                        alt="预览" 
                        style={{
                          maxWidth: '100%',
                          maxHeight: '300px',
                          borderRadius: '4px'
                        }}
                      />
                      <div style={{ marginTop: '10px', fontSize: '0.9em', color: 'var(--gray)' }}>
                        点击重新选择图片
                      </div>
                    </div>
                  ) : (
                    <div>
                      <i className="fas fa-cloud-upload-alt" style={{ fontSize: '2em', marginBottom: '10px', color: 'var(--gray)' }}></i>
                      <div>点击选择图片或拖拽图片到此处</div>
                      <div style={{ fontSize: '0.9em', color: 'var(--gray)', marginTop: '5px' }}>
                        支持格式：JPG、PNG、GIF等，最大10MB
                      </div>
                    </div>
                  )}
                </div>
                <input
                  type="file"
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  accept="image/*"
                  onChange={handleImageSelect}
                />
              </div>

              <div className="form-group">
                <label>图片描述（可选）</label>
                <textarea
                  className="form-control"
                  placeholder="请描述图片内容，如：CT图、X光片、皮肤照片等，有助于提高分析准确性"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  rows={3}
                />
              </div>

              <div className="btn-group">
                <button 
                  className="btn btn-primary" 
                  onClick={handleAnalyze}
                  disabled={!selectedImage || loading}
                >
                  <i className={loading ? "fas fa-spinner fa-spin" : "fas fa-search"}></i>
                  {loading ? '分析中...' : '开始分析'}
                </button>
              </div>
            </div>
          ) : (
            // 分析结果界面
            <div className="analysis-results">
              <h4 style={{ marginBottom: '15px', color: 'var(--dark)' }}>
                <i className="fas fa-microscope"></i> 望诊分析结果
              </h4>
              
              <div className="result-content" style={{
                backgroundColor: '#f8f9fa',
                padding: '20px',
                borderRadius: '8px',
                marginBottom: '20px',
                border: '1px solid var(--border)',
                whiteSpace: 'pre-line',
                lineHeight: '1.6'
              }}>
                {analysis.results}
              </div>

              {/* 补充分析区域 */}
              <div className="supplement-section" style={{
                backgroundColor: '#f0f7ff',
                padding: '20px',
                borderRadius: '8px',
                border: '1px solid #cce7ff'
              }}>
                <h5 style={{ marginBottom: '15px', color: 'var(--primary)' }}>
                  <i className="fas fa-plus-circle"></i> 补充分析信息
                </h5>

                <div className="form-group">
                  <label>补充描述</label>
                  <textarea
                    className="form-control"
                    placeholder="请提供额外的症状描述、病史信息或其他相关情况"
                    value={additionalInfo}
                    onChange={(e) => setAdditionalInfo(e.target.value)}
                    rows={3}
                  />
                </div>

                <div className="form-group">
                  <label>补充图片（可选）</label>
                  <div className="image-upload-area" style={{
                    border: '2px dashed #cce7ff',
                    borderRadius: '8px',
                    padding: '15px',
                    textAlign: 'center',
                    cursor: 'pointer',
                    backgroundColor: supplementPreview ? 'transparent' : 'white'
                  }} onClick={() => supplementInputRef.current?.click()}>
                    {supplementPreview ? (
                      <div className="image-preview">
                        <img 
                          src={supplementPreview} 
                          alt="补充图片预览" 
                          style={{
                            maxWidth: '100%',
                            maxHeight: '200px',
                            borderRadius: '4px'
                          }}
                        />
                        <div style={{ marginTop: '10px', fontSize: '0.9em', color: 'var(--gray)' }}>
                          点击重新选择补充图片
                        </div>
                      </div>
                    ) : (
                      <div>
                        <i className="fas fa-image" style={{ fontSize: '1.5em', marginBottom: '8px', color: 'var(--primary)' }}></i>
                        <div>选择补充图片</div>
                      </div>
                    )}
                  </div>
                  <input
                    type="file"
                    ref={supplementInputRef}
                    style={{ display: 'none' }}
                    accept="image/*"
                    onChange={handleSupplementImageSelect}
                  />
                </div>

                <div className="btn-group">
                  <button 
                    className="btn btn-secondary" 
                    onClick={handleSupplement}
                    disabled={!additionalInfo.trim() || supplementing}
                  >
                    <i className={supplementing ? "fas fa-spinner fa-spin" : "fas fa-plus"}></i>
                    {supplementing ? '更新中...' : '补充分析'}
                  </button>
                </div>
              </div>

              <div className="btn-group" style={{ marginTop: '20px' }}>
                <button className="btn btn-outline-primary" onClick={handleRestart}>
                  <i className="fas fa-redo"></i> 重新分析
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default WatchModule;
