import React, { useState, useEffect } from 'react';
// TODO: 后续实现真实API调用时启用
// import { apiService, UserProfile, MedicalRecord } from '../../../services/apiService';
import { UserProfile, MedicalRecord } from '../../../services/apiService';

const DashboardModule: React.FC = () => {
  // TODO: 后续实现真实API调用，当前使用静态示例数据
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [medicalRecords, setMedicalRecords] = useState<MedicalRecord[]>([]);
  const [loading] = useState(false); // 设为false以直接显示静态数据
  const [error, setError] = useState<string>('');

  // 静态示例数据
  const staticProfile: UserProfile = {
    userId: 12345,
    username: 'zhangsan',
    name: '张三',
    age: 35,
    gender: '男',
    phone: '138****8888'
  };

  const staticRecords: MedicalRecord[] = [
    {
      recordId: 1,
      symptoms: '头痛、发热、咳嗽',
      disease: '上呼吸道感染',
      prescription: '阿莫西林胶囊 500mg，每日3次；布洛芬片 200mg，发热时服用',
      date: '2024-08-05'
    },
    {
      recordId: 2,
      symptoms: '胃痛、恶心、食欲不振',
      disease: '慢性胃炎',
      prescription: '奥美拉唑肠溶胶囊 20mg，每日2次；铝碳酸镁片 500mg，餐前服用',
      date: '2024-07-20'
    },
    {
      recordId: 3,
      symptoms: '腰痛、腿部酸痛',
      disease: '腰肌劳损',
      prescription: '双氯芬酸钠缓释片 75mg，每日1次；活血止痛胶囊，每日3次',
      date: '2024-06-15'
    }
  ];

  useEffect(() => {
    // 使用静态数据初始化
    setUserProfile(staticProfile);
    setMedicalRecords(staticRecords);
    
    // TODO: 后续替换为真实API调用
    // fetchUserProfile();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // TODO: 后续启用真实API调用
  /*
  const fetchUserProfile = async () => {
    try {
      setLoading(true);
      setError('');
      
      const response = await apiService.getUserProfile();
      
      if (response.success && response.data) {
        setUserProfile(response.data.user);
        
        // 如果登录时有病历数据，可以从localStorage获取
        const storedRecords = localStorage.getItem('userRecords');
        if (storedRecords) {
          setMedicalRecords(JSON.parse(storedRecords));
        }
      } else {
        setError(response.message || '获取用户信息失败');
      }
    } catch (err) {
      setError('网络错误，请稍后重试');
      console.error('获取用户信息失败:', err);
    } finally {
      setLoading(false);
    }
  };
  */

  const formatDate = (dateString?: string) => {
    if (!dateString) return '未知日期';
    return new Date(dateString).toLocaleDateString('zh-CN');
  };

  if (loading) {
    return (
      <div className="dashboard-module">
        <div className="card">
          <div className="card-body text-center">
            <i className="fas fa-spinner fa-spin" style={{ fontSize: '24px', color: '#2c6fbb' }}></i>
            <p style={{ marginTop: '10px' }}>正在加载用户信息...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard-module">
        <div className="card">
          <div className="card-body text-center">
            <i className="fas fa-exclamation-triangle" style={{ fontSize: '24px', color: '#dc3545' }}></i>
            <p style={{ marginTop: '10px', color: '#dc3545' }}>{error}</p>
            <button 
              className="btn btn-primary" 
              onClick={() => {
                setError('');
                setUserProfile(staticProfile);
                setMedicalRecords(staticRecords);
              }}
              style={{ marginTop: '10px' }}
            >
              重新加载
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-module">
      {/* 患者信息卡片 */}
      <div className="card">
        <div className="card-header">
          <i className="fas fa-user-injured"></i>
          <h3>患者信息</h3>
        </div>
        <div className="card-body">
          <div className="patient-info-grid">
            <div className="info-item">
              <label>用户名</label>
              <div className="info-value">{userProfile?.username || '未设置'}</div>
            </div>
            
            <div className="info-item">
              <label>姓名</label>
              <div className="info-value">{userProfile?.name || '未设置'}</div>
            </div>
            
            <div className="info-item">
              <label>年龄</label>
              <div className="info-value">{userProfile?.age ? `${userProfile.age}岁` : '未设置'}</div>
            </div>
            
            <div className="info-item">
              <label>性别</label>
              <div className="info-value">{userProfile?.gender || '未设置'}</div>
            </div>
            
            <div className="info-item">
              <label>联系电话</label>
              <div className="info-value">{userProfile?.phone || '未设置'}</div>
            </div>
            
            <div className="info-item">
              <label>用户ID</label>
              <div className="info-value">{userProfile?.userId || '未知'}</div>
            </div>
          </div>
        </div>
      </div>

      {/* 病历记录卡片 */}
      <div className="card" style={{ marginTop: '20px' }}>
        <div className="card-header">
          <i className="fas fa-file-medical"></i>
          <h3>病历记录</h3>
          <span className="record-count">
            共 {medicalRecords.length} 条记录
          </span>
        </div>
        <div className="card-body">
          {medicalRecords.length === 0 ? (
            <div className="empty-records text-center">
              <i className="fas fa-file-medical-alt" style={{ fontSize: '48px', color: '#dee2e6', marginBottom: '16px' }}></i>
              <p style={{ color: '#6c757d', fontSize: '16px' }}>暂无病历记录</p>
              <p style={{ color: '#6c757d', fontSize: '14px' }}>
                您可以通过"辅助问诊"、"辅助望诊"诊断后通过"病历生成"功能创建病历记录
              </p>
            </div>
          ) : (
            <div className="records-list">
              {medicalRecords.map((record, index) => (
                <div key={record.recordId || index} className="record-item">
                  <div className="record-header">
                    <span className="record-id">病历 #{record.recordId || index + 1}</span>
                    <span className="record-date">{formatDate(record.date)}</span>
                  </div>
                  
                  <div className="record-content">
                    <div className="record-field">
                      <label>症状描述：</label>
                      <span>{record.symptoms || '未记录'}</span>
                    </div>
                    
                    <div className="record-field">
                      <label>诊断结果：</label>
                      <span>{record.disease || '未记录'}</span>
                    </div>
                    
                    <div className="record-field">
                      <label>治疗建议：</label>
                      <span>{record.prescription || '未记录'}</span>
                    </div>
                  </div>
                  
                  <div className="record-actions">
                    <button className="btn btn-sm btn-outline">
                      <i className="fas fa-eye"></i> 查看详情
                    </button>
                    <button className="btn btn-sm btn-outline">
                      <i className="fas fa-download"></i> 导出图片
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DashboardModule;
