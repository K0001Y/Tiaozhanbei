import React, { useState, useEffect } from 'react';
import { UserProfile } from '../../../services/apiService';
import { MedicalRecord } from '../../../types/auth';
import { useUserStore } from '../../../store/userStore';
import { useNavigate } from 'react-router-dom';

const DashboardModule: React.FC = () => {
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [medicalRecords, setMedicalRecords] = useState<MedicalRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');
  const [isLoggingOut, setIsLoggingOut] = useState(false);

  // 获取用户store和导航
  const { user, logout } = useUserStore();
  const navigate = useNavigate();

  // 处理退出登录
  const handleLogout = async () => {
    try {
      setIsLoggingOut(true);
      await logout();
      navigate('/auth/login');
    } catch (err) {
      console.error('退出登录失败:', err);
      // 即使出错也跳转到登录页
      navigate('/auth/login');
    } finally {
      setIsLoggingOut(false);
    }
  };

  // 加载用户数据
  useEffect(() => {
    const loadUserData = async () => {
      try {
        setLoading(true);
        setError('');
        
        if (!user) {
          setError('用户未登录');
          navigate('/auth');
          return;
        }

        // 使用store中的用户信息
        const profile: UserProfile = {
          userId: user.userId || 0,
          username: user.username || '',
          name: user.name || '',
          age: user.age || 0,
          gender: user.gender || '',
          phone: user.phone || ''
        };
        
        setUserProfile(profile);
        
        // 设置医疗记录，如果用户有记录则使用，否则使用空数组
        setMedicalRecords(user.records || []);
        
      } catch (err) {
        console.error('加载用户数据失败:', err);
        setError('加载用户数据失败，请稍后重试');
      } finally {
        setLoading(false);
      }
    };

    loadUserData();
  }, [user, navigate]);

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
                const loadUserData = async () => {
                  try {
                    setLoading(true);
                    if (!user) {
                      navigate('/auth');
                      return;
                    }
                    const profile: UserProfile = {
                      userId: user.userId || 0,
                      username: user.username || '',
                      name: user.name || '',
                      age: user.age || 0,
                      gender: user.gender || '',
                      phone: user.phone || ''
                    };
                    setUserProfile(profile);
                    setMedicalRecords(user.records || []);
                  } catch (err) {
                    console.error('重新加载失败:', err);
                    setError('重新加载失败');
                  } finally {
                    setLoading(false);
                  }
                };
                loadUserData();
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
          <button 
            className="logout-btn"
            onClick={handleLogout}
            disabled={isLoggingOut}
            title="退出登录"
          >
            {isLoggingOut ? (
              <i className="fas fa-spinner fa-spin"></i>
            ) : (
              <i className="fas fa-sign-out-alt"></i>
            )}
            {isLoggingOut ? '退出中...' : '退出登录'}
          </button>
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
                <div key={record.id || index} className="record-item">
                  <div className="record-header">
                    <span className="record-id">病历 #{record.id || index + 1}</span>
                    <span className="record-date">{formatDate(record.created_at)}</span>
                  </div>
                  
                  <div className="record-content">
                    <div className="record-field">
                      <label>症状描述：</label>
                      <span>{record.symptoms || '未记录'}</span>
                    </div>
                    
                    <div className="record-field">
                      <label>诊断结果：</label>
                      <span>{record.diagnosis || '未记录'}</span>
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
