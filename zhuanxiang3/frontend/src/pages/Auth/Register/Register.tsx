// 文件名: Register.tsx
// 位置: src/pages/Register.tsx
import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useUserStore } from '../../../store/userStore';
import { RegisterRequest } from '../../../types/auth';
import './Register.scss';

const Register: React.FC = () => {
  const navigate = useNavigate();
  const { 
    register, 
    isLoading, 
    error, 
    fieldErrors, 
    clearError, 
    clearFieldErrors,
    setFieldError,
    isAuthenticated 
  } = useUserStore();

  const [formData, setFormData] = useState<RegisterRequest>({
    username: '',
    password: '',
    name: '',
    age: 0,
    gender: '',
    phone: ''
  });

  const [confirmPassword, setConfirmPassword] = useState('');

  // 如果已经登录，重定向到主页
  useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard');
    }
  }, [isAuthenticated, navigate]);

  // 组件卸载时清除错误
  useEffect(() => {
    return () => {
      clearError();
      clearFieldErrors();
    };
  }, [clearError, clearFieldErrors]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    
    let processedValue: string | number = value;
    
    // 处理数字类型字段
    if (name === 'age') {
      processedValue = value ? parseInt(value, 10) : 0;
    }
    
    setFormData(prev => ({
      ...prev,
      [name]: processedValue
    }));
    
    // 清除相关错误信息
    if (error) clearError();
    if (fieldErrors[name]) clearFieldErrors();
  };

  const handleConfirmPasswordChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setConfirmPassword(e.target.value);
    if (fieldErrors.confirmPassword) clearFieldErrors();
  };

  const validateForm = (): boolean => {
    let isValid = true;

    // 用户名验证
    if (formData.username.length < 3 || formData.username.length > 20) {
      setFieldError('username', '用户名长度必须在3-20字符之间');
      isValid = false;
    } else if (!/^[a-zA-Z0-9_]+$/.test(formData.username)) {
      setFieldError('username', '用户名只能包含字母、数字和下划线');
      isValid = false;
    }

    // 真实姓名验证
    if (formData.name.trim().length < 2) {
      setFieldError('name', '请输入您的真实姓名');
      isValid = false;
    }

    // 年龄验证
    if (!formData.age || formData.age < 1 || formData.age > 150) {
      setFieldError('age', '请输入有效的年龄(1-150岁)');
      isValid = false;
    }

    // 性别验证
    if (!formData.gender || !['男', '女'].includes(formData.gender)) {
      setFieldError('gender', '请选择性别');
      isValid = false;
    }

    // 手机号验证（必填）
    if (!formData.phone || formData.phone.trim() === '') {
      setFieldError('phone', '请输入联系电话');
      isValid = false;
    } else {
      const phoneRegex = /^1[3-9]\d{9}$/;
      if (!phoneRegex.test(formData.phone)) {
        setFieldError('phone', '请输入有效的手机号码');
        isValid = false;
      }
    }

    // 密码验证
    if (formData.password.length < 6) {
      setFieldError('password', '密码长度至少6位');
      isValid = false;
    } else if (!/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/.test(formData.password)) {
      setFieldError('password', '密码必须包含至少一个小写字母、一个大写字母和一个数字');
      isValid = false;
    }

    // 确认密码验证
    if (formData.password !== confirmPassword) {
      setFieldError('confirmPassword', '两次输入的密码不一致');
      isValid = false;
    }

    return isValid;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    const success = await register(formData);
    
    if (success) {
      // 注册成功后跳转到登录页面
      navigate('/login');
    }
  };

  return (
    <div className="register-container">
      <div className="register-card">
        <div className="register-header">
          <h1>注册</h1>
          <p>创建您的账户，开始使用中医智能辅助诊疗系统</p>
        </div>

        <form onSubmit={handleSubmit} className="register-form">
          {error && (
            <div className="error-message">
              <i className="error-icon">⚠️</i>
              {error}
            </div>
          )}

          <div className="form-group">
            <label htmlFor="username">用户名</label>
            <input
              type="text"
              id="username"
              name="username"
              value={formData.username}
              onChange={handleChange}
              placeholder="请输入您的用户名"
              required
              disabled={isLoading}
              className={`form-control ${fieldErrors.username ? 'error' : ''}`}
            />
            {fieldErrors.username && (
              <span className="field-error">{fieldErrors.username}</span>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="name">真实姓名</label>
            <input
              type="text"
              id="name"
              name="name"
              value={formData.name}
              onChange={handleChange}
              placeholder="请输入您的真实姓名"
              required
              disabled={isLoading}
              className={`form-control ${fieldErrors.name ? 'error' : ''}`}
            />
            {fieldErrors.name && (
              <span className="field-error">{fieldErrors.name}</span>
            )}
          </div>

          <div className="form-group-row">
            <div className="form-group">
              <label htmlFor="age">年龄</label>
              <input
                type="number"
                id="age"
                name="age"
                value={formData.age || ''}
                onChange={handleChange}
                placeholder="请输入年龄"
                min="1"
                max="150"
                required
                disabled={isLoading}
                className={`form-control ${fieldErrors.age ? 'error' : ''}`}
              />
              {fieldErrors.age && (
                <span className="field-error">{fieldErrors.age}</span>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="gender">性别</label>
              <select
                id="gender"
                name="gender"
                value={formData.gender}
                onChange={handleChange}
                required
                disabled={isLoading}
                className={`form-control ${fieldErrors.gender ? 'error' : ''}`}
              >
                <option value="">请选择性别</option>
                <option value="男">男</option>
                <option value="女">女</option>
              </select>
              {fieldErrors.gender && (
                <span className="field-error">{fieldErrors.gender}</span>
              )}
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="phone">联系电话</label>
            <input
              type="tel"
              id="phone"
              name="phone"
              value={formData.phone}
              onChange={handleChange}
              placeholder="请输入您的手机号码"
              disabled={isLoading}
              className={`form-control ${fieldErrors.phone ? 'error' : ''}`}
            />
            {fieldErrors.phone && (
              <span className="field-error">{fieldErrors.phone}</span>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="password">密码</label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              placeholder="请输入您的密码"
              required
              disabled={isLoading}
              className={`form-control ${fieldErrors.password ? 'error' : ''}`}
            />
            {fieldErrors.password && (
              <span className="field-error">{fieldErrors.password}</span>
            )}
            <div className="password-hint">
              密码至少6位，包含大小写字母和数字
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="confirmPassword">确认密码</label>
            <input
              type="password"
              id="confirmPassword"
              name="confirmPassword"
              value={confirmPassword}
              onChange={handleConfirmPasswordChange}
              placeholder="请再次输入您的密码"
              required
              disabled={isLoading}
              className={`form-control ${fieldErrors.confirmPassword ? 'error' : ''}`}
            />
            {fieldErrors.confirmPassword && (
              <span className="field-error">{fieldErrors.confirmPassword}</span>
            )}
          </div>

          <button 
            type="submit" 
            className="register-button"
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <span className="loading-spinner"></span>
                注册中...
              </>
            ) : (
              '注册'
            )}
          </button>
        </form>

        <div className="register-footer">
          <p>
            已有账户？ 
            <Link to="/login" className="auth-link">
              立即登录
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
};

export default Register;