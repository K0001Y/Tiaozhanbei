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
    email: '',
    password: ''
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

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
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
    if (formData.username.length < 3) {
      setFieldError('username', '用户名长度至少3位');
      isValid = false;
    } else if (!/^[a-zA-Z0-9_]+$/.test(formData.username)) {
      setFieldError('username', '用户名只能包含字母、数字和下划线');
      isValid = false;
    }

    // 邮箱验证
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(formData.email)) {
      setFieldError('email', '请输入有效的邮箱地址');
      isValid = false;
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
      navigate('/dashboard');
    }
  };

  return (
    <div className="register-container">
      <div className="register-card">
        <div className="register-header">
          <h1>注册</h1>
          <p>创建您的账户，开始使用AI助手</p>
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
              placeholder="请输入用户名"
              required
              disabled={isLoading}
              className={fieldErrors.username ? 'error' : ''}
            />
            {fieldErrors.username && (
              <span className="field-error">{fieldErrors.username}</span>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="email">邮箱地址</label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              placeholder="请输入您的邮箱"
              required
              disabled={isLoading}
              className={fieldErrors.email ? 'error' : ''}
            />
            {fieldErrors.email && (
              <span className="field-error">{fieldErrors.email}</span>
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
              placeholder="请输入密码"
              required
              disabled={isLoading}
              className={fieldErrors.password ? 'error' : ''}
            />
            {fieldErrors.password && (
              <span className="field-error">{fieldErrors.password}</span>
            )}
            <div className="password-hint">
              密码必须包含至少一个小写字母、一个大写字母和一个数字
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
              placeholder="请再次输入密码"
              required
              disabled={isLoading}
              className={fieldErrors.confirmPassword ? 'error' : ''}
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