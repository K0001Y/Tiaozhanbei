import React from 'react';
import { Link } from 'react-router-dom';
import './NotFound.scss';

const NotFound: React.FC = () => {
  return (
    <div className="not-found-container">
      <div className="not-found-content">
        <div className="error-code">404</div>
        <h1>页面未找到</h1>
        <p>抱歉，您访问的页面不存在或已被删除。</p>
        <div className="action-buttons">
          <Link to="/dashboard" className="btn btn-primary">
            返回首页
          </Link>
          <Link to="/login" className="btn btn-secondary">
            重新登录
          </Link>
        </div>
      </div>
    </div>
  );
};

export default NotFound;
