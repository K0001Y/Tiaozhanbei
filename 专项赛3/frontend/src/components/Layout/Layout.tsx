import React, { useState } from 'react';
import './Layout.scss';

interface LayoutProps {
  children: React.ReactElement<{ activeModule?: string }>;
}

interface NavigationItem {
  id: string;
  label: string;
  icon: string;
  active?: boolean;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [activeModule, setActiveModule] = useState<string>('dashboard');
  const [currentTitle, setCurrentTitle] = useState<string>('辅助诊疗');

  const navigationItems: NavigationItem[] = [
    { id: 'dashboard', label: '患者信息', icon: 'fas fa-home' },
    { id: 'knowledge', label: '知识库', icon: 'fas fa-book-medical' },
    { id: 'search', label: '病理检索', icon: 'fas fa-search' },
    { id: 'diagnosis', label: '辅助望诊', icon: 'fas fa-eye' },
    { id: 'inquiry', label: '辅助问切', icon: 'fas fa-user-md' },
    { id: 'record', label: '病历生成', icon: 'fas fa-file-medical' },
    { id: 'ai-assist', label: '智能助手', icon: 'fas fa-robot' },
  ];

  const handleNavigation = (moduleId: string, moduleLabel: string) => {
    setActiveModule(moduleId);
    setCurrentTitle(moduleLabel);
  };

  return (
    <div className="medical-layout">
      {/* 侧边导航 */}
      <div className="sidebar">
        <div className="logo">
          <i className="fas fa-heartbeat"></i>
          <h1>AI辅助诊疗系统</h1>
        </div>
        <div className="nav-items">
          {navigationItems.map((item) => (
            <div
              key={item.id}
              className={`nav-item ${activeModule === item.id ? 'active' : ''}`}
              onClick={() => handleNavigation(item.id, item.label)}
            >
              <i className={item.icon}></i>
              <span>{item.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* 主内容区 */}
      <div className="main-content">
        <div className="header">
          <h2 id="module-title">{currentTitle}</h2>
          <div className="user-info">
            <img src="https://randomuser.me/api/portraits/women/65.jpg" alt="用户头像" />
            <div>
              <div>张医生</div>
              <div className="user-title">副主任医师</div>
            </div>
          </div>
        </div>

        {/* 动态内容区域 */}
        <div className="content-area">
          {React.cloneElement(children, { activeModule })}
        </div>
      </div>
    </div>
  );
};

export default Layout;
