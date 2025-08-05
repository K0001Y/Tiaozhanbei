// src/components/Tabbar/Tabbar.tsx
import "./Tabbar.scss";
import { OpenAIOutlined, MessageOutlined, EyeOutlined, SearchOutlined, FileTextOutlined } from '@ant-design/icons';
import React from 'react';

interface TabbarProps {
  setCurrentSection: (section: string) => void;
}

const Tabbar: React.FC<TabbarProps> = ({ setCurrentSection }) => {
  return (
    <div className="tabbar-container">
      <div className="tabbar-item" onClick={() => setCurrentSection("望闻")}>
        <EyeOutlined style={{ fontSize: "24px" }} />
        <span className="tabbar-text">望闻</span>
      </div>
      <div className="tabbar-item" onClick={() => setCurrentSection("问切")}>
        <MessageOutlined style={{ fontSize: "24px" }} />
        <span className="tabbar-text">问切</span>
      </div>
      <div className="tabbar-ai" onClick={() => setCurrentSection("AI辅助")}>
        <OpenAIOutlined style={{ fontSize: "36px", marginTop: "-150px" }} />
        <span className="tabbar-ai-text">AI辅助</span>
      </div>
      <div className="tabbar-item" onClick={() => setCurrentSection("病理检索")}>
        <SearchOutlined style={{ fontSize: "24px" }} />
        <span className="tabbar-text">病理检索</span>
      </div>
      <div className="tabbar-item" onClick={() => setCurrentSection("病历生成")}>
        <FileTextOutlined style={{ fontSize: "24px" }} />
        <span className="tabbar-text">病历生成</span>
      </div>
    </div>
  );
};

export default Tabbar;
