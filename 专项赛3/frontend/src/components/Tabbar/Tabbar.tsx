import "./Tabbar.scss";
import { OpenAIOutlined, MessageOutlined, EyeOutlined, SearchOutlined, FileTextOutlined } from '@ant-design/icons';

const Tabbar = () => {
  return (
    <div className="tabbar-container">
      <div className="tabbar-item">
        <EyeOutlined style={{fontSize: "24px"}}/>
        <span className="tabbar-text">望闻</span>
      </div>
      <div className="tabbar-item">
        <MessageOutlined style={{fontSize: "24px"}}/>
        <span className="tabbar-text">问切</span>
      </div>
      <div className="tabbar-ai">
        <OpenAIOutlined style={{fontSize: "36px", marginTop: "-150px"}}/>
        <span className="tabbar-ai-text">AI辅助</span>
      </div>
      <div className="tabbar-item">
        <SearchOutlined style={{fontSize: "24px"}}/>
        <span className="tabbar-text">病理检索</span>
      </div>
      <div className="tabbar-item">
        <FileTextOutlined style={{fontSize: "24px"}}/>
        <span className="tabbar-text">病历生成</span>
      </div>
    </div>
  );
};

export default Tabbar;
