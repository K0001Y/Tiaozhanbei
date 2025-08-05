import "./Navbar.scss";
import { UserOutlined } from "@ant-design/icons";

interface NavbarProps {
  section: string;
  username: string;
}

const Navbar: React.FC<NavbarProps> = ( {section, username} ) => {
  return (
    <div className="navbar-container">
      <div className="navbar-left">
        <div className="navbar-title">计世悬壶</div>
        <div className="navbar-subtitle">——基于大语言模型的中医专病智能辅助诊疗系统</div>
        <div className="navbar-section">/ {section}</div>
      </div>
      <div className="navbar-user">
        <UserOutlined style={{ fontSize: "24px", marginRight: "8px" }} />
        <span>{username} / 用户信息</span>
      </div>
    </div>
  );
};

export default Navbar;
