import React, { useState } from 'react';

interface KnowledgeItem {
  id: number;
  title: string;
  tag: string;
}

const KnowledgeModule: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [knowledgeItems, setKnowledgeItems] = useState<KnowledgeItem[]>([
    { id: 1, title: '中医诊断学', tag: '中医' },
    { id: 2, title: '现代内科学', tag: '内科' },
    { id: 3, title: '临床病理学图谱', tag: '病理' }
  ]);

  const handleFileUpload = () => {
    console.log('打开文件选择器');
    // 这里可以添加文件上传逻辑
  };

  const handleOnlineSearch = () => {
    console.log('在线搜索知识库');
    // 这里可以添加在线搜索逻辑
  };

  const handleDeleteItem = (id: number) => {
    setKnowledgeItems(prev => prev.filter(item => item.id !== id));
  };

  return (
    <div className="knowledge-module">
      <div className="card">
        <div className="card-header">
          <i className="fas fa-book-medical"></i>
          <h3>编辑知识库</h3>
        </div>
        <div className="card-body">
          <div className="form-group">
            <label>添加书目</label>
            <input 
              type="text" 
              className="form-control" 
              placeholder="输入书名或关键词"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          
          <div className="btn-group">
            <button className="btn btn-primary" onClick={handleFileUpload}>
              <i className="fas fa-upload"></i> 从本地导入文件
            </button>
            <button className="btn btn-secondary" onClick={handleOnlineSearch}>
              <i className="fas fa-search"></i> 在线搜索
            </button>
          </div>
          
          <h4 style={{ margin: '25px 0 15px', color: 'var(--dark)' }}>知识库列表</h4>
          <div className="knowledge-table">
            <div className="table-header">
              <div>书名</div>
              <div>标签</div>
              <div>操作</div>
            </div>
            {knowledgeItems.map((item) => (
              <div key={item.id} className="table-row">
                <div>{item.title}</div>
                <div>{item.tag}</div>
                <div>
                  <button 
                    className="delete-btn"
                    onClick={() => handleDeleteItem(item.id)}
                  >
                    <i className="fas fa-trash"></i>
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default KnowledgeModule;
