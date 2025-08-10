import React, { useState } from 'react';

interface SearchResult {
  id: number;
  title: string;
  content: string;
  source: string;
  relevance: string;
}

const SearchModule: React.FC = () => {
  const [keyword, setKeyword] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([
    {
      id: 1,
      title: '肺炎的临床表现与诊断',
      content: '肺炎是指终末气道、肺泡和肺间质的炎症，可由病原微生物、理化因素、免疫损伤、过敏及药物所致...',
      source: '内科学(第9版)',
      relevance: '92%'
    },
    {
      id: 2,
      title: '病毒性肺炎的影像学特征',
      content: '病毒性肺炎的影像学表现多样，常见磨玻璃影、小叶间隔增厚、支气管血管束增粗等改变...',
      source: '放射诊断学',
      relevance: '87%'
    }
  ]);

  const handleSearch = () => {
    console.log(`搜索关键词: ${keyword}`);
    // 这里可以添加实际的搜索逻辑
    // 模拟搜索结果更新
    if (keyword.trim()) {
      // 可以调用API进行搜索
      // 示例：更新搜索结果
      const newResults: SearchResult[] = [
        {
          id: Date.now(),
          title: `${keyword}相关的医学文献`,
          content: `关于${keyword}的详细医学资料和临床研究...`,
          source: '医学数据库',
          relevance: '95%'
        },
        ...searchResults
      ];
      setSearchResults(newResults);
    }
  };

  return (
    <div className="search-module">
      <div className="card">
        <div className="card-header">
          <i className="fas fa-search"></i>
          <h3>病理检索</h3>
        </div>
        <div className="card-body">
          <div className="form-group">
            <label>请输入关键词</label>
            <div className="search-input-group">
              <input 
                type="text" 
                className="form-control search-input" 
                placeholder="输入疾病名称、症状或关键词"
                value={keyword}
                onChange={(e) => setKeyword(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              />
              <button className="btn btn-primary search-btn" onClick={handleSearch}>
                <i className="fas fa-search"></i> 搜索
              </button>
            </div>
          </div>
          
          <h4 style={{ margin: '25px 0 15px', color: 'var(--dark)' }}>搜索结果</h4>
          {searchResults.map((result) => (
            <div key={result.id} className="search-result-card">
              <div className="card-body">
                <h5 className="result-title">{result.title}</h5>
                <p className="result-content">{result.content}</p>
                <div className="result-meta">
                  <span>来源：{result.source}</span>
                  <span>相关度：{result.relevance}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default SearchModule;
