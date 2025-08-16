import React, { useState } from 'react';
import { apiService } from '../../../services/apiService';

interface SearchResult {
  diseaseId: number;
  diseaseName: string;
  description: string;
  source: string;
  relevance: string;
}

const SearchModule: React.FC = () => {
  const [keyword, setKeyword] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = async () => {
    if (!keyword.trim()) {
      setError('请输入搜索关键词');
      return;
    }

    setLoading(true);
    setError('');
    setHasSearched(true);

    try {
      console.log(`搜索关键词: ${keyword}`);
      const response = await apiService.searchDiseases(keyword);
      
      if (response.success && response.data) {
        setSearchResults(response.data.diseases || []);
      } else {
        setError(response.message || '搜索失败');
        setSearchResults([]);
      }
    } catch (err) {
      console.error('搜索失败:', err);
      setError('搜索失败，请稍后重试');
      setSearchResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
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
                onKeyPress={handleKeyPress}
              />
              <button 
                className="btn btn-primary search-btn" 
                onClick={handleSearch}
                disabled={loading}
              >
                <i className={loading ? "fas fa-spinner fa-spin" : "fas fa-search"}></i> 
                {loading ? '搜索中...' : '搜索'}
              </button>
            </div>
          </div>

          {error && (
            <div className="error-message" style={{ 
              color: 'var(--danger)', 
              marginBottom: '15px',
              padding: '10px',
              backgroundColor: 'rgba(var(--danger-rgb), 0.1)',
              borderRadius: '4px',
              border: '1px solid rgba(var(--danger-rgb), 0.3)'
            }}>
              {error}
            </div>
          )}
          
          <h4 style={{ margin: '25px 0 15px', color: 'var(--dark)' }}>
            搜索结果
            {searchResults.length > 0 && (
              <span style={{ fontSize: '0.8em', color: 'var(--gray)', marginLeft: '10px' }}>
                找到 {searchResults.length} 条结果
              </span>
            )}
          </h4>
          
          {loading ? (
            <div style={{ textAlign: 'center', padding: '20px' }}>
              <i className="fas fa-spinner fa-spin"></i> 搜索中...
            </div>
          ) : !hasSearched ? (
            <div style={{ 
              textAlign: 'center', 
              padding: '40px 20px', 
              color: 'var(--gray)',
              fontStyle: 'italic'
            }}>
              <i className="fas fa-search" style={{ fontSize: '2em', marginBottom: '15px', display: 'block' }}></i>
              请输入关键词开始搜索医学知识
              <div style={{ fontSize: '0.9em', marginTop: '10px' }}>
                支持搜索：疾病名称、症状描述、医学术语
              </div>
            </div>
          ) : searchResults.length === 0 ? (
            <div style={{ 
              textAlign: 'center', 
              padding: '40px 20px', 
              color: 'var(--gray)',
              fontStyle: 'italic'
            }}>
              <i className="fas fa-exclamation-circle" style={{ fontSize: '2em', marginBottom: '15px', display: 'block' }}></i>
              未找到相关的医学信息
              <div style={{ fontSize: '0.9em', marginTop: '10px' }}>
                请尝试使用其他关键词或更具体的描述
              </div>
            </div>
          ) : (
            searchResults.map((result) => (
              <div key={result.diseaseId} className="search-result-card">
                <div className="card-body">
                  <h5 className="result-title">{result.diseaseName}</h5>
                  <p className="result-content">{result.description}</p>
                  <div className="result-meta">
                    <span>来源：{result.source}</span>
                    <span>相关度：{result.relevance}</span>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default SearchModule;
