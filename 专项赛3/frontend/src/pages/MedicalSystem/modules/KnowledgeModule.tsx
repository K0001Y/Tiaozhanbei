import React, { useState, useEffect, useRef } from 'react';
import { apiService } from '../../../services/apiService';
import FileUploadDialog from '../../../components/Modal/FileUploadDialog';
import ConfirmDialog from '../../../components/Modal/ConfirmDialog';

interface KnowledgeItem {
  libraryId: number;
  fileName: string;
  tags?: string;
  uploadTime: string;
}

interface LibraryResponse {
  libraryId: number;
  fileName: string;
  filePath: string;
  tags?: string;
  uploadTime: string;
}

const KnowledgeModule: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [knowledgeItems, setKnowledgeItems] = useState<KnowledgeItem[]>([]);
  const [filteredItems, setFilteredItems] = useState<KnowledgeItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 对话框状态
  const [showUploadDialog, setShowUploadDialog] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [itemToDelete, setItemToDelete] = useState<{ id: number; name: string } | null>(null);

  // 加载知识库列表
  const loadKnowledgeItems = async () => {
    try {
      setLoading(true);
      setError('');
      
      const response = await apiService.getLibraryList(1, 100); // 获取所有资料
      
      if (response.success && response.data) {
        const items: KnowledgeItem[] = response.data.libraries.map((lib: LibraryResponse) => ({
          libraryId: lib.libraryId,
          fileName: lib.fileName,
          tags: lib.tags || '', // 从后端获取tags
          uploadTime: lib.uploadTime
        }));
        setKnowledgeItems(items);
        setFilteredItems(items);
      } else {
        setError(response.message || '获取知识库列表失败');
      }
    } catch (err) {
      console.error('加载知识库列表失败:', err);
      setError('加载知识库列表失败，请稍后重试');
    } finally {
      setLoading(false);
    }
  };

  // 页面加载时获取知识库列表
  useEffect(() => {
    loadKnowledgeItems();
  }, []);

  // 根据搜索词过滤资料
  useEffect(() => {
    if (!searchTerm.trim()) {
      setFilteredItems(knowledgeItems);
    } else {
      try {
        // 使用正则表达式进行搜索
        const regex = new RegExp(searchTerm, 'i'); // 不区分大小写
        const filtered = knowledgeItems.filter(item => 
          regex.test(item.fileName) || 
          (item.tags && regex.test(item.tags))
        );
        setFilteredItems(filtered);
      } catch {
        // 如果正则表达式无效，使用普通字符串搜索
        const filtered = knowledgeItems.filter(item => 
          item.fileName.toLowerCase().includes(searchTerm.toLowerCase()) || 
          (item.tags && item.tags.toLowerCase().includes(searchTerm.toLowerCase()))
        );
        setFilteredItems(filtered);
      }
    }
  }, [searchTerm, knowledgeItems]);

  // 文件上传处理
  const handleFileUpload = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setShowUploadDialog(true); // 显示上传对话框要求输入资料名和分类
    }
  };

  // 处理文件上传确认
  const handleUploadConfirm = async (name: string, category: string) => {
    if (!selectedFile) return;

    try {
      setUploading(true);
      setError('');

      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('fileName', name || selectedFile.name);
      formData.append('tags', category);

      const response = await apiService.uploadLibraryFile(formData);
      
      if (response.success) {
        setShowUploadDialog(false);
        setSelectedFile(null);
        // 上传成功，重新加载列表
        await loadKnowledgeItems();
        // 清空文件选择
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      } else {
        setError(response.message || '文件上传失败');
      }
    } catch (err: unknown) {
      console.error('文件上传失败:', err);
      setError(err instanceof Error ? err.message : '文件上传失败，请稍后重试');
    } finally {
      setUploading(false);
    }
  };

  // 删除资料
  const handleDeleteItem = (libraryId: number, fileName: string) => {
    setItemToDelete({ id: libraryId, name: fileName });
    setShowDeleteDialog(true);
  };

  // 处理删除确认
  const handleDeleteConfirm = async () => {
    if (!itemToDelete) return;

    try {
      setError('');
      
      const response = await apiService.deleteLibraryFile(itemToDelete.id);
      
      if (response.success) {
        setShowDeleteDialog(false);
        setItemToDelete(null);
        // 删除成功，从本地状态中移除
        setKnowledgeItems(prev => prev.filter(item => item.libraryId !== itemToDelete.id));
      } else {
        setError(response.message || '删除失败');
      }
    } catch (err: unknown) {
      console.error('删除资料失败:', err);
      setError(err instanceof Error ? err.message : '删除资料失败，请稍后重试');
    }
  };

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString('zh-CN');
    } catch {
      return '未知日期';
    }
  };

  return (
    <div className="knowledge-module">
      <div className="card">
        <div className="card-header">
          <i className="fas fa-book-medical"></i>
          <h3>编辑知识库</h3>
        </div>
        <div className="card-body">
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

          <div className="search-upload-section" style={{ 
            display: 'flex', 
            gap: '15px', 
            alignItems: 'end',
            marginBottom: '20px'
          }}>
            <div className="form-group" style={{ flex: '1', marginBottom: '0' }}>
              <label>搜索资料</label>
              <input 
                type="text" 
                className="form-control" 
                placeholder="输入资料名称或分类"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            
            <button 
              className="btn btn-primary" 
              onClick={handleFileUpload}
              disabled={uploading}
              style={{ 
                height: 'fit-content',
                whiteSpace: 'nowrap'
              }}
            >
              <i className="fas fa-upload"></i> 
              {uploading ? '上传中...' : '导入文件'}
            </button>
          </div>

          {/* 隐藏的文件输入 */}
          <input
            type="file"
            ref={fileInputRef}
            style={{ display: 'none' }}
            accept=".txt,.pdf,.doc,.docx,.md"
            onChange={handleFileSelect}
          />
          
          <div className="knowledge-list-header" style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            marginBottom: '15px'
          }}>
            <h4 style={{ margin: '0', color: 'var(--dark)', fontSize: '1.1rem' }}>
              知识库列表
            </h4>
            {searchTerm && (
              <span style={{ fontSize: '0.85em', color: 'var(--gray)' }}>
                找到 {filteredItems.length} 条结果
              </span>
            )}
          </div>

          {loading ? (
            <div style={{ textAlign: 'center', padding: '20px' }}>
              <i className="fas fa-spinner fa-spin"></i> 加载中...
            </div>
          ) : (
            <div className="knowledge-table">
              <div className="table-header">
                <div>资料名称</div>
                <div>分类</div>
                <div>上传时间</div>
                <div></div>
              </div>
              {filteredItems.length === 0 ? (
                <div style={{ 
                  textAlign: 'center', 
                  padding: '20px', 
                  color: 'var(--gray)',
                  fontStyle: 'italic'
                }}>
                  {knowledgeItems.length === 0 ? '暂无资料' : '未找到匹配的资料'}
                </div>
              ) : (
                filteredItems.map((item) => (
                  <div key={item.libraryId} className="table-row">
                    <div>{item.fileName}</div>
                    <div>{item.tags || '未分类'}</div>
                    <div>{formatDate(item.uploadTime)}</div>
                    <div>
                      <button 
                        className="delete-btn"
                        onClick={() => handleDeleteItem(item.libraryId, item.fileName)}
                        title="删除资料"
                      >
                        <i className="fas fa-trash"></i>
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </div>

      {/* 文件上传对话框 */}
      <FileUploadDialog
        isOpen={showUploadDialog}
        file={selectedFile}
        onConfirm={handleUploadConfirm}
        onClose={() => {
          setShowUploadDialog(false);
          setSelectedFile(null);
          if (fileInputRef.current) {
            fileInputRef.current.value = '';
          }
        }}
      />

      {/* 删除确认对话框 */}
      <ConfirmDialog
        isOpen={showDeleteDialog}
        title="确认删除"
        message={itemToDelete ? `确定要删除资料 "${itemToDelete.name}" 吗？` : ''}
        onConfirm={handleDeleteConfirm}
        onClose={() => {
          setShowDeleteDialog(false);
          setItemToDelete(null);
        }}
        type="danger"
      />
    </div>
  );
};

export default KnowledgeModule;
