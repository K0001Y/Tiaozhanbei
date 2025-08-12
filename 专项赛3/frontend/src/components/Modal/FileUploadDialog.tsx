import React, { useState } from 'react';
import Modal from './Modal';

interface FileUploadDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (fileName: string, tags: string) => void;
  file: File | null;
}

const FileUploadDialog: React.FC<FileUploadDialogProps> = ({
  isOpen,
  onClose,
  onConfirm,
  file
}) => {
  const [fileName, setFileName] = useState('');
  const [tags, setTags] = useState('');

  // 当对话框打开时，设置默认文件名
  React.useEffect(() => {
    if (isOpen && file) {
      setFileName(file.name);
    }
  }, [isOpen, file]);

  const handleConfirm = () => {
    if (!fileName.trim()) {
      alert('请输入资料名称');
      return;
    }
    onConfirm(fileName.trim(), tags.trim());
    handleClose();
  };

  const handleClose = () => {
    setFileName('');
    setTags('');
    onClose();
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="添加资料到知识库">
      <div className="file-upload-dialog">
        {file && (
          <div className="file-info">
            <div className="file-name">
              <i className="fas fa-file-alt"></i> {file.name}
            </div>
            <div className="file-size">{formatFileSize(file.size)}</div>
          </div>
        )}

        <div className="form-group">
          <label>资料名称 *</label>
          <input
            type="text"
            value={fileName}
            onChange={(e) => setFileName(e.target.value)}
            placeholder="请输入资料名称"
            autoFocus
          />
        </div>

        <div className="form-group">
          <label>分类标签</label>
          <input
            type="text"
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            placeholder="例如：中医、内科、外科等（可选）"
          />
        </div>

        <div className="modal-actions">
          <button className="btn btn-secondary" onClick={handleClose}>
            取消
          </button>
          <button className="btn btn-primary" onClick={handleConfirm}>
            确认上传
          </button>
        </div>
      </div>
    </Modal>
  );
};

export default FileUploadDialog;
