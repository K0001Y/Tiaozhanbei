import React from 'react';
import Modal from './Modal';

interface ConfirmDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  type?: 'danger' | 'warning' | 'info';
}

const ConfirmDialog: React.FC<ConfirmDialogProps> = ({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
  confirmText = '确认',
  cancelText = '取消',
  type = 'danger'
}) => {
  const handleConfirm = () => {
    onConfirm();
    onClose();
  };

  const getIconClass = () => {
    switch (type) {
      case 'danger':
        return 'fas fa-exclamation-triangle';
      case 'warning':
        return 'fas fa-exclamation-circle';
      case 'info':
        return 'fas fa-info-circle';
      default:
        return 'fas fa-question-circle';
    }
  };

  const getConfirmButtonClass = () => {
    switch (type) {
      case 'danger':
        return 'btn btn-danger';
      case 'warning':
        return 'btn btn-warning';
      case 'info':
        return 'btn btn-primary';
      default:
        return 'btn btn-primary';
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={title}>
      <div className="confirm-dialog">
        <div className="confirm-content">
          <div className="confirm-icon">
            <i className={getIconClass()}></i>
          </div>
          <div className="confirm-message">{message}</div>
        </div>

        <div className="modal-actions">
          <button className="btn btn-secondary" onClick={onClose}>
            {cancelText}
          </button>
          <button className={getConfirmButtonClass()} onClick={handleConfirm}>
            {confirmText}
          </button>
        </div>
      </div>
    </Modal>
  );
};

export default ConfirmDialog;
