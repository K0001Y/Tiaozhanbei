const express = require('express');
const router = express.Router();
const libraryController = require('../controllers/libraryController');
const auth = require('../middleware/auth');
const { upload, handleUploadError } = require('../middleware/upload');

// 获取知识库列表
router.get('/', auth, libraryController.getLibraryList);

// 上传资料到知识库
router.post('/', 
  auth, 
  upload.single('file'), 
  handleUploadError, 
  libraryController.uploadLibrary
);

// 删除知识库资料
router.delete('/:id', auth, libraryController.deleteLibrary);

// 获取资料文件（下载/预览）
router.get('/files/:id', auth, libraryController.getLibraryFile);

module.exports = router;
