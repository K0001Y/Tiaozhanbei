const mysql = require('mysql2/promise');
const fs = require('fs').promises;
const path = require('path');
const { pool } = require('../config/database');

// 获取知识库列表
const getLibraryList = async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 20;
    const offset = (page - 1) * limit;

    // 确保limit和offset是有效的整数
    const validLimit = Math.max(1, Math.min(limit, 100)); // 限制在1-100之间
    const validOffset = Math.max(0, offset);

    console.log('分页参数:', { page, limit: validLimit, offset: validOffset });

    // 获取总数
    const [countResult] = await pool.execute(
      'SELECT COUNT(*) as total FROM library'
    );
    const total = countResult[0].total;

    console.log('知识库总数:', total);

    // 获取分页数据 - 使用字符串拼接而非参数绑定（MySQL兼容性更好）
    const sql = `SELECT id as libraryId, title as fileName, file_path as filePath, tags, created_at as uploadTime FROM library ORDER BY created_at DESC LIMIT ${validLimit} OFFSET ${validOffset}`;
    console.log('执行SQL:', sql);
    
    const [libraries] = await pool.execute(sql);

    res.json({
      success: true,
      message: '获取知识库列表成功',
      data: {
        page,
        limit,
        total,
        libraries: libraries.map(lib => ({
          ...lib,
          filePath: `/api/files/${lib.libraryId}`, // 提供统一的文件访问路径
          uploadTime: lib.uploadTime
        }))
      }
    });

  } catch (error) {
    console.error('获取知识库列表失败:', error);
    res.status(500).json({
      success: false,
      message: '获取知识库列表失败，请稍后重试'
    });
  }
};

// 上传资料到知识库
const uploadLibrary = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: '请选择要上传的文件'
      });
    }

    const { fileName, tags } = req.body;
    const filePath = req.file.path;
    const fileExtension = path.extname(req.file.originalname);
    const finalFileName = fileName || req.file.originalname;

    // 验证文件类型（可以根据需要调整支持的文件类型）
    const allowedExtensions = ['.txt', '.pdf', '.doc', '.docx', '.md'];
    if (!allowedExtensions.includes(fileExtension.toLowerCase())) {
      // 删除已上传的文件
      await fs.unlink(filePath);
      return res.status(400).json({
        success: false,
        message: '不支持的文件类型，请上传txt、pdf、doc、docx或md文件'
      });
    }

    // 将文件信息保存到数据库
    await pool.execute(
      'INSERT INTO library (title, tags, file_path, created_at) VALUES (?, ?, ?, NOW())',
      [finalFileName, tags || '', filePath]
    );

    res.status(201).json({
      success: true,
      message: '资料上传成功'
    });

  } catch (error) {
    console.error('上传资料失败:', error);
    
    // 如果数据库操作失败，删除已上传的文件
    if (req.file && req.file.path) {
      try {
        await fs.unlink(req.file.path);
      } catch (unlinkError) {
        console.error('删除文件失败:', unlinkError);
      }
    }

    res.status(500).json({
      success: false,
      message: '上传资料失败，请稍后重试'
    });
  }
};

// 删除知识库资料
const deleteLibrary = async (req, res) => {
  try {
    const libraryId = parseInt(req.params.id);

    if (!libraryId || isNaN(libraryId)) {
      return res.status(400).json({
        success: false,
        message: '无效的资料ID'
      });
    }

    // 首先查询资料是否存在，并获取文件路径
    const [libraries] = await pool.execute(
      'SELECT id, title, file_path FROM library WHERE id = ?',
      [libraryId]
    );

    if (libraries.length === 0) {
      return res.status(404).json({
        success: false,
        message: '资料不存在'
      });
    }

    const library = libraries[0];
    
    // 删除数据库记录
    await pool.execute('DELETE FROM library WHERE id = ?', [libraryId]);

    // 删除服务器上的文件
    if (library.file_path) {
      try {
        const filePath = path.resolve(library.file_path);
        await fs.access(filePath); // 检查文件是否存在
        await fs.unlink(filePath);
        console.log(`已删除文件: ${filePath}`);
      } catch (fileError) {
        console.warn(`删除文件失败或文件不存在: ${library.file_path}`, fileError.message);
        // 文件不存在或删除失败不影响数据库删除的成功
      }
    }

    res.json({
      success: true,
      message: '资料删除成功'
    });

  } catch (error) {
    console.error('删除资料失败:', error);
    res.status(500).json({
      success: false,
      message: '删除资料失败，请稍后重试'
    });
  }
};

// 获取单个资料文件内容（用于文件下载/预览）
const getLibraryFile = async (req, res) => {
  try {
    const libraryId = parseInt(req.params.id);

    if (!libraryId || isNaN(libraryId)) {
      return res.status(400).json({
        success: false,
        message: '无效的资料ID'
      });
    }

    // 查询资料信息
    const [libraries] = await pool.execute(
      'SELECT id, title, file_path FROM library WHERE id = ?',
      [libraryId]
    );

    if (libraries.length === 0) {
      return res.status(404).json({
        success: false,
        message: '资料不存在'
      });
    }

    const library = libraries[0];
    const filePath = path.resolve(library.file_path);

    // 检查文件是否存在
    try {
      await fs.access(filePath);
    } catch (error) {
      return res.status(404).json({
        success: false,
        message: '文件不存在'
      });
    }

    // 设置响应头
    res.setHeader('Content-Disposition', `attachment; filename="${encodeURIComponent(library.title)}"`);
    res.setHeader('Content-Type', 'application/octet-stream');

    // 发送文件
    res.sendFile(filePath);

  } catch (error) {
    console.error('获取文件失败:', error);
    res.status(500).json({
      success: false,
      message: '获取文件失败，请稍后重试'
    });
  }
};

module.exports = {
  getLibraryList,
  uploadLibrary,
  deleteLibrary,
  getLibraryFile
};
