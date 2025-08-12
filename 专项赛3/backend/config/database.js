const mysql = require('mysql2/promise');
require('dotenv').config();

// 创建不带数据库名的连接用于创建数据库
const createPool = mysql.createPool({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0
});

// 带数据库名的连接池
const pool = mysql.createPool({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0
});

// 创建数据库和表
const createTables = async () => {
  try {
    // 首先创建数据库（如果不存在）
    await createPool.execute(`CREATE DATABASE IF NOT EXISTS ${process.env.DB_NAME}`);
    console.log(`数据库 ${process.env.DB_NAME} 创建成功`);
    
    // 然后创建表
    // 首先检查并创建用户表
    await pool.execute(`
      CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(50) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
      )
    `);
    
    // 添加新字段（如果不存在的话）
    try {
      await pool.execute(`ALTER TABLE users ADD COLUMN name VARCHAR(50) NOT NULL DEFAULT ''`);
      console.log('添加name字段成功');
    } catch (err) {
      if (err.code !== 'ER_DUP_FIELDNAME') {
        throw err;
      }
    }
    
    try {
      await pool.execute(`ALTER TABLE users ADD COLUMN age INT NOT NULL DEFAULT 0`);
      console.log('添加age字段成功');
    } catch (err) {
      if (err.code !== 'ER_DUP_FIELDNAME') {
        throw err;
      }
    }
    
    try {
      await pool.execute(`ALTER TABLE users ADD COLUMN gender ENUM('男', '女') NOT NULL DEFAULT '男'`);
      console.log('添加gender字段成功');
    } catch (err) {
      if (err.code !== 'ER_DUP_FIELDNAME') {
        throw err;
      }
    }
    
    try {
      await pool.execute(`ALTER TABLE users ADD COLUMN phone VARCHAR(20) NOT NULL DEFAULT ''`);
      console.log('添加phone字段成功');
    } catch (err) {
      if (err.code !== 'ER_DUP_FIELDNAME') {
        throw err;
      }
    }
    
    console.log('用户表创建/更新成功');

    // 创建知识库内容表
    await pool.execute(`
      CREATE TABLE IF NOT EXISTS library (
        id INT AUTO_INCREMENT PRIMARY KEY,
        title VARCHAR(255) NOT NULL,
        tags VARCHAR(500),
        file_path VARCHAR(500),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    console.log('知识库表创建成功');

    // 创建疾病信息表
    await pool.execute(`
      CREATE TABLE IF NOT EXISTS diseases (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        source VARCHAR(100),
        description VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    console.log('疾病信息表创建成功');

    // 创建病历表
    await pool.execute(`
      CREATE TABLE IF NOT EXISTS records (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        symptoms TEXT NOT NULL,
        diagnosis VARCHAR(255),
        prescription TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        file_path VARCHAR(500),
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
      )
    `);
    console.log('病历表创建成功');

  } catch (error) {
    console.error('创建数据库表失败:', error);
    throw error;
  }
};

module.exports = { pool, createTables };