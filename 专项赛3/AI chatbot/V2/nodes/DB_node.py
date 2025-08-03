"""
数据库查询节点 - 为LangGraph框架设计
专注于连接数据库和查询数据
"""
import os
import logging
from typing import Dict, List, Any, TypedDict, Optional, Union, Tuple
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义类型
class DatabaseState(TypedDict):
    """数据库节点的状态类型"""
    query: str  # SQL查询或表名
    results: Optional[List[Dict[str, Any]]]  # 查询结果
    schema_info: Optional[Dict[str, Any]]  # 数据库结构信息
    error: Optional[str]  # 错误信息
    config: Dict[str, Any]  # 配置信息


class DatabaseNode:
    """
    数据库查询节点类
    作为LangGraph中的一个节点使用，实现数据库连接和查询功能
    """
    
    def __init__(self, 
                 connection_string: str,
                 echo: bool = False,
                 max_rows: int = 1000):
        """
        初始化数据库节点
        
        Args:
            connection_string: SQLAlchemy连接字符串，例如：
                               - PostgreSQL: "postgresql://username:password@localhost:5432/dbname"
                               - MySQL: "mysql+pymysql://username:password@localhost:3306/dbname"
                               - SQLite: "sqlite:///path/to/database.db"
            echo: 是否回显SQL语句，用于调试
            max_rows: 查询结果最大行数
        """
        try:
            self.connection_string = connection_string
            self.max_rows = max_rows
            
            # 创建引擎
            self.engine = create_engine(connection_string, echo=echo)
            
            # 测试连接
            with self.engine.connect() as conn:
                pass
            
            # 获取数据库信息
            self.inspector = inspect(self.engine)
            self.tables = self.inspector.get_table_names()
            
            logger.info(f"数据库节点初始化成功，已连接到数据库，发现 {len(self.tables)} 个表")
        
        except Exception as e:
            logger.error(f"数据库节点初始化失败: {str(e)}")
            raise RuntimeError(f"数据库连接失败: {str(e)}")
    
    def __call__(self, state: DatabaseState) -> DatabaseState:
        """
        执行数据库查询，作为LangGraph节点的主函数
        
        Args:
            state: 当前状态，包含查询和配置
            
        Returns:
            更新后的状态，包含查询结果
        """
        try:
            # 从状态中获取查询
            query = state.get("query")
            if not query:
                return {"error": "查询为空", "results": None, **state}
            
            # 从状态中获取配置
            config = state.get("config", {})
            
            # 确定是表名还是SQL查询
            if query.strip().lower().startswith(("select", "show", "describe", "explain")):
                # 是SQL查询
                return self._execute_sql_query(query, state)
            else:
                # 假设是表名
                return self._get_table_data(query, state)
        
        except Exception as e:
            error_msg = f"查询失败: {str(e)}"
            logger.error(error_msg)
            return {
                "query": state.get("query", ""),
                "results": None,
                "schema_info": state.get("schema_info"),
                "error": error_msg,
                "config": state.get("config", {})
            }
    
    def _execute_sql_query(self, query: str, state: DatabaseState) -> DatabaseState:
        """执行SQL查询并返回结果"""
        try:
            logger.info(f"执行SQL查询: {query[:100]}...")
            
            # 执行查询
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                
                # 转换为字典列表
                column_names = result.keys()
                rows = []
                
                for i, row in enumerate(result):
                    if i >= self.max_rows:
                        break
                    rows.append({col: val for col, val in zip(column_names, row)})
            
            # 更新状态
            return {
                "query": query,
                "results": rows,
                "schema_info": state.get("schema_info"),
                "error": None,
                "config": state.get("config", {})
            }
        
        except SQLAlchemyError as e:
            error_msg = f"SQL查询错误: {str(e)}"
            logger.error(error_msg)
            return {
                "query": query,
                "results": None,
                "schema_info": state.get("schema_info"),
                "error": error_msg,
                "config": state.get("config", {})
            }
    
    def _get_table_data(self, table_name: str, state: DatabaseState) -> DatabaseState:
        """获取表数据"""
        try:
            # 检查表是否存在
            if table_name not in self.tables:
                return {
                    "query": table_name,
                    "results": None,
                    "schema_info": state.get("schema_info"),
                    "error": f"表 '{table_name}' 不存在",
                    "config": state.get("config", {})
                }
            
            logger.info(f"获取表数据: {table_name}")
            
            # 获取表结构
            columns = self.inspector.get_columns(table_name)
            column_names = [col['name'] for col in columns]
            
            # 构建查询
            limit = min(self.max_rows, state.get("config", {}).get("limit", self.max_rows))
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            
            # 执行查询
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                
                # 转换为字典列表
                rows = []
                for row in result:
                    rows.append({col: val for col, val in zip(column_names, row)})
            
            # 获取表结构信息
            schema_info = {
                "table_name": table_name,
                "columns": [
                    {
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col.get("nullable", True)
                    }
                    for col in columns
                ],
                "primary_keys": self.inspector.get_primary_keys(table_name),
                "foreign_keys": self.inspector.get_foreign_keys(table_name)
            }
            
            # 更新状态
            return {
                "query": table_name,
                "results": rows,
                "schema_info": schema_info,
                "error": None,
                "config": state.get("config", {})
            }
        
        except SQLAlchemyError as e:
            error_msg = f"获取表数据错误: {str(e)}"
            logger.error(error_msg)
            return {
                "query": table_name,
                "results": None,
                "schema_info": state.get("schema_info"),
                "error": error_msg,
                "config": state.get("config", {})
            }
    
    def get_database_schema(self) -> Dict[str, Any]:
        """
        获取数据库结构信息
        
        Returns:
            数据库结构信息
        """
        try:
            schema = {}
            
            # 获取所有表
            tables = self.inspector.get_table_names()
            
            # 获取每个表的结构
            for table in tables:
                columns = self.inspector.get_columns(table)
                primary_keys = self.inspector.get_primary_keys(table)
                foreign_keys = self.inspector.get_foreign_keys(table)
                
                schema[table] = {
                    "columns": [
                        {
                            "name": col["name"],
                            "type": str(col["type"]),
                            "nullable": col.get("nullable", True)
                        }
                        for col in columns
                    ],
                    "primary_key": primary_keys,
                    "foreign_keys": [
                        {
                            "constrained_columns": fk["constrained_columns"],
                            "referred_table": fk["referred_table"],
                            "referred_columns": fk["referred_columns"]
                        }
                        for fk in foreign_keys
                    ]
                }
            
            return {
                "status": "success",
                "tables": schema,
                "table_count": len(tables)
            }
        
        except Exception as e:
            logger.error(f"获取数据库结构失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "tables": {}
            }