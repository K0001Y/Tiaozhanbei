#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量存储批量转换工具
用于将指定文件夹中的所有txt文件转换为向量存储
"""

import os
import sys
import glob
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# 导入你的RAG管理器
from Rag_manager import RAGManager
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorStoreConverter:
    """向量存储转换器"""
    
    def __init__(self, embedding_model_path: str = "model"):
        """
        初始化转换器
        :param embedding_model_path: 嵌入模型路径
        """
        try:
            self.rag_manager = RAGManager(embedding_model_path)
            logger.info("向量存储转换器初始化成功")
        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            raise
    
    def scan_txt_files(self, folder_path: str, recursive: bool = True) -> List[str]:
        """
        扫描文件夹中的所有txt文件
        :param folder_path: 文件夹路径
        :param recursive: 是否递归扫描子文件夹
        :return: txt文件路径列表
        """
        try:
            logger.info(f"正在扫描文件夹: {folder_path}")
            
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"文件夹不存在: {folder_path}")
            
            if not os.path.isdir(folder_path):
                raise ValueError(f"路径不是文件夹: {folder_path}")
            
            txt_files = []
            
            if recursive:
                # 递归扫描所有子文件夹
                pattern = os.path.join(folder_path, "**", "*.txt")
                txt_files = glob.glob(pattern, recursive=True)
            else:
                # 只扫描当前文件夹
                pattern = os.path.join(folder_path, "*.txt")
                txt_files = glob.glob(pattern)
            
            # 过滤掉无效文件
            valid_files = []
            for file_path in txt_files:
                if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
                    valid_files.append(file_path)
            
            logger.info(f"找到 {len(valid_files)} 个有效的txt文件")
            return valid_files
            
        except Exception as e:
            logger.error(f"扫描文件夹失败: {str(e)}")
            raise
    
    def convert_folder_to_vector_store(
        self,
        folder_path: str,
        recursive: bool = True,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        create_new: bool = False
    ) -> bool:
        """
        将文件夹中的txt文件转换为向量存储
        :param folder_path: 文件夹路径
        :param recursive: 是否递归扫描
        :param chunk_size: 分块大小
        :param chunk_overlap: 分块重叠
        :param create_new: 是否创建新的向量存储（否则添加到现有存储）
        :return: 转换是否成功
        """
        try:
            logger.info(f"开始转换文件夹: {folder_path}")
            
            # 扫描txt文件
            txt_files = self.scan_txt_files(folder_path, recursive)
            
            if not txt_files:
                logger.warning("未找到任何txt文件")
                return False
            
            # 启动RAG管理器
            logger.info("启动RAG管理器...")
            self.rag_manager._start_rag_manager()
            
            # 如果需要创建新的向量存储，先清空
            if create_new:
                logger.info("创建新的向量存储...")
                self.rag_manager.clear_vector_store()
            
            # 批量处理文件
            batch_size = 10  # 每批处理10个文件
            total_files = len(txt_files)
            
            for i in range(0, total_files, batch_size):
                batch_files = txt_files[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (total_files + batch_size - 1) // batch_size
                
                logger.info(f"处理第 {batch_num}/{total_batches} 批文件 ({len(batch_files)} 个文件)")
                
                try:
                    # 添加文档到向量存储
                    self.rag_manager.add_documents_to_store(
                        documents=batch_files,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    logger.info(f"第 {batch_num} 批文件处理完成")
                    
                except Exception as e:
                    logger.error(f"处理第 {batch_num} 批文件失败: {str(e)}")
                    # 继续处理下一批
                    continue
            
            # 获取最终统计信息
            info = self.rag_manager.get_vector_store_info()
            logger.info(f"转换完成！向量存储信息: {info}")
            
            return True
            
        except Exception as e:
            logger.error(f"转换失败: {str(e)}")
            return False
    
    def show_vector_store_info(self):
        """显示向量存储信息"""
        try:
            # 先启动RAG管理器
            self.rag_manager._start_rag_manager()
            
            info = self.rag_manager.get_vector_store_info()
            
            print("\n=== 向量存储信息 ===")
            print(f"状态: {info.get('status', '未知')}")
            print(f"文档块数量: {info.get('count', 0)}")
            
            metadata = info.get('metadata', {})
            if metadata:
                print(f"原始文档数: {metadata.get('documents_count', 0)}")
                print(f"文档块数: {metadata.get('chunks_count', 0)}")
                print(f"分块大小: {metadata.get('chunk_size', 'N/A')}")
                print(f"分块重叠: {metadata.get('chunk_overlap', 'N/A')}")
                print(f"创建时间: {metadata.get('created_time', 'N/A')}")
                print(f"最后更新: {metadata.get('last_updated', 'N/A')}")
            
        except Exception as e:
            logger.error(f"获取向量存储信息失败: {str(e)}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将文件夹中的txt文件转换为向量存储')
    
    parser.add_argument('folder_path', help='包含txt文件的文件夹路径')
    parser.add_argument('--no-recursive', action='store_true', help='不递归扫描子文件夹')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE, help=f'分块大小 (默认: {DEFAULT_CHUNK_SIZE})')
    parser.add_argument('--chunk-overlap', type=int, default=DEFAULT_CHUNK_OVERLAP, help=f'分块重叠 (默认: {DEFAULT_CHUNK_OVERLAP})')
    parser.add_argument('--create-new', action='store_true', help='创建新的向量存储而不是添加到现有存储')
    parser.add_argument('--model-path', default='model', help='嵌入模型路径 (默认: model)')
    parser.add_argument('--info', action='store_true', help='显示当前向量存储信息')
    
    args = parser.parse_args()
    
    try:
        # 创建转换器
        converter = VectorStoreConverter(args.model_path)
        
        # 如果只是查看信息
        if args.info:
            converter.show_vector_store_info()
            return
        
        # 验证文件夹路径
        if not os.path.exists(args.folder_path):
            print(f"错误: 文件夹不存在: {args.folder_path}")
            sys.exit(1)
        
        # 执行转换
        print(f"开始处理文件夹: {args.folder_path}")
        print(f"递归扫描: {'关闭' if args.no_recursive else '开启'}")
        print(f"分块大小: {args.chunk_size}")
        print(f"分块重叠: {args.chunk_overlap}")
        print(f"模式: {'创建新存储' if args.create_new else '添加到现有存储'}")
        
        success = converter.convert_folder_to_vector_store(
            folder_path=args.folder_path,
            recursive=not args.no_recursive,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            create_new=args.create_new
        )
        
        if success:
            print("\n✅ 转换成功完成！")
            converter.show_vector_store_info()
        else:
            print("\n❌ 转换失败！")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()