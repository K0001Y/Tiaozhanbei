"""
主API服务器
统一管理所有API模块，支持动态添加新的API
"""
import logging
import os
import sys
from typing import Dict, Any
from flask import Flask, request, jsonify
from flask_cors import CORS

# 导入API模块
from search_api import RAGSearchAPI
from watch_api import MedicalImageAPI
from inquiry_api import MedicalInquiryAPI
from record_api import RecordAPI  # 病历生成API
from import_api import ImportAPI  # 文档导入分析API

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class APIServer:
    """主API服务器类 - 统一管理所有API模块"""
    
    def __init__(self, graph=None):
        """
        初始化API服务器
        
        :param graph: 图状态管理器实例（可选）
        """
        # 创建Flask应用
        self.app = Flask(__name__)
        CORS(self.app)  # 启用跨域支持
        
        # 保存graph实例引用
        self.graph = graph
        
        # API模块注册表
        self.api_modules = {}
        
        # 服务器信息
        self.server_info = {
            "name": "医疗诊断API服务器",
            "version": "1.0.0",
            "description": "基于RAG技术的智能医疗诊断API集合"
        }
        
        # 初始化API模块
        self._initialize_api_modules()
        
        # 注册路由
        self._register_routes()
        
        # 注册错误处理器
        self._register_error_handlers()
        
        logger.info("API服务器初始化完成")
    
    def _initialize_api_modules(self):
        """初始化所有API模块"""
        try:
            # 注册RAG搜索API
            logger.info("初始化RAG搜索API模块")
            self.api_modules['rag_search'] = RAGSearchAPI()
            
            # 注册医学图像分析API
            logger.info("初始化医学图像分析API模块")
            self.api_modules['medical_image'] = MedicalImageAPI()
            
            # 注册中医问诊分析API
            logger.info("初始化中医问诊分析API模块")
            self.api_modules['medical_inquiry'] = MedicalInquiryAPI()
            
            # 新增：注册病历生成API
            logger.info("初始化病历生成API模块")
            self.api_modules['record'] = RecordAPI(graph=self.graph)
            
            # 新增：注册文档导入分析API
            logger.info("初始化文档导入分析API模块")
            self.api_modules['import'] = ImportAPI()
            
            # 在这里可以继续添加其他API模块
            # self.api_modules['diagnosis'] = DiagnosisAPI()
            # self.api_modules['symptom_analysis'] = SymptomAnalysisAPI()
            
            logger.info(f"成功初始化 {len(self.api_modules)} 个API模块")
            
        except Exception as e:
            logger.error(f"API模块初始化失败: {str(e)}")
            raise
    
    def set_graph(self, graph):
        """
        设置graph实例并更新RecordAPI
        
        :param graph: 图状态管理器实例
        """
        self.graph = graph
        if 'record' in self.api_modules:
            self.api_modules['record'].set_graph(graph)
            logger.info("Graph实例已更新到RecordAPI")
    
    def _register_routes(self):
        """注册所有路由"""
        
        # 根路径 - 服务器信息
        @self.app.route('/', methods=['GET'])
        def index():
            """根路径接口"""
            api_list = {}
            for module_name, module in self.api_modules.items():
                api_list[module_name] = module.get_api_info()
            
            return jsonify({
                "server": self.server_info,
                "apis": api_list,
                "endpoints": {
                    "health": "/health",
                    "search": "/api/search?search=关键词",
                    "watch": "/api/watch (POST with image)",
                    "watch_complete": "/api/watch/complete (POST with form data)",
                    "inquiry": "/api/inquiry (POST with JSON)",
                    "inquiry_complete": "/api/inquiry/complete (POST with form data)",
                    "record": "/api/record (POST with JSON)",  # 病历生成
                    "import": "/api/import (POST with file or JSON)"  # 文档导入分析
                }
            })
        
        # 健康检查
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查接口"""
            module_status = {}
            overall_healthy = True
            
            for module_name, module in self.api_modules.items():
                try:
                    # 简单检查模块是否可用
                    module_info = module.get_api_info()
                    module_status[module_name] = {
                        "status": "healthy",
                        "name": module_info.get("name", module_name)
                    }
                except Exception as e:
                    module_status[module_name] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    overall_healthy = False
            
            status_code = 200 if overall_healthy else 500
            
            return jsonify({
                "status": "healthy" if overall_healthy else "unhealthy",
                "message": f"API服务器运行{'正常' if overall_healthy else '异常'}",
                "server": self.server_info,
                "modules": module_status
            }), status_code
        
        # RAG搜索API路由
        @self.app.route('/api/search', methods=['GET'])
        def search_diseases():
            """
            疾病搜索API接口
            接口编号: 3.1
            GET /api/search?search=关键词
            """
            if 'rag_search' not in self.api_modules:
                return jsonify({
                    "success": False,
                    "message": "RAG搜索API模块未初始化",
                    "data": {"diseases": []}
                }), 500
            
            return self.api_modules['rag_search'].handle_search_request()
        
        # 医学图像分析API路由
        @self.app.route('/api/watch', methods=['POST'])
        def watch_analysis():
            """
            图片望诊API接口
            接口编号: 4.1
            POST /api/watch (multipart/form-data)
            """
            if 'medical_image' not in self.api_modules:
                return jsonify({
                    "success": False,
                    "message": "医学图像分析API模块未初始化",
                    "data": {"results": ""}
                }), 500
            
            return self.api_modules['medical_image'].handle_watch_request()
        
        @self.app.route('/api/watch/complete', methods=['POST'])
        def watch_complete():
            """
            望诊补充API接口
            接口编号: 4.2
            POST /api/watch/complete (multipart/form-data)
            """
            if 'medical_image' not in self.api_modules:
                return jsonify({
                    "success": False,
                    "message": "医学图像分析API模块未初始化",
                    "data": {"results": ""}
                }), 500
            
            return self.api_modules['medical_image'].handle_watch_complete_request()
        
        # 中医问诊分析API路由
        @self.app.route('/api/inquiry', methods=['POST'])
        def inquiry_analysis():
            """
            初步问诊API接口
            接口编号: 5.1
            POST /api/inquiry (JSON)
            """
            if 'medical_inquiry' not in self.api_modules:
                return jsonify({
                    "success": False,
                    "message": "中医问诊分析API模块未初始化",
                    "data": {"results": ""}
                }), 500
            
            return self.api_modules['medical_inquiry'].handle_inquiry_request()
        
        @self.app.route('/api/inquiry/complete', methods=['POST'])
        def inquiry_complete():
            """
            补充问诊API接口
            接口编号: 5.2
            POST /api/inquiry/complete (multipart/form-data)
            """
            if 'medical_inquiry' not in self.api_modules:
                return jsonify({
                    "success": False,
                    "message": "中医问诊分析API模块未初始化",
                    "data": {"results": ""}
                }), 500
            
            return self.api_modules['medical_inquiry'].handle_inquiry_complete_request()
        
        # 新增：病历生成API路由
        @self.app.route('/api/record', methods=['POST'])
        def medical_record():
            """
            病历生成API接口
            接口编号: 6.1
            POST /api/record (JSON)
            """
            if 'record' not in self.api_modules:
                return jsonify({
                    "success": False,
                    "message": "病历生成API模块未初始化",
                    "data": {
                        "symptoms": "",
                        "disease": "",
                        "prescription": ""
                    }
                }), 500
            
            return self.api_modules['record'].handle_record_request()
        
        # 新增：文档导入分析API路由
        @self.app.route('/api/import', methods=['POST'])
        def document_import():
            """
            文档导入分析API接口
            接口编号: 6.2
            POST /api/import (multipart/form-data 或 JSON)
            """
            if 'import' not in self.api_modules:
                return jsonify({
                    "success": False,
                    "message": "文档导入分析API模块未初始化",
                    "data": {
                        "symptoms": "",
                        "disease": "",
                        "prescription": ""
                    }
                }), 500
            
            return self.api_modules['import'].handle_import_request()
        
        # 预留其他API路由
        # 可以在这里添加更多API路由
        # 例如：
        # @self.app.route('/api/diagnosis', methods=['POST'])
        # def diagnosis():
        #     return self.api_modules['diagnosis'].handle_diagnosis_request()
        
        logger.info("所有路由注册完成")
    
    def _register_error_handlers(self):
        """注册错误处理器"""
        
        @self.app.errorhandler(404)
        def not_found(error):
            """404错误处理"""
            return jsonify({
                "success": False,
                "message": "接口不存在",
                "error": "Not Found",
                "available_endpoints": {
                    "search": "/api/search?search=关键词",
                    "watch": "/api/watch (POST)",
                    "inquiry": "/api/inquiry (POST)",
                    "record": "/api/record (POST)",  # 病历生成
                    "import": "/api/import (POST)",  # 文档导入
                    "health": "/health",
                    "info": "/"
                }
            }), 404
        
        @self.app.errorhandler(405)
        def method_not_allowed(error):
            """405错误处理"""
            return jsonify({
                "success": False,
                "message": "请求方法不被允许",
                "error": "Method Not Allowed"
            }), 405
        
        @self.app.errorhandler(500)
        def internal_error(error):
            """500错误处理"""
            logger.error(f"服务器内部错误: {str(error)}")
            return jsonify({
                "success": False,
                "message": "服务器内部错误",
                "error": "Internal Server Error"
            }), 500
        
        @self.app.errorhandler(Exception)
        def handle_exception(error):
            """通用异常处理"""
            logger.error(f"未处理的异常: {str(error)}")
            return jsonify({
                "success": False,
                "message": "服务器发生未知错误",
                "error": str(error)
            }), 500
    
    def add_api_module(self, name: str, module):
        """
        动态添加API模块
        
        :param name: 模块名称
        :param module: 模块实例
        """
        try:
            self.api_modules[name] = module
            logger.info(f"成功添加API模块: {name}")
        except Exception as e:
            logger.error(f"添加API模块失败: {name} - {str(e)}")
            raise
    
    def remove_api_module(self, name: str):
        """
        移除API模块
        
        :param name: 模块名称
        """
        if name in self.api_modules:
            del self.api_modules[name]
            logger.info(f"成功移除API模块: {name}")
        else:
            logger.warning(f"API模块不存在: {name}")
    
    def get_app(self):
        """获取Flask应用实例"""
        return self.app
    
    def run(self, host='0.0.0.0', port=8080, debug=False):
        """
        启动服务器
        
        :param host: 服务器地址
        :param port: 服务器端口
        :param debug: 是否开启调试模式
        """
        try:
            logger.info("=" * 60)
            logger.info(f"启动 {self.server_info['name']}")
            logger.info("=" * 60)
            logger.info(f"服务器地址: http://{host}:{port}")
            logger.info(f"API文档: http://{host}:{port}/")
            logger.info(f"健康检查: http://{host}:{port}/health")
            logger.info(f"疾病搜索: http://{host}:{port}/api/search?search=关键词")
            logger.info(f"图片望诊: http://{host}:{port}/api/watch (POST)")
            logger.info(f"望诊补充: http://{host}:{port}/api/watch/complete (POST)")
            logger.info(f"初步问诊: http://{host}:{port}/api/inquiry (POST)")
            logger.info(f"补充问诊: http://{host}:{port}/api/inquiry/complete (POST)")
            # 新增：病理分析API日志
            logger.info(f"病历生成: http://{host}:{port}/api/record (POST)")
            logger.info(f"文档分析: http://{host}:{port}/api/record/import (POST)")
            logger.info(f"已加载API模块: {list(self.api_modules.keys())}")
            logger.info("=" * 60)
            
            self.app.run(host=host, port=port, debug=debug)
            
        except Exception as e:
            logger.error(f"服务器启动失败: {str(e)}")
            raise

# 创建全局服务器实例
server = None

def create_server(graph=None):
    """
    创建服务器实例
    
    :param graph: 图状态管理器实例（可选）
    """
    global server
    if server is None:
        server = APIServer(graph=graph)
    return server

def get_app(graph=None):
    """
    获取Flask应用实例（用于WSGI部署）
    
    :param graph: 图状态管理器实例（可选）
    """
    return create_server(graph=graph).get_app()

def run_server(host='0.0.0.0', port=8080, debug=False, graph=None):
    """
    启动服务器的便捷函数
    
    :param host: 服务器地址
    :param port: 服务器端口
    :param debug: 是否开启调试模式
    :param graph: 图状态管理器实例（可选）
    """
    server = create_server(graph=graph)
    server.run(host=host, port=port, debug=debug)

def set_server_graph(graph):
    """
    为已创建的服务器设置graph实例
    
    :param graph: 图状态管理器实例
    """
    global server
    if server is not None:
        server.set_graph(graph)
    else:
        logger.warning("服务器实例尚未创建，无法设置graph")

if __name__ == '__main__':
    # 从环境变量或命令行参数获取配置
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', '8080'))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    # 如果有命令行参数，优先使用
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    if len(sys.argv) > 2:
        host = sys.argv[2]
    if len(sys.argv) > 3:
        debug = sys.argv[3].lower() == 'true'
    
    run_server(host=host, port=port, debug=debug)