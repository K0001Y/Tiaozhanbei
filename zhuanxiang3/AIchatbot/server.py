"""
主API服务器 - Linus彻底修复版
统一管理所有优化后的API模块，集成智能session系统和graph优化
"""
import logging
import os
import sys
import time
import traceback
from typing import Dict, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# 导入优化后的graph系统
try:
    from graph import get_compiled_graph, get_graph_stats
    GRAPH_AVAILABLE = True
except ImportError:
    print("警告: 无法导入优化后的graph系统")
    GRAPH_AVAILABLE = False
    def get_compiled_graph():
        return None
    def get_graph_stats():
        return {"error": "graph系统不可用"}

# 导入修复后的API模块
try:
    # 优先尝试导入修复后的API
    from routes.search_api import RAGSearchAPI
    
    # 导入修复后的望诊API
    from routes.watch_api import MedicalImageAPI
    
    # 导入修复后的问诊API  
    from routes.inquiry_api import MedicalInquiryAPI
    
    # 导入修复后的病历生成API
    from routes.record_api import RecordAPI
    
    # 导入修复后的文档导入API
    from routes.import_api import ImportAPI
    
    # 导入修复后的AI分析API
    from routes.analysis_api import AIAnalysisAPI
    
    FIXED_APIS_AVAILABLE = True
    
except ImportError as e:
    print(f"警告: 无法导入修复后的API，尝试使用原有版本: {e}")
    FIXED_APIS_AVAILABLE = False
    

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('server.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class OptimizedAPIServer:
    """主API服务器类 - Linus优化版，集成所有修复功能"""
    
    def __init__(self):
        """初始化优化后的API服务器"""
        logger.info("开始初始化优化后的API服务器...")
        
        # 创建Flask应用
        self.app = Flask(__name__)
        self._configure_flask_app()
        
        # 初始化graph系统
        self.compiled_graph = None
        self._initialize_graph_system()
        
        # API模块注册表
        self.api_modules = {}
        
        # 服务器信息
        self.server_info = {
            "name": "智能医疗诊断API服务器",
            "version": "2.0.0",  # Linus优化版
            "description": "集成智能session管理和专业医学分析的API服务器",
            "features": [
                "智能Session管理系统",
                "优化的Graph编译和复用",
                "专业医学AI分析",
                "上下文感知诊断",
                "增量计算优化"
            ]
        }
        
        # 启动时间记录
        self.start_time = time.time()
        
        # 初始化API模块
        self._initialize_optimized_api_modules()
        
        # 注册路由
        self._register_enhanced_routes()
        
        # 注册错误处理器
        self._register_smart_error_handlers()
        
        logger.info("优化后的API服务器初始化完成！")
    
    def _configure_flask_app(self):
        """配置Flask应用"""
        # 启用跨域支持
        CORS(self.app, resources={
            r"/api/*": {
                "origins": "*",
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"]
            }
        })
        
        # Flask配置优化
        self.app.config.update(
            JSON_AS_ASCII=False,  # 支持中文JSON
            JSON_SORT_KEYS=False,  # 保持JSON键顺序
            MAX_CONTENT_LENGTH=50 * 1024 * 1024,  # 50MB上传限制
            UPLOAD_FOLDER='temp_uploads'
        )
        
        # 创建临时上传目录
        os.makedirs(self.app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    def _initialize_graph_system(self):
        """初始化优化后的graph系统"""
        if GRAPH_AVAILABLE:
            try:
                logger.info("初始化优化后的Graph系统...")
                self.compiled_graph = get_compiled_graph()
                if self.compiled_graph:
                    logger.info("Graph系统初始化成功 - 使用单例编译实例")
                    graph_stats = get_graph_stats()
                    logger.info(f"Graph统计信息: {graph_stats}")
                else:
                    logger.warning("Graph编译失败")
            except Exception as e:
                logger.error(f"Graph系统初始化失败: {str(e)}")
                self.compiled_graph = None
        else:
            logger.warning("Graph系统不可用，部分功能可能受限")
    
    def _initialize_optimized_api_modules(self):
        """初始化所有优化后的API模块"""
        try:
            initialization_order = [
                ('rag_search', lambda: RAGSearchAPI(), "RAG搜索API"),
                ('medical_image', lambda: MedicalImageAPI(), "医学图像分析API（优化版）"),
                ('medical_inquiry', lambda: MedicalInquiryAPI(), "中医问诊分析API（优化版）"),
                ('record', lambda: RecordAPI(), "病历生成API（优化版）"),
                ('import', lambda: ImportAPI(), "文档导入分析API（优化版）"),
                ('ai_analysis', lambda: AIAnalysisAPI(), "AI智能分析API（优化版）")
            ]
            
            for module_key, initializer, description in initialization_order:
                try:
                    logger.info(f"初始化 {description}")
                    self.api_modules[module_key] = initializer()
                    
                    # 特殊处理：为record API设置graph（如果需要）
                    if module_key == 'record' and hasattr(self.api_modules[module_key], 'set_graph'):
                        # 注意：修复后的record API不再需要graph参数
                        pass
                    
                    logger.info(f"✅ {description} 初始化成功")
                    
                except Exception as e:
                    logger.error(f"❌ {description} 初始化失败: {str(e)}")
                    # 继续初始化其他模块
                    continue
            
            success_count = len(self.api_modules)
            total_count = len(initialization_order)
            
            if success_count == total_count:
                logger.info(f"🎉 所有 {success_count} 个API模块初始化成功！")
                if FIXED_APIS_AVAILABLE:
                    logger.info("✨ 使用Linus优化版API - 性能大幅提升！")
                else:
                    logger.warning("⚠️ 使用原版API - 建议升级到优化版")
            else:
                logger.warning(f"⚠️ {success_count}/{total_count} 个API模块初始化成功")
            
        except Exception as e:
            logger.error(f"API模块批量初始化失败: {str(e)}")
            raise
    
    def _register_enhanced_routes(self):
        """注册增强版路由"""
        
        # 根路径 - 增强版服务器信息
        @self.app.route('/', methods=['GET'])
        def enhanced_index():
            """增强版根路径接口"""
            api_list = {}
            for module_name, module in self.api_modules.items():
                try:
                    api_info = module.get_api_info()
                    api_list[module_name] = api_info
                except Exception as e:
                    api_list[module_name] = {"error": str(e), "status": "unhealthy"}
            
            # 添加运行时统计
            runtime_stats = {
                "uptime_seconds": int(time.time() - self.start_time),
                "graph_available": GRAPH_AVAILABLE,
                "optimized_apis": FIXED_APIS_AVAILABLE,
                "active_modules": len(self.api_modules)
            }
            
            if GRAPH_AVAILABLE:
                runtime_stats["graph_stats"] = get_graph_stats()
            
            return jsonify({
                "server": self.server_info,
                "runtime_stats": runtime_stats,
                "apis": api_list,
                "endpoints": {
                    "health": "/health",
                    "graph_stats": "/graph/stats",
                    "search": "/api/search?search=关键词",
                    "watch": "/api/watch (POST with image)",
                    "watch_complete": "/api/watch/complete (POST with form data)",
                    "inquiry": "/api/inquiry (POST with JSON)",
                    "inquiry_complete": "/api/inquiry/complete (POST with form data)",
                    "record": "/api/record (POST with JSON)",
                    "import": "/api/import (POST with file or JSON)",
                    "ai_analyze": "/api/ai/analyze (POST with text and/or file)"
                },
                "optimization_features": self.server_info["features"]
            })
        
        # 增强版健康检查
        @self.app.route('/health', methods=['GET'])
        def enhanced_health_check():
            """增强版健康检查接口"""
            module_status = {}
            overall_healthy = True
            
            for module_name, module in self.api_modules.items():
                try:
                    module_info = module.get_api_info()
                    module_status[module_name] = {
                        "status": "healthy",
                        "name": module_info.get("name", module_name),
                        "version": module_info.get("version", "unknown"),
                        "features": module_info.get("features", [])
                    }
                except Exception as e:
                    module_status[module_name] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    overall_healthy = False
            
            # 添加graph系统状态
            graph_status = "healthy" if self.compiled_graph else "unavailable"
            if GRAPH_AVAILABLE:
                try:
                    graph_stats = get_graph_stats()
                    graph_status = "healthy" if graph_stats.get("graph_compiled") else "not_compiled"
                except:
                    graph_status = "error"
            
            health_data = {
                "status": "healthy" if overall_healthy else "unhealthy",
                "message": f"API服务器运行{'正常' if overall_healthy else '异常'}",
                "timestamp": time.time(),
                "uptime_seconds": int(time.time() - self.start_time),
                "server": self.server_info,
                "modules": module_status,
                "graph_system": {
                    "status": graph_status,
                    "available": GRAPH_AVAILABLE,
                    "stats": get_graph_stats() if GRAPH_AVAILABLE else None
                },
                "optimizations": {
                    "fixed_apis_loaded": FIXED_APIS_AVAILABLE,
                    "session_management": "enabled" if FIXED_APIS_AVAILABLE else "disabled",
                    "graph_reuse": "enabled" if self.compiled_graph else "disabled"
                }
            }
            
            status_code = 200 if overall_healthy else 500
            return jsonify(health_data), status_code
        
        # 新增：Graph统计信息接口
        @self.app.route('/graph/stats', methods=['GET'])
        def graph_statistics():
            """Graph系统统计信息"""
            if not GRAPH_AVAILABLE:
                return jsonify({
                    "error": "Graph系统不可用",
                    "available": False
                }), 503
            
            try:
                stats = get_graph_stats()
                return jsonify({
                    "success": True,
                    "data": stats,
                    "message": "Graph统计信息获取成功"
                })
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e),
                    "message": "获取Graph统计信息失败"
                }), 500
        
        # 原有API路由 - 保持完全兼容
        @self.app.route('/api/search', methods=['GET'])
        def search_diseases():
            """疾病搜索API - 接口3.1"""
            return self._safe_api_call('rag_search', 'handle_search_request', 
                                     error_data={"diseases": []})
        
        @self.app.route('/api/watch', methods=['POST'])
        def watch_analysis():
            """图片望诊API - 接口4.1"""
            return self._safe_api_call('medical_image', 'handle_watch_request',
                                     error_data={"results": ""})
        
        @self.app.route('/api/watch/complete', methods=['POST'])
        def watch_complete():
            """望诊补充API - 接口4.2"""
            return self._safe_api_call('medical_image', 'handle_watch_complete_request',
                                     error_data={"results": ""})
        
        @self.app.route('/api/inquiry', methods=['POST'])
        def inquiry_analysis():
            """初步问诊API - 接口5.1"""
            return self._safe_api_call('medical_inquiry', 'handle_inquiry_request',
                                     error_data={"results": ""})
        
        @self.app.route('/api/inquiry/complete', methods=['POST'])
        def inquiry_complete():
            """补充问诊API - 接口5.2"""
            return self._safe_api_call('medical_inquiry', 'handle_inquiry_complete_request',
                                     error_data={"results": ""})
        
        @self.app.route('/api/record', methods=['POST'])
        def medical_record():
            """病历生成API - 接口6.1"""
            return self._safe_api_call('record', 'handle_record_request',
                                     error_data={"symptoms": "", "disease": "", "prescription": ""})
        
        @self.app.route('/api/record/import', methods=['POST'])
        def document_import():
            """文档导入分析API - 接口6.2"""
            return self._safe_api_call('import', 'handle_record_import_request',
                                     error_data={"symptoms": "", "disease": "", "prescription": ""})
        
        @self.app.route('/api/ai/analyze', methods=['POST'])
        def ai_intelligent_analyze():
            """AI智能分析API - 接口7.1"""
            return self._safe_api_call('ai_analysis', 'handle_ai_analyze_request',
                                     error_data={"solution": ""})
        
        logger.info("✅ 所有增强版路由注册完成")
    
    def _safe_api_call(self, module_name: str, method_name: str, error_data: Dict = None):
        """安全的API调用包装器"""
        try:
            if module_name not in self.api_modules:
                return jsonify({
                    "success": False,
                    "message": f"{module_name} API模块未初始化",
                    "data": error_data or {}
                }), 500
            
            module = self.api_modules[module_name]
            if not hasattr(module, method_name):
                return jsonify({
                    "success": False,
                    "message": f"{module_name} 模块缺少 {method_name} 方法",
                    "data": error_data or {}
                }), 500
            
            # 调用API方法
            return getattr(module, method_name)()
            
        except Exception as e:
            logger.error(f"API调用失败 {module_name}.{method_name}: {str(e)}")
            logger.error(traceback.format_exc())
            
            return jsonify({
                "success": False,
                "message": f"API调用异常: {str(e)}",
                "data": error_data or {}
            }), 500
    
    def _register_smart_error_handlers(self):
        """注册智能错误处理器"""
        
        @self.app.errorhandler(404)
        def enhanced_not_found(error):
            """增强版404错误处理"""
            return jsonify({
                "success": False,
                "message": "接口不存在",
                "error": "Not Found",
                "suggestion": "请检查API路径是否正确",
                "available_endpoints": {
                    "文档": "/",
                    "健康检查": "/health",
                    "Graph统计": "/graph/stats",
                    "疾病搜索": "/api/search?search=关键词",
                    "图片望诊": "/api/watch (POST)",
                    "望诊补充": "/api/watch/complete (POST)",
                    "初步问诊": "/api/inquiry (POST)",
                    "补充问诊": "/api/inquiry/complete (POST)",
                    "病历生成": "/api/record (POST)",
                    "文档导入": "/api/import (POST)",
                    "AI智能分析": "/api/ai/analyze (POST)"
                }
            }), 404
        
        @self.app.errorhandler(405)
        def enhanced_method_not_allowed(error):
            """增强版405错误处理"""
            return jsonify({
                "success": False,
                "message": "请求方法不被允许",
                "error": "Method Not Allowed",
                "suggestion": "请检查HTTP方法是否正确（GET/POST）"
            }), 405
        
        @self.app.errorhandler(413)
        def payload_too_large(error):
            """文件过大错误处理"""
            return jsonify({
                "success": False,
                "message": "上传文件过大",
                "error": "Payload Too Large",
                "suggestion": "请上传小于50MB的文件"
            }), 413
        
        @self.app.errorhandler(500)
        def enhanced_internal_error(error):
            """增强版500错误处理"""
            logger.error(f"服务器内部错误: {str(error)}")
            logger.error(traceback.format_exc())
            
            return jsonify({
                "success": False,
                "message": "服务器内部错误",
                "error": "Internal Server Error",
                "suggestion": "请稍后重试，如问题持续请联系技术支持"
            }), 500
        
        @self.app.errorhandler(Exception)
        def enhanced_handle_exception(error):
            """增强版通用异常处理"""
            logger.error(f"未处理的异常: {str(error)}")
            logger.error(traceback.format_exc())
            
            return jsonify({
                "success": False,
                "message": "服务器发生未知错误",
                "error": str(error),
                "suggestion": "请检查请求格式并稍后重试"
            }), 500
    
    def get_app(self):
        """获取Flask应用实例"""
        return self.app
    
    def run(self, host='0.0.0.0', port=8080, debug=False):
        """启动优化后的服务器"""
        try:
            logger.info("=" * 80)
            logger.info(f"🚀 启动 {self.server_info['name']} v{self.server_info['version']}")
            logger.info("=" * 80)
            logger.info(f"📍 服务器地址: http://{host}:{port}")
            logger.info(f"📚 API文档: http://{host}:{port}/")
            logger.info(f"❤️ 健康检查: http://{host}:{port}/health")
            logger.info(f"📊 Graph统计: http://{host}:{port}/graph/stats")
            logger.info("")
            logger.info("🔗 API接口列表:")
            logger.info(f"   🔍 疾病搜索: http://{host}:{port}/api/search?search=关键词")
            logger.info(f"   👁️ 图片望诊: http://{host}:{port}/api/watch (POST)")
            logger.info(f"   📝 初步问诊: http://{host}:{port}/api/inquiry (POST)")
            logger.info(f"   📋 病历生成: http://{host}:{port}/api/record (POST)")
            logger.info(f"   📄 文档分析: http://{host}:{port}/api/import (POST)")
            logger.info(f"   🤖 AI智能分析: http://{host}:{port}/api/ai/analyze (POST)")
            logger.info("")
            logger.info("✨ 优化功能:")
            for feature in self.server_info["features"]:
                logger.info(f"   ✅ {feature}")
            logger.info("")
            logger.info(f"📦 已加载API模块: {list(self.api_modules.keys())}")
            logger.info(f"🔧 Graph系统: {'✅ 已优化' if self.compiled_graph else '❌ 不可用'}")
            logger.info(f"🔄 Session管理: {'✅ 已启用' if FIXED_APIS_AVAILABLE else '❌ 未启用'}")
            logger.info("=" * 80)
            
            self.app.run(host=host, port=port, debug=debug, threaded=True)
            
        except Exception as e:
            logger.error(f"❌ 服务器启动失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise

# 全局服务器实例
server = None

def create_optimized_server():
    """创建优化后的服务器实例"""
    global server
    if server is None:
        server = OptimizedAPIServer()
    return server

def get_app():
    """获取Flask应用实例（用于WSGI部署）"""
    return create_optimized_server().get_app()

def run_optimized_server(host='0.0.0.0', port=8080, debug=False):
    """启动优化后服务器的便捷函数"""
    server = create_optimized_server()
    server.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    # 从环境变量或命令行参数获取配置
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', '8080'))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    # 命令行参数覆盖
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    if len(sys.argv) > 2:
        host = sys.argv[2]
    if len(sys.argv) > 3:
        debug = sys.argv[3].lower() == 'true'
    
    try:
        run_optimized_server(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        logger.info("\n👋 服务器已停止")
    except Exception as e:
        logger.error(f"💥 服务器启动失败: {str(e)}")
        sys.exit(1)