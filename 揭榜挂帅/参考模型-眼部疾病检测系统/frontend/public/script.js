class EyeDiagnosisSystem {
    constructor() {
        this.initializeElements();
        this.addEventListeners();
        this.currentMode = 'single';
        this.files = {
            left: [],
            right: []
        };
    }

    initializeElements() {
        // 获取所有需要的元素
        this.modeBtns = document.querySelectorAll('.mode-btn');
        this.singleUploadContainer = document.querySelector('.single-upload-container');
        this.batchUploadContainer = document.querySelector('.batch-upload-container');

        // 单次上传输入
        this.leftEyeInput = document.getElementById('leftEye');
        this.rightEyeInput = document.getElementById('rightEye');

        // 批量上传输入（文件夹）
        this.leftEyeFolder = document.getElementById('leftEyeFolder');
        this.rightEyeFolder = document.getElementById('rightEyeFolder');

        this.submitBtn = document.getElementById('submitBtn');

        // 预览区域
        this.leftEyePreview = document.getElementById('leftEyePreview');
        this.rightEyePreview = document.getElementById('rightEyePreview');
    }

    // 合并文件处理逻辑
    handleFiles(event, eye) {
        const files = Array.from(event.target.files)
           .filter(file => file.type.startsWith('image/')); // 确保只处理图片文件

        if (this.currentMode === 'single') {
            // 单次上传模式：只取第一张图片
            this.files[eye] = files.length > 0 ? [files[0]] : [];
        } else {
            // 批量上传模式：保留所有图片
            this.files[eye] = files;
        }

        this.updatePreview();
        this.validateFiles();
    }

    // 更新事件监听器
    addEventListeners() {
        // 模式切换
        this.modeBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                console.log('切换模式:', e.target.dataset.mode);
                this.switchMode(e.target.dataset.mode);
            });
        });

        // 单次上传事件
        this.leftEyeInput.addEventListener('change', (e) => {
            console.log('处理左眼文件选择');
            this.handleFiles(e, 'left');
        });
        this.rightEyeInput.addEventListener('change', (e) => {
            console.log('处理右眼文件选择');
            this.handleFiles(e, 'right');
        });

        // 文件夹上传事件
        this.leftEyeFolder.addEventListener('change', (e) => {
            console.log('处理左眼文件夹选择');
            this.handleFiles(e, 'left');
        });
        this.rightEyeFolder.addEventListener('change', (e) => {
            console.log('处理右眼文件夹选择');
            this.handleFiles(e, 'right');
        });

        // 提交按钮
        this.submitBtn.addEventListener('click', () => this.submitImages());
    }

    switchMode(mode) {
        this.currentMode = mode;

        // 更新按钮状态
        this.modeBtns.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });

        // 切换显示区域
        this.singleUploadContainer.style.display = mode === 'single' ? 'block' : 'none';
        this.batchUploadContainer.style.display = mode === 'batch' ? 'block' : 'none';

        // 清空所有预览和文件
        this.resetFiles();
    }

    resetFiles() {
        this.files = {
            left: [],
            right: []
        };
        this.updatePreview();
        this.validateFiles();
    }

    updatePreview() {
        // 清空现有预览
        document.getElementById('leftEyePreview').innerHTML = '';
        document.getElementById('rightEyePreview').innerHTML = '';

        const createPreview = (files, eye) => {
            const previewArea = document.getElementById(`${eye}EyePreview`);

            // 只显示前5张图片
            const displayFiles = files.slice(0, 5);

            // 创建预览
            displayFiles.forEach((file, index) => {
                const reader = new FileReader();
                reader.onload = (e) => {
                    const div = document.createElement('div');
                    div.className = 'preview-item';
                    div.innerHTML = `
                        <img src="${e.target.result}" alt="${eye}眼图片">
                        <button class="remove-btn">×</button>
                        <div class="file-name">${file.name}</div>
                    `;

                    const removeBtn = div.querySelector('.remove-btn');
                    removeBtn.addEventListener('click', () => {
                        this.files[eye] = this.files[eye].filter(f => f !== file);
                        this.updatePreview();
                        this.validateFiles();
                    });

                    previewArea.appendChild(div);
                };
                reader.readAsDataURL(file);
            });

            // 如果图片数量超过5张，显示剩余数量
            if (files.length > 5) {
                const remainingCount = files.length - 5;
                const remainingDiv = document.createElement('div');
                remainingDiv.className = 'preview-item remaining-count';
                remainingDiv.innerHTML = `
                    <div class="remaining-overlay">
                        <span>+${remainingCount}</span>
                    </div>
                `;
                previewArea.appendChild(remainingDiv);
            }
        };

        // 更新文件计数
        this.updateFileCount();

        // 创建预览
        createPreview(this.files.left, 'left');
        createPreview(this.files.right, 'right');
    }

    updateFileCount() {
        const leftCount = document.querySelector('.eye-preview-column:first-child .eye-count span');
        const rightCount = document.querySelector('.eye-preview-column:last-child .eye-count span');

        leftCount.textContent = this.files.left.length;
        rightCount.textContent = this.files.right.length;
    }

    validateFiles() {
        let isValid = false;

        if (this.currentMode === 'single') {
            // 单次上传：每边必须有且只有一张图片
            isValid = this.files.left.length === 1 && this.files.right.length === 1;
        } else {
            // 批量上传：左右眼图片数量必须相等且大于0
            isValid = this.files.left.length === this.files.right.length &&
                this.files.left.length > 0;
        }

        this.submitBtn.disabled = !isValid;
        return isValid;
    }

    // 添加测试JSON
    testDiagnosisResults = {
        status: "success",
        diagnosis: [
            { name: "AAA", disease: "正常" },
            { name: "BBB", disease: "青光眼" }
        ]
    };

    // 提交图片进行诊断
    submitImages() {
        if (!this.validateFiles()) {
            alert('请确保上传了有效的图片文件。');
            return;
        }

        const formData = new FormData();
        formData.append('mode', this.currentMode);

        this.files['left'].forEach((file, index) => {
            formData.append(`leftEyeFiles[${index}]`, file);
        });

        this.files['right'].forEach((file, index) => {
            formData.append(`rightEyeFiles[${index}]`, file);
        });

        // 显示加载状态
        this.submitBtn.disabled = true;
        this.submitBtn.textContent = '诊断中...';

        // 发送请求到后端
        fetch('http://localhost:5000/api/diagnose', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('网络响应不正常');
            }
            return response.json();
        })
        .then(data => {
            if (data.status === 'success') {
                this.displayResults(data.diagnosis);
            } else {
                alert('诊断失败：' + (data.error || '未知错误'));
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('发生错误：' + error.message);
        })
        .finally(() => {
            // 恢复按钮状态
            this.submitBtn.disabled = false;
            this.submitBtn.textContent = '开始诊断';
        });
    }

    // 显示诊断结果
    displayResults(diagnosis) {
        const resultContainer = document.getElementById('resultContainer');
        resultContainer.innerHTML = '';
        // 先设置结果容器整体样式
        resultContainer.style.backgroundColor = 'rgba(0, 170, 255, 0.05)';
        resultContainer.style.padding = '2rem';
        resultContainer.style.borderRadius = '15px';
        resultContainer.style.boxShadow = '0 8px 24px rgba(0, 255, 208, 0.1)';
        resultContainer.style.marginTop = '2rem';

        diagnosis.forEach(result => {
            const resultDiv = document.createElement('div');
            resultDiv.className = 'diagnosis-result';
            resultDiv.innerHTML = `
                <strong>${result.name}:</strong> ${result.disease}
            `;
            // 直接给当前创建的结果元素设置样式
            resultDiv.style.marginBottom = '1rem';
            resultDiv.style.padding = '1rem';
            resultDiv.style.backgroundColor = 'rgba(0, 195, 255, 0.1)';
            resultDiv.style.borderRadius = '8px';
            resultDiv.style.border = '1px solid rgba(0, 195, 255, 0.1)';
            resultContainer.appendChild(resultDiv);
        });
    }

    updateHTML() {
        // 更新批量上传的input属性
        const leftFolderInput = document.getElementById('leftEyeFolder');
        const rightFolderInput = document.getElementById('rightEyeFolder');

        // 确保设置了正确的属性
        [leftFolderInput, rightFolderInput].forEach(input => {
            input.setAttribute('webkitdirectory', '');
            input.setAttribute('directory', '');
            input.setAttribute('multiple', '');
        });
    }
}

// 确保DOM完全加载后再初始化
document.addEventListener('DOMContentLoaded', () => {
    console.log('初始化系统');
    const system = new EyeDiagnosisSystem();
    system.updateHTML(); // 确保文件夹上传属性正确设置
});    