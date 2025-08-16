const { body } = require('express-validator');

// 注册验证规则
const validateRegister = [
  body('username')
    .isLength({ min: 3, max: 50 })
    .withMessage('用户名长度必须在3-50字符之间')
    .matches(/^[a-zA-Z0-9_]+$/)
    .withMessage('用户名只能包含字母、数字和下划线'),
  
  body('password')
    .isLength({ min: 6 })
    .withMessage('密码长度至少6位')
    .matches(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/)
    .withMessage('密码必须包含至少一个小写字母、一个大写字母和一个数字'),
  
  body('name')
    .isLength({ min: 1, max: 50 })
    .withMessage('姓名不能为空且长度不超过50字符'),
  
  body('age')
    .isInt({ min: 1, max: 150 })
    .withMessage('年龄必须是1-150之间的整数'),
  
  body('gender')
    .isIn(['男', '女'])
    .withMessage('性别只能是男或女'),
  
  body('phone')
    .notEmpty()
    .withMessage('联系电话不能为空')
    .matches(/^1[3-9]\d{9}$/)
    .withMessage('请输入有效的手机号码')
];

// 登录验证规则
const validateLogin = [
  body('username')
    .isLength({ min: 3, max: 50 })
    .withMessage('用户名长度必须在3-50字符之间')
    .matches(/^[a-zA-Z0-9_]+$/)
    .withMessage('用户名只能包含字母、数字和下划线'),
  
  body('password')
    .notEmpty()
    .withMessage('密码不能为空')
];

module.exports = {
  validateRegister,
  validateLogin
};