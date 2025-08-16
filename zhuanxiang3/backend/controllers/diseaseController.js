const { pool } = require('../config/database');

// 获取检索结果 - 需要AI配合实现
const searchDiseases = async (req, res) => {
  try {
    console.log(`${new Date().toISOString()} - GET /api/search`);
    
    const { search } = req.query;

    console.log('疾病检索请求参数:', { search });

    // 验证输入
    if (!search || !search.trim()) {
      return res.status(400).json({
        success: false,
        message: '请提供搜索关键词'
      });
    }

    const searchKeyword = search.trim();

    // TODO: 调用大模型进行疾病知识检索和匹配
    // 大模型接口调用部分 - 需要接入AI模型
    /*
    const aiSearchResult = await callAIModelForDiseaseSearch({
      keyword: searchKeyword,
      maxResults: 5
    });
    
    // 大模型应该返回的数据格式:
    // {
    //   diseases: [
    //     {
    //       diseaseName: "疾病名称",
    //       description: "疾病描述",
    //       source: "来源",
    //       relevance: "相关度百分比"
    //     }
    //   ]
    // }
    */

    // 静态数据模拟 - 替代大模型返回的结果
    const mockDiseases = getMockDiseaseData(searchKeyword);

    console.log(`疾病检索完成 - 关键词: ${searchKeyword}, 找到 ${mockDiseases.length} 个结果`);

    res.json({
      success: true,
      message: '检索成功',
      data: {
        diseases: mockDiseases
      }
    });

  } catch (error) {
    console.error('疾病检索失败:', error);
    res.status(500).json({
      success: false,
      message: '检索失败，请稍后重试',
      error: error.message
    });
  }
};

// 模拟大模型返回的疾病数据
const getMockDiseaseData = (keyword) => {
  // 模拟疾病数据库
  const mockDatabase = [
    {
      diseaseId: 1,
      diseaseName: "高血压",
      description: "遗传因素、饮食不当、缺乏运动。头痛、头晕、心悸。药物治疗、生活方式改变。低盐饮食、适量运动、戒烟限酒。",
      source: "内科学(第八版)",
      relevance: "92%"
    },
    {
      diseaseId: 2,
      diseaseName: "糖尿病",
      description: "胰岛素分泌不足或作用异常。多饮、多尿、多食、体重减轻。胰岛素治疗、口服降糖药。饮食控制、规律运动、血糖监测。",
      source: "内分泌学指南",
      relevance: "88%"
    },
    {
      diseaseId: 3,
      diseaseName: "冠心病",
      description: "冠状动脉粥样硬化导致心肌缺血。胸痛、胸闷、心悸、气短。药物治疗、介入治疗、手术治疗。戒烟限酒、低脂饮食、适量运动。",
      source: "心血管疾病诊疗指南",
      relevance: "85%"
    },
    {
      diseaseId: 4,
      diseaseName: "感冒",
      description: "病毒感染引起的上呼吸道疾病。发热、咳嗽、流鼻涕、咽痛。对症治疗、抗病毒药物。多休息、多饮水、保暖。",
      source: "临床医学手册",
      relevance: "90%"
    },
    {
      diseaseId: 5,
      diseaseName: "胃炎",
      description: "胃黏膜炎症性疾病。上腹痛、恶心、呕吐、食欲不振。抗酸药、胃黏膜保护剂。规律饮食、避免刺激性食物。",
      source: "消化内科学",
      relevance: "87%"
    },
    {
      diseaseId: 6,
      diseaseName: "肺炎",
      description: "肺部感染性疾病。发热、咳嗽、咳痰、胸痛、呼吸困难。抗生素治疗、对症支持治疗。充分休息、营养支持。",
      source: "呼吸内科学",
      relevance: "89%"
    },
    {
      diseaseId: 7,
      diseaseName: "头痛",
      description: "头部疼痛综合征。紧张性头痛、偏头痛、丛集性头痛。镇痛药物、预防性治疗。规律作息、避免诱因、减轻压力。",
      source: "神经内科学",
      relevance: "94%"
    },
    {
      diseaseId: 8,
      diseaseName: "失眠",
      description: "睡眠障碍性疾病。入睡困难、睡眠维持困难、早醒。睡眠卫生指导、药物治疗。规律作息、放松训练、避免刺激。",
      source: "睡眠医学",
      relevance: "86%"
    }
  ];

  // 简单的关键词匹配算法模拟大模型检索
  const lowerKeyword = keyword.toLowerCase();
  
  const matchedDiseases = mockDatabase.filter(disease => {
    return disease.diseaseName.toLowerCase().includes(lowerKeyword) ||
           disease.description.toLowerCase().includes(lowerKeyword);
  });

  // 如果没有直接匹配，进行模糊匹配
  if (matchedDiseases.length === 0) {
    // 模拟大模型的语义理解能力
    const semanticMatches = mockDatabase.filter(disease => {
      // 简单的语义匹配规则
      if (lowerKeyword.includes('头') || lowerKeyword.includes('痛')) {
        return disease.diseaseName.includes('头痛') || disease.diseaseName.includes('高血压');
      }
      if (lowerKeyword.includes('发热') || lowerKeyword.includes('咳嗽')) {
        return disease.diseaseName.includes('感冒') || disease.diseaseName.includes('肺炎');
      }
      if (lowerKeyword.includes('胸') || lowerKeyword.includes('心')) {
        return disease.diseaseName.includes('冠心病');
      }
      if (lowerKeyword.includes('血糖') || lowerKeyword.includes('糖')) {
        return disease.diseaseName.includes('糖尿病');
      }
      if (lowerKeyword.includes('胃') || lowerKeyword.includes('腹')) {
        return disease.diseaseName.includes('胃炎');
      }
      if (lowerKeyword.includes('睡') || lowerKeyword.includes('眠')) {
        return disease.diseaseName.includes('失眠');
      }
      return false;
    });
    
    if (semanticMatches.length > 0) {
      return semanticMatches.slice(0, 5); // 最多返回5个结果
    }
  }

  // 返回匹配结果，最多5个
  return matchedDiseases.slice(0, 5);
};

module.exports = {
  searchDiseases
};
