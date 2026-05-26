# AutoGen 医疗多智能体系统 & AI安全攻防
东南大学网络空间安全学院 | 多智能体记忆安全实战项目

## 项目定位
基于 AutoGen 构建**8角色群组医疗多智能体系统**，实现 TF-IDF 文本记忆 / Chroma 向量记忆双版本，验证 MINJA 记忆注入攻击与安全防御，贴合企业级 AI Agent 落地与安全测试场景。

## 技术栈
- 框架：AutoGen (GroupChat 群组调度)
- 记忆库：TF-IDF + 余弦相似度 / Chroma 向量数据库
- 大模型：DeepSeek LLM
- 语言：Python
- 安全：MINJA 记忆注入攻击、记忆劫持验证

## 核心功能
1. 8角色专业化智能体协同（意图解析/病例/用药/诊断/汇总等）
2. 双版本持久化记忆模块（本地JSON/向量库存储）
3. 低阈值记忆触发，复现 MINJA 记忆攻击效果
4. 生产级鲁棒性优化、异常重试、格式标准化

## 项目亮点
网安+AI复合背景，聚焦 AI Agent 安全
工业级群组智能体架构，可直接复用
完整攻防验证：记忆注入 + 防御方案
代码规范，开箱即用，适配实习面试展示

## 文件说明
- tfidf_memory_agent.py：TF-IDF 文本记忆版本
- chroma_memory_agent.py：Chroma 向量记忆版本
- requirements.txt：项目依赖
