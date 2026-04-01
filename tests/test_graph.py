"""
基础测试用例
覆盖：查询改写、意图识别、图路由、各业务节点
使用 pytest-asyncio + mock 隔离外部依赖
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage


# =============================================================
# Fixtures
# =============================================================

@pytest.fixture
def base_state() -> dict:
    """最小化 AgentState fixture"""
    return {
        "messages": [HumanMessage(content="俯卧撑怎么做？")],
        "user_id": str(uuid.uuid4()),
        "app_user_id": "test_user_001",
        "session_id": str(uuid.uuid4()),
        "original_query": "俯卧撑怎么做？",
        "rewritten_query": "",
        "expanded_queries": [],
        "intent": "unknown",
        "intent_confidence": 0.0,
        "sub_intents": [],
        "need_clarification": False,
        "clarification_question": "",
        "clarification_context": {},
        "user_profile": {
            "fitness_level": "beginner",
            "primary_goal": "maintain_health",
            "available_equipment": ["speediance_gym_monster"],
            "weekly_workout_days": 3,
            "injury_history": [],
            "health_conditions": [],
            "profile_summary": "初学者，目标维持健康，每周3天锻炼",
        },
        "retrieved_docs": [],
        "course_results": [],
        "reminder_info": {},
        "diet_info": {},
        "exercise_info": {},
        "plan_iteration": 0,
        "plan_thoughts": "",
        "plan_draft": "",
        "plan_reflection": "",
        "plan_is_complete": False,
        "final_response": "",
        "structured_output": {},
        "metadata": {},
        "error": None,
    }


@pytest.fixture
def mock_llm_response():
    """Mock LLM 响应工厂"""
    def _make(content: str):
        mock = AsyncMock()
        mock.content = content
        return mock
    return _make


# =============================================================
# 1. 测试 query_rewriter_node
# =============================================================

class TestQueryRewriter:
    """查询改写节点测试"""

    @pytest.mark.asyncio
    async def test_rewrite_simple_query(self, base_state, mock_llm_response):
        """简单查询应该能正常改写"""
        from fitness_agent.graph.nodes.query_rewriter import query_rewriter_node

        with patch("fitness_agent.graph.nodes.query_rewriter.ChatOpenAI") as MockLLM:
            instance = MockLLM.return_value
            # ainvoke 被调用两次：改写 + 扩展
            instance.ainvoke = AsyncMock(side_effect=[
                mock_llm_response("俯卧撑的标准动作怎么做？"),
                mock_llm_response('["俯卧撑标准动作", "俯卧撑初学者教程", "胸肌训练方法"]'),
            ])

            result = await query_rewriter_node(base_state)

            assert result["original_query"] == "俯卧撑怎么做？"
            assert result["rewritten_query"] == "俯卧撑的标准动作怎么做？"
            assert len(result["expanded_queries"]) == 3

    @pytest.mark.asyncio
    async def test_rewrite_empty_query(self, base_state):
        """空查询应该返回空结果而不报错"""
        from fitness_agent.graph.nodes.query_rewriter import query_rewriter_node

        state = {**base_state, "messages": [], "original_query": ""}
        result = await query_rewriter_node(state)

        assert result["original_query"] == ""
        assert result["rewritten_query"] == ""
        assert result["expanded_queries"] == []

    @pytest.mark.asyncio
    async def test_fallback_on_llm_error(self, base_state):
        """LLM 调用失败时应该降级使用原始查询"""
        from fitness_agent.graph.nodes.query_rewriter import query_rewriter_node

        with patch("fitness_agent.graph.nodes.query_rewriter.ChatOpenAI") as MockLLM:
            instance = MockLLM.return_value
            instance.ainvoke = AsyncMock(side_effect=Exception("API 超时"))

            result = await query_rewriter_node(base_state)

            # 降级：原始查询 = 改写查询
            assert result["rewritten_query"] == base_state["original_query"]


# =============================================================
# 2. 测试 intent_classifier_node
# =============================================================

class TestIntentClassifier:
    """意图识别节点测试"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,expected_intent", [
        ("俯卧撑怎么做", "qa"),
        ("推荐一个减脂课程", "course_recommendation"),
        ("帮我定个明天早上7点起床的提醒", "set_reminder"),
        ("我今天吃了一碗米饭", "record_diet_exercise"),
        ("帮我制定一个4周增肌计划", "generate_workout_plan"),
    ])
    async def test_intent_classification(
        self,
        base_state,
        mock_llm_response,
        query: str,
        expected_intent: str,
    ):
        """不同查询应该被正确分类"""
        from fitness_agent.graph.nodes.intent_classifier import intent_classifier_node
        import json

        state = {
            **base_state,
            "rewritten_query": query,
            "original_query": query,
        }

        llm_output = json.dumps({
            "intent": expected_intent,
            "confidence": 0.92,
            "sub_intents": [],
            "reasoning": f"用户意图明确是 {expected_intent}",
        }, ensure_ascii=False)

        with patch("fitness_agent.graph.nodes.intent_classifier.ChatOpenAI") as MockLLM:
            instance = MockLLM.return_value
            instance.ainvoke = AsyncMock(return_value=mock_llm_response(llm_output))

            result = await intent_classifier_node(state)

            assert result["intent"] == expected_intent
            assert result["intent_confidence"] >= 0.9

    @pytest.mark.asyncio
    async def test_mixed_intent(self, base_state, mock_llm_response):
        """混合意图（饮食记录+提醒）应该正确识别"""
        from fitness_agent.graph.nodes.intent_classifier import intent_classifier_node
        import json

        state = {
            **base_state,
            "rewritten_query": "记录下我刚吃的早饭，顺便提醒我下午3点去健身",
        }

        llm_output = json.dumps({
            "intent": "record_diet_exercise",
            "confidence": 0.88,
            "sub_intents": ["set_reminder"],
            "reasoning": "主意图是记录饮食，次意图是设置提醒",
        }, ensure_ascii=False)

        with patch("fitness_agent.graph.nodes.intent_classifier.ChatOpenAI") as MockLLM:
            instance = MockLLM.return_value
            instance.ainvoke = AsyncMock(return_value=mock_llm_response(llm_output))

            result = await intent_classifier_node(state)

            assert result["intent"] == "record_diet_exercise"
            assert "set_reminder" in result["sub_intents"]

    @pytest.mark.asyncio
    async def test_invalid_intent_falls_back(self, base_state, mock_llm_response):
        """非法意图应该降级为 unknown"""
        from fitness_agent.graph.nodes.intent_classifier import intent_classifier_node
        import json

        state = {**base_state, "rewritten_query": "你好"}

        llm_output = json.dumps({
            "intent": "invalid_intent_xyz",
            "confidence": 0.9,
            "sub_intents": [],
        })

        with patch("fitness_agent.graph.nodes.intent_classifier.ChatOpenAI") as MockLLM:
            instance = MockLLM.return_value
            instance.ainvoke = AsyncMock(return_value=mock_llm_response(llm_output))

            result = await intent_classifier_node(state)

            assert result["intent"] == "unknown"


# =============================================================
# 3. 测试图路由函数
# =============================================================

class TestGraphRouting:
    """LangGraph 条件路由测试"""

    def test_route_qa_intent(self, base_state):
        """qa 意图应路由到 qa_node"""
        from fitness_agent.graph.graph import route_by_intent

        state = {**base_state, "intent": "qa"}
        assert route_by_intent(state) == "qa_node"

    def test_route_course_recommendation(self, base_state):
        """course_recommendation 意图应路由到推荐节点"""
        from fitness_agent.graph.graph import route_by_intent

        state = {**base_state, "intent": "course_recommendation"}
        assert route_by_intent(state) == "course_recommendation_node"

    def test_route_workout_plan(self, base_state):
        """generate_workout_plan 意图应路由到构思节点"""
        from fitness_agent.graph.graph import route_by_intent

        state = {**base_state, "intent": "generate_workout_plan"}
        assert route_by_intent(state) == "workout_plan_think_node"

    def test_route_unknown_falls_back_to_qa(self, base_state):
        """未知意图应降级到 qa_node"""
        from fitness_agent.graph.graph import route_by_intent

        state = {**base_state, "intent": "unknown"}
        assert route_by_intent(state) == "qa_node"

    def test_workout_plan_iterate_condition(self, base_state):
        """计划未达标时应继续迭代"""
        from fitness_agent.graph.graph import should_iterate_or_format

        state = {**base_state, "plan_is_complete": False, "plan_iteration": 1}
        assert should_iterate_or_format(state) == "workout_plan_think_node"

    def test_workout_plan_format_condition(self, base_state):
        """计划达标时应进入格式化"""
        from fitness_agent.graph.graph import should_iterate_or_format

        state = {**base_state, "plan_is_complete": True, "plan_iteration": 1}
        assert should_iterate_or_format(state) == "workout_plan_format_node"

    def test_workout_plan_max_iteration_forces_format(self, base_state):
        """超过最大迭代次数应强制格式化"""
        from fitness_agent.graph.graph import should_iterate_or_format

        state = {**base_state, "plan_is_complete": False, "plan_iteration": 3}
        assert should_iterate_or_format(state) == "workout_plan_format_node"


# =============================================================
# 4. 测试 qa_node
# =============================================================

class TestQANode:
    """问答节点测试"""

    @pytest.mark.asyncio
    async def test_qa_with_retrieved_docs(self, base_state, mock_llm_response):
        """有检索文档时应生成引用来源的回答"""
        from fitness_agent.graph.nodes.qa_node import qa_node

        state = {
            **base_state,
            "rewritten_query": "俯卧撑的正确姿势是什么？",
            "retrieved_docs": [
                {
                    "chunk_id": "chunk-001",
                    "document_id": "doc-001",
                    "title": "健身动作指南",
                    "content": "俯卧撑时保持身体成一条直线，核心收紧...",
                    "score": 0.92,
                }
            ],
        }

        with patch("fitness_agent.graph.nodes.qa_node.ChatOpenAI") as MockLLM:
            instance = MockLLM.return_value
            # mock 信息充足检查 + 生成回答
            instance.ainvoke = AsyncMock(side_effect=[
                mock_llm_response('{"has_enough_info": true}'),
                mock_llm_response("俯卧撑的正确姿势：保持全身挺直，手掌与肩同宽... [参考1]"),
            ])

            result = await qa_node(state)

            assert result["need_clarification"] == False
            assert len(result["final_response"]) > 10
            assert len(result["messages"]) > 0

    @pytest.mark.asyncio
    async def test_qa_clarification_when_no_docs(self, base_state, mock_llm_response):
        """信息不足时应触发追问"""
        from fitness_agent.graph.nodes.qa_node import qa_node

        state = {
            **base_state,
            "rewritten_query": "适合我的运动强度是多少？",
            "retrieved_docs": [],
        }

        with patch("fitness_agent.graph.nodes.qa_node.ChatOpenAI") as MockLLM:
            with patch("fitness_agent.graph.nodes.qa_node.HybridRetriever") as MockRetriever:
                with patch("fitness_agent.graph.nodes.qa_node.GteReranker") as MockReranker:
                    MockRetriever.return_value.retrieve_documents = AsyncMock(return_value=[])
                    instance = MockLLM.return_value
                    instance.ainvoke = AsyncMock(return_value=mock_llm_response(
                        '{"has_enough_info": false, "clarification_question": "请问您的当前健身水平和主要目标是什么？"}'
                    ))

                    result = await qa_node(state)

                    assert result["need_clarification"] == True
                    assert "clarification_question" in result
                    assert len(result["clarification_question"]) > 5


# =============================================================
# 5. 测试 reminder_node
# =============================================================

class TestReminderNode:
    """提醒节点测试"""

    @pytest.mark.asyncio
    async def test_parse_reminder_success(self, base_state, mock_llm_response):
        """成功解析提醒信息并写入"""
        from fitness_agent.graph.nodes.reminder_node import reminder_node
        import json

        state = {
            **base_state,
            "rewritten_query": "明天下午3点提醒我健身",
        }

        parsed_reminder = json.dumps({
            "has_enough_info": True,
            "title": "健身提醒",
            "description": "该去健身了",
            "reminder_type": "workout",
            "remind_at_iso": "2024-01-15T15:00:00+08:00",
            "recurrence_rule": None,
        }, ensure_ascii=False)

        with patch("fitness_agent.graph.nodes.reminder_node.ChatOpenAI") as MockLLM:
            with patch("fitness_agent.graph.nodes.reminder_node.AppAPIClient") as MockAPI:
                with patch("fitness_agent.graph.nodes.reminder_node.create_engine"):
                    with patch("fitness_agent.graph.nodes.reminder_node.get_session"):
                        instance = MockLLM.return_value
                        instance.ainvoke = AsyncMock(return_value=mock_llm_response(parsed_reminder))

                        # Mock App API
                        mock_api_instance = AsyncMock()
                        mock_api_instance.__aenter__ = AsyncMock(return_value=mock_api_instance)
                        mock_api_instance.__aexit__ = AsyncMock(return_value=None)
                        mock_api_instance.create_reminder = AsyncMock(return_value={"reminder_id": "app-123"})
                        MockAPI.return_value = mock_api_instance

                        result = await reminder_node(state)

            assert result["need_clarification"] == False
            assert "已为您设置提醒" in result["final_response"]

    @pytest.mark.asyncio
    async def test_reminder_missing_time(self, base_state, mock_llm_response):
        """缺少时间信息时应追问"""
        from fitness_agent.graph.nodes.reminder_node import reminder_node
        import json

        state = {**base_state, "rewritten_query": "提醒我健身"}

        parsed = json.dumps({
            "has_enough_info": False,
            "missing_fields": ["time"],
            "clarification_question": "请问您想几点收到提醒？",
        })

        with patch("fitness_agent.graph.nodes.reminder_node.ChatOpenAI") as MockLLM:
            instance = MockLLM.return_value
            instance.ainvoke = AsyncMock(return_value=mock_llm_response(parsed))

            result = await reminder_node(state)

            assert result["need_clarification"] == True
            assert "提醒" in result["clarification_question"]


# =============================================================
# 6. 测试 diet_exercise_recorder_node
# =============================================================

class TestDietExerciseRecorder:
    """饮食/运动记录节点测试"""

    @pytest.mark.asyncio
    async def test_diet_record(self, base_state, mock_llm_response):
        """饮食记录应正确解析营养信息"""
        from fitness_agent.graph.nodes.diet_exercise_recorder import diet_exercise_recorder_node
        import json

        state = {
            **base_state,
            "rewritten_query": "早上吃了一碗米饭和两个鸡蛋",
        }

        diet_parsed = json.dumps({
            "has_diet_info": True,
            "meal_type": "breakfast",
            "recorded_at_iso": "2024-01-15T08:00:00+08:00",
            "food_items": [
                {"name": "米饭", "amount_g": 200, "calories_kcal": 232, "protein_g": 4.3, "carbs_g": 50.9, "fat_g": 0.5, "fiber_g": 0.5},
                {"name": "鸡蛋", "amount_g": 100, "calories_kcal": 144, "protein_g": 12.4, "carbs_g": 1.2, "fat_g": 9.6, "fiber_g": 0.0},
            ],
            "total_calories_kcal": 376,
            "total_protein_g": 16.7,
            "total_carbs_g": 52.1,
            "total_fat_g": 10.1,
            "estimate_confidence": 0.9,
        }, ensure_ascii=False)

        exercise_parsed = json.dumps({"has_exercise_info": False})

        with patch("fitness_agent.graph.nodes.diet_exercise_recorder.ChatOpenAI") as MockLLM:
            with patch("fitness_agent.graph.nodes.diet_exercise_recorder.create_engine"):
                with patch("fitness_agent.graph.nodes.diet_exercise_recorder.get_session"):
                    instance = MockLLM.return_value
                    instance.ainvoke = AsyncMock(side_effect=[
                        mock_llm_response(diet_parsed),
                        mock_llm_response(exercise_parsed),
                    ])

                    result = await diet_exercise_recorder_node(state)

                    assert result["need_clarification"] == False
                    assert "376" in result["final_response"] or "早餐" in result["final_response"]
                    assert result["diet_info"].get("has_diet_info") == True

    @pytest.mark.asyncio
    async def test_exercise_record(self, base_state, mock_llm_response):
        """运动记录应正确解析消耗信息"""
        from fitness_agent.graph.nodes.diet_exercise_recorder import diet_exercise_recorder_node
        import json

        state = {
            **base_state,
            "rewritten_query": "刚刚跑步30分钟",
        }

        diet_parsed = json.dumps({"has_diet_info": False})
        exercise_parsed = json.dumps({
            "has_exercise_info": True,
            "exercise_type": "cardio",
            "duration_min": 30,
            "recorded_at_iso": "2024-01-15T07:00:00+08:00",
            "exercise_items": [{"name": "跑步", "duration_min": 30, "calories_kcal": 250}],
            "calories_burned_kcal": 250,
            "estimate_confidence": 0.85,
        }, ensure_ascii=False)

        with patch("fitness_agent.graph.nodes.diet_exercise_recorder.ChatOpenAI") as MockLLM:
            with patch("fitness_agent.graph.nodes.diet_exercise_recorder.create_engine"):
                with patch("fitness_agent.graph.nodes.diet_exercise_recorder.get_session"):
                    instance = MockLLM.return_value
                    instance.ainvoke = AsyncMock(side_effect=[
                        mock_llm_response(diet_parsed),
                        mock_llm_response(exercise_parsed),
                    ])

                    result = await diet_exercise_recorder_node(state)

                    assert "30" in result["final_response"] or "运动" in result["final_response"]
                    assert result["exercise_info"].get("has_exercise_info") == True


# =============================================================
# 7. 测试运动计划 Think-Execute-Reflect 循环
# =============================================================

class TestWorkoutPlanGenerator:
    """运动计划生成节点测试"""

    @pytest.mark.asyncio
    async def test_think_with_enough_info(self, base_state, mock_llm_response):
        """用户画像完整时应进入执行阶段"""
        from fitness_agent.graph.nodes.workout_plan_generator import workout_plan_think_node
        import json

        state = {
            **base_state,
            "rewritten_query": "帮我制定一个4周增肌计划",
            "plan_iteration": 0,
        }

        think_output = json.dumps({
            "has_enough_info": True,
            "analysis": {
                "fitness_assessment": "初学者，基础良好",
                "goal_analysis": "增肌目标可行，建议渐进超负荷",
                "constraint_analysis": "有 Gym Monster，可做复合动作",
                "training_principle": "分化训练，每周3天",
                "plan_framework": "推拉腿三分化",
            }
        }, ensure_ascii=False)

        with patch("fitness_agent.graph.nodes.workout_plan_generator.ChatOpenAI") as MockLLM:
            instance = MockLLM.return_value
            instance.ainvoke = AsyncMock(return_value=mock_llm_response(think_output))

            result = await workout_plan_think_node(state)

            assert result["need_clarification"] == False
            assert result["plan_thoughts"] != ""
            assert result["plan_iteration"] == 0

    @pytest.mark.asyncio
    async def test_reflect_marks_complete(self, base_state, mock_llm_response):
        """高质量计划反思后应标记为完成"""
        from fitness_agent.graph.nodes.workout_plan_generator import workout_plan_reflect_node
        import json

        state = {
            **base_state,
            "plan_iteration": 0,
            "plan_thoughts": "分析完毕",
            "plan_draft": json.dumps({
                "plan_name": "4周增肌计划",
                "goal": "增肌",
                "duration_weeks": 4,
                "weekly_schedule": [],
            }),
        }

        reflect_output = json.dumps({
            "is_complete": True,
            "score": 8.8,
            "issues": [],
            "need_revision": False,
            "reflection_summary": "计划完整、安全、个性化程度高",
        }, ensure_ascii=False)

        with patch("fitness_agent.graph.nodes.workout_plan_generator.ChatOpenAI") as MockLLM:
            instance = MockLLM.return_value
            instance.ainvoke = AsyncMock(return_value=mock_llm_response(reflect_output))

            result = await workout_plan_reflect_node(state)

            assert result["plan_is_complete"] == True
            assert result["plan_iteration"] == 1

    @pytest.mark.asyncio
    async def test_reflect_triggers_retry_when_score_low(self, base_state, mock_llm_response):
        """低分计划反思后应触发重试"""
        from fitness_agent.graph.nodes.workout_plan_generator import workout_plan_reflect_node
        import json

        state = {
            **base_state,
            "plan_iteration": 0,
            "plan_thoughts": "分析完毕",
            "plan_draft": '{"plan_name": "计划"}',
        }

        reflect_output = json.dumps({
            "is_complete": False,
            "score": 6.5,
            "issues": ["缺少渐进超负荷说明", "未考虑用户伤病"],
            "need_revision": True,
        }, ensure_ascii=False)

        with patch("fitness_agent.graph.nodes.workout_plan_generator.ChatOpenAI") as MockLLM:
            instance = MockLLM.return_value
            instance.ainvoke = AsyncMock(return_value=mock_llm_response(reflect_output))

            result = await workout_plan_reflect_node(state)

            assert result["plan_is_complete"] == False
            assert result["plan_iteration"] == 1


# =============================================================
# 8. 测试 RedisShortTermMemory
# =============================================================

class TestRedisShortTermMemory:
    """Redis 短期记忆测试（mock Redis）"""

    @pytest.mark.asyncio
    async def test_add_and_get_messages(self):
        """消息应该能正常存入和取出"""
        from fitness_agent.memory.redis_short_term import RedisShortTermMemory
        import json

        memory = RedisShortTermMemory()
        session_id = str(uuid.uuid4())

        # Mock Redis 客户端
        mock_redis = AsyncMock()
        mock_pipeline = AsyncMock()
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=None)
        mock_pipeline.rpush = MagicMock()
        mock_pipeline.ltrim = MagicMock()
        mock_pipeline.expire = MagicMock()
        mock_pipeline.execute = AsyncMock(return_value=[1, True, True])
        mock_redis.pipeline = MagicMock(return_value=mock_pipeline)

        stored_messages = []

        async def mock_lrange(key, start, end):
            return [json.dumps(m, ensure_ascii=False) for m in stored_messages]

        mock_redis.lrange = mock_lrange
        mock_redis.llen = AsyncMock(return_value=len(stored_messages))

        memory._redis_client = mock_redis

        # 添加消息（模拟 pipeline 执行存储）
        await memory.add_message(session_id, "user", "你好")
        stored_messages.append({"role": "user", "content": "你好", "timestamp": "2024-01-01T00:00:00", "metadata": {}})

        # 读取消息
        messages = await memory.get_recent_messages(session_id)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "你好"

    @pytest.mark.asyncio
    async def test_cache_and_get_user_profile(self):
        """用户画像缓存应正确存取"""
        from fitness_agent.memory.redis_short_term import RedisShortTermMemory
        import json

        memory = RedisShortTermMemory()
        user_id = str(uuid.uuid4())
        profile = {"fitness_level": "intermediate", "weight_kg": 70.0}

        mock_redis = AsyncMock()
        cached_value = None

        async def mock_setex(key, ttl, value):
            nonlocal cached_value
            cached_value = value

        async def mock_get(key):
            return cached_value

        mock_redis.setex = mock_setex
        mock_redis.get = mock_get
        memory._redis_client = mock_redis

        await memory.cache_user_profile(user_id, profile)
        retrieved = await memory.get_cached_user_profile(user_id)

        assert retrieved is not None
        assert retrieved["fitness_level"] == "intermediate"
        assert retrieved["weight_kg"] == 70.0


# =============================================================
# 9. 集成测试（使用真实图结构，mock 所有 I/O）
# =============================================================

class TestGraphIntegration:
    """图集成测试：验证节点间数据流正确"""

    @pytest.mark.asyncio
    async def test_graph_qa_flow(self):
        """完整 QA 流程：load_profile → rewrite → classify → qa → save"""
        from fitness_agent.graph.graph import build_graph
        from langchain_core.messages import HumanMessage

        # 构建无 checkpointer 的测试图
        graph = build_graph()

        initial_state = {
            "messages": [HumanMessage(content="俯卧撑的正确姿势？")],
            "user_id": str(uuid.uuid4()),
            "app_user_id": "test_user",
            "session_id": str(uuid.uuid4()),
            "original_query": "俯卧撑的正确姿势？",
            "rewritten_query": "",
            "expanded_queries": [],
            "intent": "unknown",
            "intent_confidence": 0.0,
            "sub_intents": [],
            "need_clarification": False,
            "clarification_question": "",
            "clarification_context": {},
            "user_profile": {},
            "retrieved_docs": [],
            "course_results": [],
            "reminder_info": {},
            "diet_info": {},
            "exercise_info": {},
            "plan_iteration": 0,
            "plan_thoughts": "",
            "plan_draft": "",
            "plan_reflection": "",
            "plan_is_complete": False,
            "final_response": "",
            "structured_output": {},
            "metadata": {},
            "error": None,
        }

        import json

        # Mock 所有外部调用
        with patch("fitness_agent.graph.graph.get_session"), \
             patch("fitness_agent.graph.graph.create_engine"), \
             patch("fitness_agent.graph.nodes.query_rewriter.ChatOpenAI") as MockRewriteLLM, \
             patch("fitness_agent.graph.nodes.intent_classifier.ChatOpenAI") as MockIntentLLM, \
             patch("fitness_agent.graph.nodes.qa_node.ChatOpenAI") as MockQALLM, \
             patch("fitness_agent.graph.nodes.qa_node.HybridRetriever") as MockRetriever, \
             patch("fitness_agent.graph.nodes.qa_node.GteReranker") as MockReranker, \
             patch("fitness_agent.graph.graph.ChatMessage"), \
             patch("fitness_agent.graph.graph.ChatSession"):

            # Mock 改写
            rewrite_llm = MockRewriteLLM.return_value
            rewrite_llm.ainvoke = AsyncMock(side_effect=[
                MagicMock(content="俯卧撑的标准动作和注意事项是什么？"),
                MagicMock(content='["俯卧撑教程", "俯卧撑变式", "胸肌训练"]'),
            ])

            # Mock 意图
            intent_llm = MockIntentLLM.return_value
            intent_llm.ainvoke = AsyncMock(return_value=MagicMock(
                content=json.dumps({
                    "intent": "qa", "confidence": 0.95,
                    "sub_intents": [], "reasoning": "用户提问",
                })
            ))

            # Mock QA
            qa_llm = MockQALLM.return_value
            qa_llm.ainvoke = AsyncMock(side_effect=[
                MagicMock(content='{"has_enough_info": true}'),
                MagicMock(content="俯卧撑的标准姿势：双手与肩同宽..."),
            ])
            MockRetriever.return_value.retrieve_documents = AsyncMock(return_value=[])
            MockReranker.return_value.rerank = AsyncMock(return_value=[])

            config = {"configurable": {"thread_id": initial_state["session_id"]}}
            final_state = await graph.ainvoke(initial_state, config=config)

            # 验证数据流
            assert final_state.get("intent") == "qa"
            assert final_state.get("rewritten_query") != ""
            # final_response 可能来自 qa_node 或 save_message_node
            assert "final_response" in final_state
