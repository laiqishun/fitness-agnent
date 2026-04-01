-- =============================================================
-- Speediance AI 私教智能体 - PostgreSQL Schema
-- 依赖扩展: pgvector, uuid-ossp, pg_trgm
-- =============================================================

-- 启用必要扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- =============================================================
-- 枚举类型
-- =============================================================

CREATE TYPE gender_type AS ENUM ('male', 'female', 'other', 'unknown');
CREATE TYPE fitness_level_type AS ENUM ('beginner', 'intermediate', 'advanced', 'elite');
CREATE TYPE goal_type AS ENUM ('lose_weight', 'gain_muscle', 'improve_endurance', 'maintain_health', 'rehabilitation', 'other');
CREATE TYPE message_role AS ENUM ('user', 'assistant', 'system', 'tool');
CREATE TYPE intent_type AS ENUM ('qa', 'course_recommendation', 'set_reminder', 'record_diet_exercise', 'generate_workout_plan', 'need_clarification', 'unknown');
CREATE TYPE session_status AS ENUM ('active', 'summarized', 'archived');
CREATE TYPE reminder_status AS ENUM ('pending', 'sent', 'cancelled', 'failed');
CREATE TYPE document_status AS ENUM ('active', 'deprecated', 'processing');
CREATE TYPE course_difficulty AS ENUM ('easy', 'moderate', 'hard', 'extreme');
CREATE TYPE meal_type AS ENUM ('breakfast', 'lunch', 'dinner', 'snack', 'other');
CREATE TYPE exercise_type AS ENUM ('strength', 'cardio', 'flexibility', 'balance', 'hiit', 'other');

-- =============================================================
-- 1. users 表（基础用户信息）
-- =============================================================

CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    app_user_id     VARCHAR(128) UNIQUE NOT NULL,       -- App 侧用户 ID
    username        VARCHAR(64),
    email           VARCHAR(256),
    phone           VARCHAR(32),
    gender          gender_type DEFAULT 'unknown',
    birth_date      DATE,
    timezone        VARCHAR(64) DEFAULT 'Asia/Shanghai',
    locale          VARCHAR(16) DEFAULT 'zh-CN',
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    deleted_at      TIMESTAMPTZ                          -- 软删除
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_users_app_user_id   ON users(app_user_id);
CREATE INDEX IF NOT EXISTS idx_users_email          ON users(email) WHERE email IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_users_deleted_at     ON users(deleted_at) WHERE deleted_at IS NULL;

COMMENT ON TABLE users IS '基础用户表';
COMMENT ON COLUMN users.app_user_id IS 'App 侧透传的用户唯一标识';

-- =============================================================
-- 2. user_profiles 表（用户画像，含向量）
-- =============================================================

CREATE TABLE IF NOT EXISTS user_profiles (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id             UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    -- 体征数据
    height_cm           NUMERIC(5,1),
    weight_kg           NUMERIC(5,1),
    body_fat_pct        NUMERIC(4,1),
    bmi                 NUMERIC(4,1) GENERATED ALWAYS AS (
                            CASE WHEN height_cm > 0 THEN
                                ROUND((weight_kg / ((height_cm / 100.0) ^ 2))::NUMERIC, 1)
                            ELSE NULL END
                        ) STORED,
    -- 健身目标与水平
    fitness_level       fitness_level_type DEFAULT 'beginner',
    primary_goal        goal_type DEFAULT 'maintain_health',
    secondary_goals     goal_type[] DEFAULT '{}',
    -- 器械与环境
    available_equipment TEXT[] DEFAULT '{}',            -- 例如 ['speediance_gym_monster', 'dumbbell', 'yoga_mat']
    workout_location    VARCHAR(32) DEFAULT 'home',     -- home | gym | outdoor
    -- 健康状况
    injury_history      TEXT[] DEFAULT '{}',
    health_conditions   TEXT[] DEFAULT '{}',            -- 慢性病、禁忌症
    -- 运动习惯
    weekly_workout_days INTEGER DEFAULT 3 CHECK (weekly_workout_days BETWEEN 0 AND 7),
    preferred_workout_duration_min INTEGER DEFAULT 30,
    preferred_workout_time VARCHAR(32),                 -- morning | afternoon | evening | flexible
    -- 饮食偏好
    dietary_restrictions TEXT[] DEFAULT '{}',           -- vegetarian, vegan, gluten_free 等
    -- 画像向量（用于相似度匹配）
    profile_embedding   VECTOR(1536),
    -- 文本摘要（供 LLM 快速读取）
    profile_summary     TEXT,
    -- 元数据
    is_current          BOOLEAN DEFAULT TRUE,           -- 当前生效画像
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id      ON user_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_user_profiles_current      ON user_profiles(user_id) WHERE is_current = TRUE;
CREATE INDEX IF NOT EXISTS idx_user_profiles_embedding    ON user_profiles USING ivfflat (profile_embedding vector_cosine_ops)
    WITH (lists = 100);

COMMENT ON TABLE user_profiles IS '用户健身画像，每次更新追加新记录并设置 is_current';

-- =============================================================
-- 3. user_profiles_update_history 表（画像变更历史）
-- =============================================================

CREATE TABLE IF NOT EXISTS user_profiles_update_history (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    profile_id      UUID NOT NULL REFERENCES user_profiles(id) ON DELETE CASCADE,
    changed_fields  JSONB NOT NULL DEFAULT '{}',         -- {'weight_kg': {'old': 75, 'new': 73}}
    change_reason   VARCHAR(256),                        -- 'user_input' | 'session_summary' | 'admin'
    session_id      UUID,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_profile_history_user_id  ON user_profiles_update_history(user_id);
CREATE INDEX IF NOT EXISTS idx_profile_history_created  ON user_profiles_update_history(created_at DESC);

COMMENT ON TABLE user_profiles_update_history IS '用户画像变更历史，支持审计和回滚';

-- =============================================================
-- 4. chat_sessions 表（会话表）
-- =============================================================

CREATE TABLE IF NOT EXISTS chat_sessions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    -- 状态与摘要
    status          session_status DEFAULT 'active',
    title           VARCHAR(256),                        -- 会话标题（自动生成或用户命名）
    summary         TEXT,                                -- 会话摘要（由 session_summarizer 生成）
    key_facts       JSONB DEFAULT '[]',                  -- 摘要中提取的关键事实列表
    -- 统计
    message_count   INTEGER DEFAULT 0,
    total_tokens    INTEGER DEFAULT 0,
    -- 最后活跃
    last_message_at TIMESTAMPTZ DEFAULT NOW(),
    -- 元数据
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    archived_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id         ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status          ON chat_sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_last_message    ON chat_sessions(user_id, last_message_at DESC);

COMMENT ON TABLE chat_sessions IS '对话会话表，每个会话有独立摘要用于长期记忆压缩';

-- =============================================================
-- 5. chat_messages 表（消息表，含向量索引）
-- =============================================================

CREATE TABLE IF NOT EXISTS chat_messages (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    -- 消息内容
    role            message_role NOT NULL,
    content         TEXT NOT NULL,
    -- 意图与元数据
    intent          intent_type,
    intent_confidence NUMERIC(3,2),
    tool_calls      JSONB DEFAULT '[]',                  -- 工具调用记录
    -- 向量（用于语义检索历史消息）
    content_embedding VECTOR(1536),
    -- token 统计
    prompt_tokens   INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    -- 顺序
    sequence_num    INTEGER NOT NULL,                    -- 会话内消息序号
    -- 元数据
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 复合唯一索引保证消息顺序
CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_session_seq ON chat_messages(session_id, sequence_num);
CREATE INDEX IF NOT EXISTS idx_messages_session_id       ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_user_id          ON chat_messages(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at       ON chat_messages(session_id, created_at DESC);
-- 向量索引（用于语义相似检索历史消息）
CREATE INDEX IF NOT EXISTS idx_messages_embedding        ON chat_messages USING ivfflat (content_embedding vector_cosine_ops)
    WITH (lists = 100)
    WHERE content_embedding IS NOT NULL;

COMMENT ON TABLE chat_messages IS '消息明细表，role=tool 时 content 为工具返回结果';

-- =============================================================
-- 6. source_documents 表（原始文档）
-- =============================================================

CREATE TABLE IF NOT EXISTS source_documents (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title           VARCHAR(512) NOT NULL,
    description     TEXT,
    source_type     VARCHAR(64) DEFAULT 'article',       -- article | video | pdf | guideline
    author          VARCHAR(256),
    published_date  DATE,
    -- OSS 存储
    oss_bucket      VARCHAR(128),
    oss_key         VARCHAR(512),
    oss_url         TEXT,                                -- 预签名或公开 URL
    -- 内容
    content_hash    VARCHAR(64),                         -- SHA256，防重复入库
    raw_content     TEXT,                                -- 原始文本（可选，大文件存 OSS）
    language        VARCHAR(16) DEFAULT 'zh',
    -- 状态
    status          document_status DEFAULT 'active',
    chunk_count     INTEGER DEFAULT 0,
    -- 标签与分类
    tags            TEXT[] DEFAULT '{}',
    category        VARCHAR(128),                        -- 'strength_training' | 'nutrition' | 'recovery' 等
    -- 元数据
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_source_docs_hash  ON source_documents(content_hash) WHERE content_hash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_source_docs_status       ON source_documents(status);
CREATE INDEX IF NOT EXISTS idx_source_docs_category     ON source_documents(category);
CREATE INDEX IF NOT EXISTS idx_source_docs_tags         ON source_documents USING gin(tags);

COMMENT ON TABLE source_documents IS '知识库原始文档，支持 OSS 大文件存储';

-- =============================================================
-- 7. document_chunks 表（文档分块，含向量索引）
-- =============================================================

CREATE TABLE IF NOT EXISTS document_chunks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
    -- 分块内容
    chunk_index     INTEGER NOT NULL,                    -- 块在文档中的顺序
    content         TEXT NOT NULL,
    content_length  INTEGER GENERATED ALWAYS AS (char_length(content)) STORED,
    -- 向量
    embedding       VECTOR(1536) NOT NULL,
    -- 层次化分块：父块引用
    parent_chunk_id UUID REFERENCES document_chunks(id) ON DELETE SET NULL,
    -- 位置信息
    page_number     INTEGER,
    char_start      INTEGER,
    char_end        INTEGER,
    -- 关键词（用于混合检索）
    keywords        TEXT[] DEFAULT '{}',
    -- 元数据
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 向量索引（核心）
CREATE INDEX IF NOT EXISTS idx_chunks_embedding         ON document_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 256);
-- 全文检索索引
CREATE INDEX IF NOT EXISTS idx_chunks_content_fts       ON document_chunks USING gin(to_tsvector('simple', content));
-- trigram 索引（模糊匹配）
CREATE INDEX IF NOT EXISTS idx_chunks_content_trgm      ON document_chunks USING gin(content gin_trgm_ops);
-- 关键词数组索引
CREATE INDEX IF NOT EXISTS idx_chunks_keywords          ON document_chunks USING gin(keywords);
CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_doc_index  ON document_chunks(document_id, chunk_index);

COMMENT ON TABLE document_chunks IS '文档分块表，支持向量+关键词混合检索';

-- =============================================================
-- 8. courses 表（课程表，含向量）
-- =============================================================

CREATE TABLE IF NOT EXISTS courses (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    -- 基本信息
    name            VARCHAR(256) NOT NULL,
    name_en         VARCHAR(256),
    description     TEXT,
    instructor      VARCHAR(128),
    -- 分类与标签
    category        VARCHAR(128),                        -- 'strength' | 'cardio' | 'yoga' | 'hiit' 等
    tags            TEXT[] DEFAULT '{}',
    muscle_groups   TEXT[] DEFAULT '{}',                 -- 目标肌群
    equipment_needed TEXT[] DEFAULT '{}',                -- 所需器械
    -- 难度与时长
    difficulty      course_difficulty DEFAULT 'moderate',
    duration_min    INTEGER,                             -- 课程时长（分钟）
    -- 适合人群
    suitable_levels fitness_level_type[] DEFAULT '{}',
    suitable_goals  goal_type[] DEFAULT '{}',
    -- 媒体
    thumbnail_url   TEXT,
    video_url       TEXT,
    oss_key         VARCHAR(512),
    -- 向量（用于相似课程推荐）
    embedding       VECTOR(1536),
    -- 统计
    view_count      INTEGER DEFAULT 0,
    rating          NUMERIC(2,1) DEFAULT 0.0,
    rating_count    INTEGER DEFAULT 0,
    -- 状态
    is_active       BOOLEAN DEFAULT TRUE,
    is_premium      BOOLEAN DEFAULT FALSE,
    -- 元数据
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 向量索引
CREATE INDEX IF NOT EXISTS idx_courses_embedding        ON courses USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100)
    WHERE embedding IS NOT NULL;
-- 其他索引
CREATE INDEX IF NOT EXISTS idx_courses_category         ON courses(category);
CREATE INDEX IF NOT EXISTS idx_courses_difficulty       ON courses(difficulty);
CREATE INDEX IF NOT EXISTS idx_courses_tags             ON courses USING gin(tags);
CREATE INDEX IF NOT EXISTS idx_courses_muscle_groups    ON courses USING gin(muscle_groups);
CREATE INDEX IF NOT EXISTS idx_courses_equipment        ON courses USING gin(equipment_needed);
CREATE INDEX IF NOT EXISTS idx_courses_active           ON courses(is_active) WHERE is_active = TRUE;
-- 全文检索
CREATE INDEX IF NOT EXISTS idx_courses_name_fts         ON courses USING gin(to_tsvector('simple', name || ' ' || COALESCE(description, '')));

COMMENT ON TABLE courses IS '课程库，支持向量相似度 + 多维度筛选推荐';

-- =============================================================
-- 9. reminders 表（提醒记录）
-- =============================================================

CREATE TABLE IF NOT EXISTS reminders (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id      UUID REFERENCES chat_sessions(id) ON DELETE SET NULL,
    -- 提醒内容
    title           VARCHAR(256) NOT NULL,
    description     TEXT,
    reminder_type   VARCHAR(64) DEFAULT 'general',       -- 'workout' | 'meal' | 'medication' | 'general'
    -- 时间
    remind_at       TIMESTAMPTZ NOT NULL,
    timezone        VARCHAR(64) DEFAULT 'Asia/Shanghai',
    -- 重复规则（cron 表达式或 rrule）
    recurrence_rule VARCHAR(256),                        -- 例如 'FREQ=DAILY;INTERVAL=1'
    -- 状态
    status          reminder_status DEFAULT 'pending',
    -- App 侧同步
    app_reminder_id VARCHAR(128),                        -- App 返回的提醒 ID
    sent_at         TIMESTAMPTZ,
    -- 元数据
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reminders_user_id        ON reminders(user_id);
CREATE INDEX IF NOT EXISTS idx_reminders_remind_at      ON reminders(remind_at) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_reminders_status         ON reminders(status);

COMMENT ON TABLE reminders IS '用户提醒记录，与 App 推送系统双向同步';

-- =============================================================
-- 10. diet_records 表（饮食记录）
-- =============================================================

CREATE TABLE IF NOT EXISTS diet_records (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id      UUID REFERENCES chat_sessions(id) ON DELETE SET NULL,
    -- 时间
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- 实际饮食时间
    meal_type       meal_type DEFAULT 'other',
    -- 饮食内容（原始输入）
    raw_input       TEXT NOT NULL,                       -- 用户原文，例如"吃了一碗米饭和两个鸡蛋"
    -- 解析结果（JSON 数组，每项一种食物）
    food_items      JSONB NOT NULL DEFAULT '[]',
    -- 示例: [{"name":"米饭","amount_g":200,"calories_kcal":232,"protein_g":4.3,"carbs_g":50.9,"fat_g":0.5}]
    -- 汇总营养数据
    total_calories_kcal NUMERIC(7,1) DEFAULT 0,
    total_protein_g     NUMERIC(6,1) DEFAULT 0,
    total_carbs_g       NUMERIC(6,1) DEFAULT 0,
    total_fat_g         NUMERIC(6,1) DEFAULT 0,
    total_fiber_g       NUMERIC(6,1) DEFAULT 0,
    -- AI 估算置信度
    estimate_confidence NUMERIC(3,2) DEFAULT 0.8,
    notes           TEXT,
    -- 元数据
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_diet_records_user_id     ON diet_records(user_id);
CREATE INDEX IF NOT EXISTS idx_diet_records_recorded_at ON diet_records(user_id, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_diet_records_meal_type   ON diet_records(meal_type);

COMMENT ON TABLE diet_records IS '饮食记录，food_items 为 AI 解析的详细营养成分';

-- =============================================================
-- 11. exercise_records 表（运动记录）
-- =============================================================

CREATE TABLE IF NOT EXISTS exercise_records (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id      UUID REFERENCES chat_sessions(id) ON DELETE SET NULL,
    course_id       UUID REFERENCES courses(id) ON DELETE SET NULL,
    -- 时间
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- 实际运动时间
    duration_min    INTEGER,                             -- 运动时长（分钟）
    -- 运动内容（原始输入）
    raw_input       TEXT NOT NULL,
    exercise_type   exercise_type DEFAULT 'other',
    -- 解析结果（JSON 数组，每项一个动作/组）
    exercise_items  JSONB NOT NULL DEFAULT '[]',
    -- 示例: [{"name":"哑铃卧推","sets":3,"reps":12,"weight_kg":20,"rest_sec":60}]
    -- 消耗估算
    calories_burned_kcal NUMERIC(6,1) DEFAULT 0,
    avg_heart_rate  INTEGER,
    max_heart_rate  INTEGER,
    -- AI 估算置信度
    estimate_confidence NUMERIC(3,2) DEFAULT 0.8,
    notes           TEXT,
    -- 元数据
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_exercise_records_user_id    ON exercise_records(user_id);
CREATE INDEX IF NOT EXISTS idx_exercise_records_recorded   ON exercise_records(user_id, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_exercise_records_type       ON exercise_records(exercise_type);

COMMENT ON TABLE exercise_records IS '运动记录，exercise_items 为 AI 解析的详细动作数据';

-- =============================================================
-- LangGraph Checkpoint 表（pg_checkpointer 需要）
-- =============================================================

CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id       VARCHAR(128) NOT NULL,
    checkpoint_ns   VARCHAR(128) NOT NULL DEFAULT '',
    checkpoint_id   VARCHAR(128) NOT NULL,
    parent_checkpoint_id VARCHAR(128),
    type            VARCHAR(128),
    checkpoint      JSONB NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id       VARCHAR(128) NOT NULL,
    checkpoint_ns   VARCHAR(128) NOT NULL DEFAULT '',
    channel         VARCHAR(128) NOT NULL,
    version         VARCHAR(128) NOT NULL,
    type            VARCHAR(128) NOT NULL,
    blob            BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);

CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id       VARCHAR(128) NOT NULL,
    checkpoint_ns   VARCHAR(128) NOT NULL DEFAULT '',
    checkpoint_id   VARCHAR(128) NOT NULL,
    task_id         VARCHAR(128) NOT NULL,
    idx             INTEGER NOT NULL,
    channel         VARCHAR(128) NOT NULL,
    type            VARCHAR(128),
    blob            BYTEA NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_thread     ON checkpoints(thread_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_created    ON checkpoints(created_at DESC);

COMMENT ON TABLE checkpoints IS 'LangGraph PostgreSQL checkpointer 持久化表';

-- =============================================================
-- 触发器：自动更新 updated_at
-- =============================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 为各表创建触发器
DO $$
DECLARE
    t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'users', 'user_profiles', 'chat_sessions',
        'courses', 'reminders', 'source_documents'
    ]
    LOOP
        EXECUTE format(
            'CREATE OR REPLACE TRIGGER trg_%s_updated_at
             BEFORE UPDATE ON %s
             FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()',
            t, t
        );
    END LOOP;
END;
$$;

-- =============================================================
-- 触发器：自动维护 chat_sessions.message_count
-- =============================================================

CREATE OR REPLACE FUNCTION update_session_message_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE chat_sessions
        SET message_count  = message_count + 1,
            last_message_at = NEW.created_at
        WHERE id = NEW.session_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE chat_sessions
        SET message_count = GREATEST(0, message_count - 1)
        WHERE id = OLD.session_id;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trg_message_count
AFTER INSERT OR DELETE ON chat_messages
FOR EACH ROW EXECUTE FUNCTION update_session_message_count();
