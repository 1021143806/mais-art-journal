"""
配文生成器

为自拍图片生成配文：
- 基于当前活动/日程 + MaiBot 人设 + 表达风格自然生成
- LLM 生成（使用 MaiBot 的 replyer 模型）
- 生成失败返回空字符串，由调用方决定是否发布
"""

import datetime
import random
from typing import Optional

from src.common.logger import get_logger
from src.plugin_system.apis import llm_api, config_api

from .schedule_provider import ActivityInfo

logger = get_logger("auto_selfie.caption")


def _get_reply_style() -> str:
    """获取表达风格，支持 multiple_reply_style 随机替换"""
    reply_style = config_api.get_global_config("personality.reply_style", "")

    multi_styles = config_api.get_global_config("personality.multiple_reply_style", [])
    probability = config_api.get_global_config("personality.multiple_probability", 0.0)

    if multi_styles and probability > 0 and random.random() < probability:
        try:
            reply_style = random.choice(list(multi_styles))
        except Exception:
            pass

    return reply_style or ""


def _build_caption_prompt(activity_info: ActivityInfo, personality: str, reply_style: str) -> str:
    """构建配文生成 prompt"""
    now = datetime.datetime.now()
    time_str = now.strftime("%H:%M")

    prompt = f"""你是{personality}。

你的说话风格：{reply_style}

现在是{time_str}，你当前的状态：{activity_info.description}

你刚拍了一张自拍，准备发到社交媒体上，请写一段配文。

要求：
1. 用你自己的口吻和说话习惯来写，保持你平时的语气
2. 配文应该和你当前正在做的事有关联
3. 简短自然，像平时发朋友圈/说说一样（15-50字）
4. 可以适当用语气词、颜文字，但不要刻意堆砌
5. 不要用 hashtag、不要 @ 任何人
6. 只输出配文内容，不要输出其他任何东西

配文："""

    return prompt


async def generate_caption(
    activity_info: ActivityInfo,
) -> str:
    """
    为自拍生成配文

    基于当前活动 + MaiBot 人设 + 表达风格，由 LLM 自然生成。
    生成失败返回空字符串。

    Args:
        activity_info: 当前活动信息

    Returns:
        配文文本，失败时返回空字符串
    """
    # 获取人设和表达风格
    personality = config_api.get_global_config("personality.personality", "一个有趣的人")
    reply_style = _get_reply_style()

    try:
        prompt = _build_caption_prompt(activity_info, personality, reply_style)

        models = llm_api.get_available_models()
        model = models.get("replyer")
        if not model:
            logger.warning("未找到 replyer 模型，配文生成失败")
            return ""

        success, caption, _, _ = await llm_api.generate_with_model(
            prompt=prompt,
            model_config=model,
            request_type="plugin.auto_selfie_caption",
            temperature=0.85,
            max_tokens=8192,
        )

        if success and caption:
            # 清理输出
            caption = caption.strip().strip('"').strip("'").strip("「").strip("」")
            # 限制长度
            if len(caption) > 80:
                caption = caption[:77] + "..."
            if len(caption) < 2:
                logger.warning("LLM 返回配文过短，视为失败")
                return ""

            # 完整性检查：配文应以标点或表情结尾，否则可能被截断
            valid_endings = ("。", "！", "？", "~", "～", "…", ")", "）",
                            "」", "'", '"', "♪", "☆", "♡",
                            "呢", "哦", "啊", "呀", "吧", "了", "嘛", "哈", "噢", "耶")
            if len(caption) >= 8 and not caption.endswith(valid_endings):
                # 尝试截断到最后一个完整句子
                for punct in ("。", "！", "？", "~", "～", "…"):
                    last_pos = caption.rfind(punct)
                    if last_pos > 0:
                        caption = caption[:last_pos + 1]
                        break

            logger.info(f"LLM 生成配文: {caption}")
            return caption
        else:
            logger.warning("LLM 返回空响应，配文生成失败")
            return ""

    except Exception as e:
        logger.error(f"LLM 配文生成失败: {e}")
        return ""
