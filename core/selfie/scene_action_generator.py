"""
场景动作生成器

根据 ActivityInfo 生成符合情境的动作和 Stable Diffusion 提示词。

自动自拍：优先使用 LLM 根据活动描述生成英文 SD 场景标签，失败时取消。
手动自拍：使用确定性映射（get_action_for_activity）。
"""

import json
import re
from typing import Dict, List, Optional

from src.common.logger import get_logger

from .schedule_provider import ActivityInfo
from ..utils import ANTI_DUAL_HANDS_PROMPT

logger = get_logger("auto_selfie.scene")


# ==================== 确定性映射（手动自拍 + LLM 兜底） ====================

# 活动类型到动作的映射（每种类型一个固定值）
ACTIVITY_ACTIONS: Dict[str, str] = {
    "sleeping": "lying down, hugging pillow, cozy",
    "waking_up": "stretching, yawning, messy hair",
    "eating": "holding chopsticks, eating",
    "working": "typing on laptop, focused",
    "studying": "holding book, reading",
    "exercising": "stretching, athletic, holding water bottle",
    "relaxing": "lying on couch, relaxed, listening to music",
    "socializing": "making peace sign, happy, laughing",
    "commuting": "holding bag, walking, wearing earbuds",
    "hobby": "holding camera, creative",
    "self_care": "applying makeup, mirror",
    "other": "standing, casual pose, natural",
}

# 活动类型到场景环境的映射
ACTIVITY_ENVIRONMENTS: Dict[str, str] = {
    "sleeping": "bedroom, dim lighting, cozy atmosphere, bed",
    "waking_up": "bedroom, morning light, curtains, warm sunlight",
    "eating": "dining room, table setting",
    "working": "office desk, computer screen",
    "studying": "library, bookshelves, desk lamp",
    "exercising": "gym, fitness equipment",
    "relaxing": "living room, sofa, afternoon sun",
    "socializing": "outdoor cafe, bright atmosphere",
    "commuting": "city street, urban",
    "hobby": "art studio, creative space",
    "self_care": "bathroom, mirror, vanity",
    "other": "indoor, natural lighting",
}

# 活动类型到表情的映射
ACTIVITY_EXPRESSIONS: Dict[str, str] = {
    "sleeping": "peaceful expression, closed eyes",
    "waking_up": "drowsy expression, half-open eyes",
    "eating": "happy expression, enjoying food",
    "working": "focused expression, serious",
    "studying": "focused, thoughtful expression",
    "exercising": "energetic expression, determined",
    "relaxing": "relaxed smile, content",
    "socializing": "bright smile, happy",
    "commuting": "calm expression",
    "hobby": "excited, passionate",
    "self_care": "gentle smile, self-care",
    "other": "natural smile",
}

# 活动类型到光线的映射
ACTIVITY_LIGHTING: Dict[str, str] = {
    "sleeping": "dim warm light, night lamp",
    "waking_up": "soft morning light, golden hour",
    "eating": "warm indoor lighting",
    "working": "office lighting, even illumination",
    "studying": "desk lamp, focused light",
    "exercising": "bright natural light",
    "relaxing": "soft afternoon light, warm ambient light",
    "socializing": "bright cheerful lighting",
    "commuting": "morning sunlight",
    "hobby": "creative studio lighting",
    "self_care": "bathroom lighting, mirror reflection",
    "other": "natural lighting",
}


# ==================== LLM 场景生成（自动自拍专用） ====================

_SCENE_LLM_PROMPT = """You are a selfie scene tag generator for anime image generation (Stable Diffusion).
Given a character's current activity description, output a JSON object with 4 keys:
- action: physical pose/gesture/hand position (3-8 English tags)
- environment: background and surroundings (3-8 English tags)
- expression: facial expression (2-5 English tags)
- lighting: light conditions (2-4 English tags)

Rules:
1. Output ONLY valid JSON, no markdown, no explanations
2. All values must be English tags suitable for Stable Diffusion
3. Do NOT include character appearance (hair, eyes, clothing)
4. Tags should feel natural for a selfie scenario
5. Keep tags concise and descriptive

Examples:

Activity: 在书房看轻小说
{"action": "holding book, reading, relaxed pose", "environment": "study room, bookshelf, warm interior", "expression": "content smile, absorbed", "lighting": "desk lamp, warm indoor light"}

Activity: 在厨房做早饭
{"action": "holding spatula, cooking", "environment": "kitchen, stove, morning atmosphere", "expression": "happy smile, focused on cooking", "lighting": "morning light through window, bright kitchen"}

Activity: 在公园散步
{"action": "walking, casual stroll", "environment": "park, trees, pathway, flowers", "expression": "peaceful smile, relaxed", "lighting": "soft natural sunlight, dappled light"}

Now generate for the following activity:"""


async def generate_scene_with_llm(activity_info: ActivityInfo) -> Optional[Dict[str, str]]:
    """使用 LLM 根据活动描述生成英文 SD 场景标签

    Args:
        activity_info: 活动信息

    Returns:
        包含 action, environment, expression, lighting 的字典，失败返回 None
    """
    try:
        from src.plugin_system.apis import llm_api

        models = llm_api.get_available_models()
        model = models.get("replyer")
        if not model:
            logger.warning("未找到 replyer 模型，LLM 场景生成失败")
            return None

        prompt = f"{_SCENE_LLM_PROMPT}\n\nActivity: {activity_info.description}"

        success, response, _, model_name = await llm_api.generate_with_model(
            prompt=prompt,
            model_config=model,
            request_type="plugin.auto_selfie_scene",
            temperature=0.7,
            max_tokens=8192,
        )

        if not success or not response:
            logger.warning("LLM 场景生成返回空响应")
            return None

        # 清理响应（移除可能的 markdown 代码块）
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        scene = json.loads(cleaned)

        # 验证必要字段
        required_keys = {"action", "environment", "expression", "lighting"}
        if not required_keys.issubset(scene.keys()):
            missing = required_keys - set(scene.keys())
            logger.warning(f"LLM 场景缺少字段: {missing}")
            return None

        # 确保所有值都是字符串
        for key in required_keys:
            if not isinstance(scene[key], str) or not scene[key].strip():
                logger.warning(f"LLM 场景字段 {key} 无效: {scene.get(key)}")
                return None

        logger.info(f"LLM 场景生成成功 (模型: {model_name}): action={scene['action'][:50]}")
        return {
            "hand_action": scene["action"],
            "environment": scene["environment"],
            "expression": scene["expression"],
            "lighting": scene["lighting"],
        }

    except json.JSONDecodeError as e:
        logger.warning(f"LLM 场景 JSON 解析失败: {e}")
        return None
    except Exception as e:
        logger.error(f"LLM 场景生成异常: {e}")
        return None


# ==================== 公共函数 ====================

def get_action_for_activity(activity_info: ActivityInfo) -> Dict[str, str]:
    """
    根据活动类型获取确定性场景数据（手动自拍使用）

    Args:
        activity_info: 活动信息

    Returns:
        包含 hand_action, environment, expression, lighting 的字典
    """
    activity_key = activity_info.activity_type.value

    return {
        "hand_action": ACTIVITY_ACTIONS.get(activity_key, ACTIVITY_ACTIONS["other"]),
        "environment": ACTIVITY_ENVIRONMENTS.get(activity_key, ACTIVITY_ENVIRONMENTS["other"]),
        "expression": ACTIVITY_EXPRESSIONS.get(activity_key, ACTIVITY_EXPRESSIONS["other"]),
        "lighting": ACTIVITY_LIGHTING.get(activity_key, ACTIVITY_LIGHTING["other"]),
    }


async def convert_to_selfie_prompt(
    activity_info: ActivityInfo,
    selfie_style: str = "standard",
    bot_appearance: str = "",
) -> Optional[str]:
    """
    将活动信息转换为完整的自拍 SD 提示词（自动自拍专用）

    使用 LLM 根据活动描述生成场景标签，LLM 失败时返回 None。

    Args:
        activity_info: 活动信息
        selfie_style: 自拍风格 ("standard" 或 "mirror")
        bot_appearance: Bot 外观描述（从配置读取的 selfie.prompt_prefix）

    Returns:
        完整的 SD 提示词，LLM 失败时返回 None
    """
    # 使用 LLM 生成场景
    scene = await generate_scene_with_llm(activity_info)
    if not scene:
        logger.warning("LLM 场景生成失败，取消本次自拍提示词生成")
        return None

    prompt_parts: List[str] = []

    # 1. 强制主体
    prompt_parts.append("(1girl:1.4), (solo:1.3)")

    # 2. Bot 外观
    if bot_appearance:
        prompt_parts.append(bot_appearance)

    # 3. 表情
    prompt_parts.append(f"({scene['expression']}:1.2)")

    # 4. 手部/身体动作
    hand_action = scene["hand_action"]

    # standard 自拍禁止手机类词汇
    if selfie_style == "standard" and hand_action:
        if re.search(r"\b(phone|smartphone|mobile|device)\b", hand_action, flags=re.IGNORECASE):
            hand_action = "resting head on hand"

    if hand_action:
        if selfie_style == "standard":
            hand_prompt = (
                f"(visible free hand {hand_action}:1.4), "
                "(only one hand visible in frame:1.5), "
                "(single hand gesture:1.3)"
            )
        else:
            hand_prompt = f"({hand_action}:1.3)"
        prompt_parts.append(hand_prompt)

    # 5. 环境
    prompt_parts.append(scene["environment"])

    # 6. 光线
    prompt_parts.append(scene["lighting"])

    # 7. 自拍风格
    if selfie_style == "mirror":
        selfie_scene = (
            "mirror selfie, reflection in mirror, "
            "holding phone in hand, phone visible, "
            "looking at mirror, indoor scene"
        )
    else:
        selfie_scene = (
            "selfie, front camera view, POV selfie, "
            "(front facing selfie camera angle:1.3), "
            "looking at camera, slight high angle selfie, "
            "upper body shot, cowboy shot, "
            "(centered composition:1.2)"
        )
    prompt_parts.append(selfie_scene)

    # 8. 过滤空值、去重、拼接
    prompt_parts = [p for p in prompt_parts if p and p.strip()]
    keywords = [kw.strip() for kw in ", ".join(prompt_parts).split(",")]
    seen = set()
    unique = []
    for kw in keywords:
        kw_lower = kw.strip().lower()
        if kw_lower and kw_lower not in seen:
            seen.add(kw_lower)
            unique.append(kw.strip())

    final_prompt = ", ".join(unique)
    logger.info(f"生成自拍提示词: {final_prompt[:150]}...")
    return final_prompt


def get_negative_prompt_for_style(selfie_style: str, base_negative: str = "") -> str:
    """
    获取指定自拍风格的负面提示词

    Args:
        selfie_style: 自拍风格
        base_negative: 基础负面提示词（从配置读取）

    Returns:
        完整的负面提示词
    """
    if selfie_style == "standard":
        anti_dual_hands = ANTI_DUAL_HANDS_PROMPT
        if base_negative:
            return f"{base_negative}, {anti_dual_hands}"
        return anti_dual_hands

    return base_negative
