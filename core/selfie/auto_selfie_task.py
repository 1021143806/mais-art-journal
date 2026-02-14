"""
自动自拍后台任务

定时执行自拍流程：
1. 从 ScheduleProvider 获取当前活动
2. 用 SceneActionGenerator 生成自拍提示词
3. 用 generate_image_standalone 生成图片
4. 用 CaptionGenerator 生成配文
5. 通过 Maizone QZone API 发布到QQ空间说说

支持：
- 可配置间隔（如每 2 小时）
- 安静时段控制
"""

import asyncio
import base64
import os
from typing import Optional

from src.common.logger import get_logger

from .schedule_provider import get_schedule_provider
from .scene_action_generator import convert_to_selfie_prompt, get_negative_prompt_for_style
from .caption_generator import generate_caption
from ..api_clients import generate_image_standalone
from ..utils import get_model_config, is_in_time_range

logger = get_logger("auto_selfie.task")


class AutoSelfieTask:
    """自动自拍后台定时任务"""

    # 连续失败达到此次数后，等待时间翻倍
    _MAX_CONSECUTIVE_FAILURES = 3

    def __init__(self, plugin):
        """
        Args:
            plugin: 插件实例，用于读取配置
        """
        self.plugin = plugin
        self.is_running = False
        self.task: Optional[asyncio.Task] = None
        self._consecutive_failures = 0

    def get_config(self, key: str, default=None):
        return self.plugin.get_config(key, default)

    async def start(self):
        """启动自动自拍任务"""
        if self.is_running:
            return
        self.is_running = True
        self.task = asyncio.create_task(self._selfie_loop())
        logger.info("自动自拍任务已启动")

    async def stop(self):
        """停止自动自拍任务"""
        if not self.is_running:
            return
        self.is_running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("自动自拍任务已停止")

    def _is_quiet_hours(self) -> bool:
        """检查当前是否在安静时段"""
        quiet_start = self.get_config("auto_selfie.quiet_hours_start", "00:00")
        quiet_end = self.get_config("auto_selfie.quiet_hours_end", "07:00")
        return is_in_time_range(quiet_start, quiet_end)

    async def _selfie_loop(self):
        """主循环"""
        # 启动延迟
        await asyncio.sleep(30)

        interval = self.get_config("auto_selfie.interval_minutes", 120)
        interval_seconds = max(interval, 10) * 60  # 至少 10 分钟

        while self.is_running:
            try:
                if self._is_quiet_hours():
                    logger.debug("当前在安静时段，跳过自拍")
                else:
                    await self._execute_selfie()
                    self._consecutive_failures = 0
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._consecutive_failures += 1
                logger.error(
                    f"自拍任务执行出错 (连续第{self._consecutive_failures}次): {e}"
                )

            # 连续失败时指数退避：正常间隔 × 2^(failures // MAX)
            backoff_multiplier = 2 ** (
                self._consecutive_failures // self._MAX_CONSECUTIVE_FAILURES
            )
            sleep_seconds = interval_seconds * backoff_multiplier
            if backoff_multiplier > 1:
                logger.warning(
                    f"连续失败{self._consecutive_failures}次，下次自拍间隔延长至 {sleep_seconds // 60} 分钟"
                )
            await asyncio.sleep(sleep_seconds)

    async def _execute_selfie(self):
        """执行一次完整的自拍流程"""
        logger.info("开始执行自动自拍流程...")

        # 1. 获取当前活动
        provider = get_schedule_provider()

        activity = None
        if provider:
            activity = await provider.get_current_activity()

        if not activity:
            logger.info("未获取到当前活动信息，跳过本次自拍")
            return

        logger.info(f"当前活动: {activity.description} ({activity.activity_type.value})")

        # 2. 生成自拍提示词
        selfie_style = self.get_config("selfie.default_style", "standard")
        bot_appearance = self.get_config("selfie.prompt_prefix", "")
        prompt = await convert_to_selfie_prompt(activity, selfie_style, bot_appearance)
        if not prompt:
            logger.warning("LLM 自拍提示词生成失败，跳过本次自拍")
            return

        negative_prompt = get_negative_prompt_for_style(
            selfie_style,
            self.get_config("selfie.negative_prompt", ""),
        )

        logger.info(f"自拍提示词: {prompt[:100]}...")

        # 3. 生成图片
        selfie_model = self.get_config("auto_selfie.selfie_model", "model1")
        model_config = self._get_model_config(selfie_model)
        if not model_config:
            logger.error(f"模型配置获取失败: {selfie_model}")
            return

        # 透传代理配置
        extra_config = {}
        if self.get_config("proxy.enabled", False):
            extra_config["proxy"] = {
                "enabled": True,
                "url": self.get_config("proxy.url", "http://127.0.0.1:7890"),
                "timeout": self.get_config("proxy.timeout", 60),
            }

        # 检查参考图片（图生图模式）
        reference_image = self._load_reference_image()
        strength = None
        if reference_image:
            if model_config.get("support_img2img", True):
                strength = 0.6
                logger.info("使用参考图片进行图生图自拍")
            else:
                reference_image = None
                logger.warning(f"模型 {selfie_model} 不支持图生图，回退文生图")

        success, image_data = await generate_image_standalone(
            prompt=prompt,
            model_config=model_config,
            size=model_config.get("default_size", "1024x1024"),
            negative_prompt=negative_prompt,
            strength=strength,
            input_image_base64=reference_image,
            max_retries=2,
            extra_config=extra_config if extra_config else None,
        )

        if not success:
            logger.error(f"自拍图片生成失败: {image_data}")
            return

        logger.info(f"自拍图片生成成功，数据长度: {len(image_data)}")

        # 4. 生成配文
        caption = ""
        if self.get_config("auto_selfie.caption_enabled", True):
            caption = await generate_caption(activity)
            if not caption:
                logger.warning("配文生成失败，跳过本次自拍发布")
                return
            logger.info(f"配文: {caption}")

        # 5. 发到QQ空间
        try:
            from plugins.Maizone.qzone import create_qzone_api
            from plugins.Maizone.helpers import get_napcat_config_and_renew
            from src.plugin_system.core import component_registry
            from src.plugin_system.apis import config_api

            # 刷新 Cookie
            maizone_cfg = component_registry.get_plugin_config('MaizonePlugin')
            if maizone_cfg:
                get_config_fn = lambda key, default=None: config_api.get_plugin_config(maizone_cfg, key, default)
                await get_napcat_config_and_renew(get_config_fn)

            # 将 image_data 转为 bytes
            image_bytes = await self._resolve_image_to_bytes(image_data)
            if not image_bytes:
                logger.error("图片数据转换失败，无法发布到QQ空间")
                return

            # 发布说说
            qzone = create_qzone_api()
            if not qzone:
                logger.error("QZone API 创建失败（Cookie 不存在或无效），无法发布自拍")
                return
            tid = await qzone.publish_emotion(caption, [image_bytes])
            logger.info(f"自拍已发布到QQ空间，tid: {tid}")
        except ImportError:
            logger.error("Maizone 插件未安装，无法发布自拍到QQ空间")
        except Exception as e:
            logger.error(f"发布自拍到QQ空间失败: {e}")

    def _get_model_config(self, model_id: str) -> Optional[dict]:
        """获取模型配置"""
        return get_model_config(self.get_config, model_id, log_prefix="[AutoSelfie]")

    def _load_reference_image(self) -> Optional[str]:
        """加载自拍参考图片的base64编码

        Returns:
            图片的base64编码，如果不存在则返回None
        """
        image_path = self.get_config("selfie.reference_image_path", "").strip()
        if not image_path:
            return None

        try:
            # 处理相对路径（相对于插件目录）
            if not os.path.isabs(image_path):
                plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                image_path = os.path.join(plugin_dir, image_path)

            if os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                logger.info(f"[AutoSelfie] 从文件加载自拍参考图片: {image_path}")
                return image_base64
            else:
                logger.warning(f"[AutoSelfie] 自拍参考图片文件不存在: {image_path}")
                return None
        except Exception as e:
            logger.error(f"[AutoSelfie] 加载自拍参考图片失败: {e}")
            return None

    @staticmethod
    async def _resolve_image_to_bytes(image_data: str) -> Optional[bytes]:
        """将 base64 或 URL 格式的图片数据转为 bytes"""
        if image_data.startswith(("http://", "https://")):
            import httpx
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(image_data)
                resp.raise_for_status()
                return resp.content
        else:
            return base64.b64decode(image_data)
