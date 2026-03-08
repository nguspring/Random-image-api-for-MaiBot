# pyright: reportIncompatibleVariableOverride=false
# pyright: reportIncompatibleMethodOverride=false
# pyright: reportMissingImports=false
# pyright: reportMissingTypeArgument=false

"""
Random-image-api-for-MaiBot 插件

把 Random-image-api (https://github.com/inliver233/Random-image-api) 接入麦麦。
支持标签筛选、R18 开关、AI 过滤、方向筛选、画师筛选、多图获取等功能。
"""

from __future__ import annotations

import asyncio
import contextlib
from io import BytesIO
import base64
import json
import re
import time

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union
from urllib import parse, request
from urllib.error import HTTPError, URLError

from src.common.logger import get_logger
from src.plugin_system import (
    ActionInfo,
    BaseAction,
    BaseCommand,
    BaseEventHandler,
    BasePlugin,
    BaseTool,
    CommandInfo,
    ConfigField,
    EventHandlerInfo,
    ReplyContentType,
    ToolInfo,
    register_plugin,
)

if TYPE_CHECKING:
    from src.chat.message_receive.message import MessageRecv

logger = get_logger("random_image_api")


# ============================================================
# 工具：跳过 VLM 识图的上下文管理器
# ============================================================
# 发送图片时，框架会在 MessageSending.process() 中调用
# MessageProcessBase._process_single_segment()，对 image 类型
# 的 segment 调用 VLM 模型识图生成文字描述。这会导致发送非常慢，
# 尤其是 VLM 超时时更是灾难。
#
# 这个上下文管理器在进入时 monkey-patch _process_single_segment，
# 让 image 类型直接返回 "[图片]" 占位符，退出时恢复原方法。
# 只在本插件发送图片时生效，不影响其他插件。
# ============================================================
@contextlib.asynccontextmanager
async def skip_vlm_for_images():
    """临时跳过发送图片时的 VLM 识图步骤。"""
    from src.chat.message_receive.message import MessageSending

    original_method = MessageSending._process_single_segment

    async def _patched_process_single_segment(self, segment):
        """对 image 类型直接返回占位符，其余走原逻辑。"""
        if segment.type == "image":
            return "[图片]"
        return await original_method(self, segment)

    MessageSending._process_single_segment = _patched_process_single_segment
    try:
        yield
    finally:
        MessageSending._process_single_segment = original_method


# ============================================================
# 工具：classproperty 描述符
# ============================================================
class classproperty(property):
    """支持类访问与实例访问的属性描述符。"""

    def __get__(self, obj: Any, owner: Optional[type] = None) -> Any:
        owner_type: type = owner or type(obj)
        fget_func = self.fget
        if fget_func is None:
            raise AttributeError("unreadable classproperty")
        return fget_func(owner_type)


# ============================================================
# API 客户端
# ============================================================
class RandomImageAPI:
    """Random-image-api 的 HTTP 客户端（标准库实现）。"""

    # ---- 图片压缩默认参数（可通过 set_compress_options 覆盖）----
    _compress_enabled: bool = False
    _max_image_width: int = 1280
    _jpeg_quality: int = 85

    def __init__(self, base_url: str, timeout: int = 30):
        """
        初始化 API 客户端。

        参数：
            base_url: API 服务地址（例如 https://i.mukyu.ru）
            timeout: 请求超时时间（秒）
        """
        self.base_url: str = base_url.rstrip("/")
        self.timeout: int = max(5, min(int(timeout), 120))
        self.user_agent: str = "MaiBot-RandomImageAPI-Plugin/2.0"

    def set_compress_options(self, enabled: bool, max_width: int, quality: int) -> None:
        """
        设置图片压缩参数。

        参数：
            enabled: 是否启用压缩
            max_width: 最大宽度（像素），超过会等比缩放
            quality: JPEG 压缩质量（1-100）
        """
        self._compress_enabled = enabled
        self._max_image_width = max(100, min(int(max_width), 4096))
        self._jpeg_quality = max(1, min(int(quality), 100))

    async def get_random_image(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """异步获取一张随机图片，失败返回 None。"""
        return await asyncio.to_thread(self._request_sync, dict(params))

    async def get_multiple_images(self, params: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
        """
        并发获取多张图片。

        因为 Random-image-api 没有原生 num 参数，
        所以通过并发多次请求来实现。
        """
        count = max(1, min(count, 10))
        tasks: List[asyncio.Task[Optional[Dict[str, Any]]]] = [
            asyncio.create_task(self.get_random_image(params)) for _ in range(count)
        ]
        raw_results: List[Optional[Dict[str, Any]]] = await asyncio.gather(*tasks)
        seen_ids: set[str] = set()
        results: List[Dict[str, Any]] = []
        for r in raw_results:
            if r is None:
                continue
            illust_id: str = str(r.get("illust_id", ""))
            if illust_id and illust_id in seen_ids:
                continue
            seen_ids.add(illust_id)
            results.append(r)
        return results

    def _request_sync(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """\u540c\u6b65\u6267\u884c HTTP \u8bf7\u6c42\uff08\u7531\u7ebf\u7a0b\u6c60\u8c03\u7528\uff09\u3002"""
        # 固定加上最快的镜像节点参数（re = 大陆访问优先）
        params.setdefault("pixiv_cat", "1")
        params.setdefault("pximg_mirror_host", "re")
        query: str = parse.urlencode(params, doseq=True)
        """同步执行 HTTP 请求（由线程池调用）。"""
        query: str = parse.urlencode(params, doseq=True)
        url: str = f"{self.base_url}/random?{query}"

        req: request.Request = request.Request(
            url=url,
            method="GET",
            headers={
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                payload: bytes = resp.read()
        except HTTPError as exc:
            logger.warning(f"Random-image-api HTTP 错误: {exc.code}")
            return None
        except URLError as exc:
            logger.warning(f"Random-image-api 网络错误: {exc}")
            return None
        except TimeoutError:
            logger.warning("Random-image-api 请求超时")
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Random-image-api 请求异常: {exc}")
            return None

        try:
            data: Any = json.loads(payload.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Random-image-api JSON 解析失败: {exc}")
            return None

        if not isinstance(data, dict):
            return None
        if not bool(data.get("ok", False)):
            logger.info(f"Random-image-api 返回非成功: {data.get('code', 'UNKNOWN')}")
            return None

        return self._parse_response(data)

    def _parse_response(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """解析 API 响应并拍平为统一结构。"""
        try:
            payload: Dict[str, Any] = data.get("data", {})
            image_info: Dict[str, Any] = payload.get("image", {})
            user_info: Dict[str, Any] = image_info.get("user", {})
            urls_info: Dict[str, Any] = payload.get("urls", {})

            proxy_path: str = str(urls_info.get("proxy", "") or "")
            if proxy_path.startswith("http://") or proxy_path.startswith("https://"):
                image_url: str = proxy_path
            elif proxy_path:
                image_url = f"{self.base_url}{proxy_path}"
            else:
                image_url = str(urls_info.get("legacy_single", "") or "")

            tags_raw: Any = payload.get("tags", [])
            tags: List[str] = [str(tag) for tag in tags_raw] if isinstance(tags_raw, list) else []

            return {
                "illust_id": str(image_info.get("illust_id", "")),
                "title": str(image_info.get("title", "未知")),
                "user_name": str(user_info.get("name", "未知")),
                "user_id": str(user_info.get("id", "")),
                "width": int(image_info.get("width", 0) or 0),
                "height": int(image_info.get("height", 0) or 0),
                "bookmarks": int(image_info.get("bookmark_count", 0) or 0),
                "views": int(image_info.get("view_count", 0) or 0),
                "ai_type": int(image_info.get("ai_type", 0) or 0),
                "tags": tags,
                "image_url": image_url,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"解析 Random-image-api 响应失败: {exc}")
            return None

    async def download_image_base64(self, url: str) -> Optional[str]:
        """
        下载图片并转为 base64 字符串。

        先把图片下载到本地，再发送，避免 QQ 平台拉取图片超时导致“图片已过期”。
        """
        return await asyncio.to_thread(self._download_sync, url)

    def _download_sync(self, url: str) -> Optional[str]:
        """同步下载图片并转 base64（由线程池调用）。如果启用了压缩，会先压缩再转 base64。"""
        req: request.Request = request.Request(
            url=url,
            method="GET",
            headers={"User-Agent": self.user_agent},
        )
        try:
            with request.urlopen(req, timeout=60) as resp:
                image_bytes: bytes = resp.read()
            # 如果启用了压缩，先压缩再转 base64
            if self._compress_enabled:
                image_bytes = self._compress_image(image_bytes)
            return base64.b64encode(image_bytes).decode("ascii")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"下载图片失败 ({url}): {exc}")
            return None

    def _compress_image(self, image_bytes: bytes) -> bytes:
        """
        压缩图片：限制最大宽度 + JPEG 压缩。

        如果图片宽度超过 max_image_width，会等比缩放。
        然后转为 JPEG 格式并按 jpeg_quality 压缩。
        如果压缩过程出错（比如图片格式不支持），返回原始数据。
        """
        try:
            from PIL import Image

            img: Image.Image = Image.open(BytesIO(image_bytes))

            # 等比缩放：宽度超过上限时缩小
            if img.width > self._max_image_width:
                ratio: float = self._max_image_width / img.width
                new_height: int = int(img.height * ratio)
                img = img.resize((self._max_image_width, new_height), Image.Resampling.LANCZOS)

            # 转为 RGB（JPEG 不支持 RGBA 透明通道）
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")

            # 压缩为 JPEG
            buffer: BytesIO = BytesIO()
            img.save(buffer, format="JPEG", quality=self._jpeg_quality, optimize=True)
            compressed: bytes = buffer.getvalue()

            original_kb: float = len(image_bytes) / 1024
            compressed_kb: float = len(compressed) / 1024
            logger.debug(
                f"图片压缩: {original_kb:.0f}KB -> {compressed_kb:.0f}KB ({compressed_kb / original_kb * 100:.0f}%)"
            )
            return compressed
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"图片压缩失败，使用原图: {exc}")
            return image_bytes


# ============================================================
# 命令组件
# ============================================================
class RandomImageCommand(BaseCommand):
    """随机图片命令。"""

    command_name: str = "随机图片"
    command_description: str = "从 Random-image-api 获取随机 Pixiv 图片"
    command_pattern: str = r"^(?P<trigger>\u968f\u673a\u56fe\u7247|\u6765\u5f20\u56fe|\u6765\u70b9\u56fe|\u6765\u70b9\u56fe\u7247|\u6765\u5f20\u56fe\u7247|\u6765\u5f20\u6da9\u56fe|\u6765\u70b9\u6da9\u56fe|\u968f\u673a\u6da9\u56fe|\u6da9\u56fe|setu|\u6765\u5f20setu|\u6765\u70b9setu)\s*(?P<args>.*)?$"
    command_help: str = (
        "随机图片 [数量] [参数]\n"
        "参数说明：\n"
        "  数字(1-10)   获取多张图片\n"
        "  #标签        按标签筛选（可多个）\n"
        "  -#标签       排除标签（可多个）\n"
        "  r18          获取 R18（需管理员 allow_r18=true）\n"
        "  noai / ai    排除 AI / 只看 AI\n"
        "  横屏 竖屏 方形  图片方向\n"
        "  uid:数字     指定画师\n"
        "  收藏>数字    最低收藏数\n"
        "  浏览>数字    最低浏览数\n"
        "  quality/random  获取策略\n"
        "示例：随机图片 3 #原神 noai 横屏"
    )
    command_examples: List[str] = [
        "随机图片",
        "来张图 3",
        "随机图片 #原神 noai",
        "来张涩图 5 #碧蓝航线 横屏",
        "随机图片 uid:12345678 收藏>1000",
    ]
    enable_command: bool = True

    def __init__(self, message: MessageRecv, plugin_config: Optional[Dict[str, Any]] = None):
        super().__init__(message, plugin_config)
        self._cooldown_cache: Dict[str, float] = {}

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        """命令执行入口。"""
        args_str: str = str(self.matched_groups.get("args", "") or "").strip()

        # help 命令
        if args_str.lower() in ("help", "帮助", "?", "？"):
            await self.send_text(self.command_help)
            return (True, "显示帮助", 2)

        # 读取配置
        base_url: str = str(self.get_config("api.base_url", "https://i.mukyu.ru"))
        timeout_seconds: int = self._safe_int(self.get_config("api.timeout", 30), 30)
        cooldown: int = self._safe_int(self.get_config("features.cooldown_seconds", 10), 10)
        allow_r18: bool = self._safe_bool(self.get_config("features.allow_r18", False))
        default_strategy: str = str(self.get_config("features.default_strategy", "random"))
        default_num: int = self._safe_int(self.get_config("features.default_num", 1), 1)
        max_num: int = self._safe_int(self.get_config("features.max_num", 10), 10)
        use_forward: bool = self._safe_bool(self.get_config("features.use_forward_message", True))
        exclude_ai: bool = self._safe_bool(self.get_config("features.exclude_ai", True))
        quality_samples: int = self._safe_int(self.get_config("features.quality_samples", 10), 10)

        # 读取图片筛选配置：最小宽/高/像素
        # 读取图片筛选配置：最小宽/高/像素
        min_width: int = self._safe_int(self.get_config("filter.min_width", 0), 0)
        min_height: int = self._safe_int(self.get_config("filter.min_height", 0), 0)
        min_pixels: int = self._safe_int(self.get_config("filter.min_pixels", 0), 0)
        # 冷却检查
        user_id: str = self._safe_user_id()
        now: float = time.time()
        last_use: float = self._cooldown_cache.get(user_id, 0.0)
        if cooldown > 0 and (now - last_use) < cooldown:
            remaining: int = int(cooldown - (now - last_use)) + 1
            await self.send_text(f"冷却中，请 {remaining} 秒后再试~")
            return (True, "冷却中", 2)

        # 解析参数
        params, request_count = self._parse_args(
            args_str,
            allow_r18=allow_r18,
            default_strategy=default_strategy,
            default_num=default_num,
            max_num=max_num,
            exclude_ai=exclude_ai,
            trigger_word=self.matched_groups.get("trigger", ""),
            quality_samples=quality_samples,
        )
        # 将筛选配置注入请求参数
        if min_width > 0:
            params["min_width"] = str(min_width)
        if min_height > 0:
            params["min_height"] = str(min_height)
        if min_pixels > 0:
            params["min_pixels"] = str(min_pixels)
        # 创建 API 客户端
        api_client: RandomImageAPI = RandomImageAPI(base_url=base_url, timeout=timeout_seconds)

        # 读取图片压缩配置
        compress_enabled: bool = self._safe_bool(self.get_config("image_compress.enabled", True))
        max_image_width: int = self._safe_int(self.get_config("image_compress.max_image_width", 1280), 1280)
        jpeg_quality: int = self._safe_int(self.get_config("image_compress.jpeg_quality", 85), 85)
        api_client.set_compress_options(compress_enabled, max_image_width, jpeg_quality)

        await self.send_text(f"正在获取{request_count}张随机图片，请稍候...")

        # 获取图片
        # 获取图片
        if request_count == 1:
            result: Optional[Dict[str, Any]] = await api_client.get_random_image(params)
            results: List[Dict[str, Any]] = [result] if result else []
        else:
            results = await api_client.get_multiple_images(params, request_count)

        if not results:
            await self.send_text("没有找到符合条件的图片，换个条件试试？")
            return (True, "未找到图片", 2)

        self._cooldown_cache[user_id] = time.time()

        # 发送结果
        if use_forward:
            await self._send_as_forward(results, api_client)
        else:
            await self._send_as_separate(results, api_client)

        return (True, f"发送{len(results)}张随机图片成功", 2)

    async def _send_as_forward(self, results: List[Dict[str, Any]], api_client: RandomImageAPI) -> None:
        """
        以合并转发消息形式发送。
        先并发下载所有图片转 base64，再统一发送，避免“图片已过期”。
        """
        from src.config.config import global_config

        bot_qq: str = str(global_config.bot.qq_account)
        bot_name: str = str(global_config.bot.nickname)

        # 第一步：并发下载所有图片
        image_urls: List[str] = [str(r.get("image_url", "") or "") for r in results]
        download_tasks: List[asyncio.Task[Optional[str]]] = []
        for url in image_urls:
            if url:
                download_tasks.append(asyncio.create_task(api_client.download_image_base64(url)))
            else:
                # 没有 URL 的情况，放一个返回 None 的协程
                async def _noop() -> Optional[str]:
                    return None

                download_tasks.append(asyncio.create_task(_noop()))
        # 等待所有图片下载完成
        base64_results: List[Optional[str]] = await asyncio.gather(*download_tasks)

        # 第二步：组装转发消息
        forward_messages: List[Any] = []
        for i, result in enumerate(results):
            info_text: str = self._format_result(result)
            content: List[Any] = [(ReplyContentType.TEXT, info_text)]
            img_b64: Optional[str] = base64_results[i] if i < len(base64_results) else None
            if img_b64:
                content.append((ReplyContentType.IMAGE, img_b64))
            elif image_urls[i]:
                # base64 下载失败，回退到 URL 方式
                content.append(("imageurl", image_urls[i]))
            forward_messages.append((bot_qq, bot_name, content))

        async with skip_vlm_for_images():
            await self.send_forward(forward_messages, storage_message=True)

    async def _send_as_separate(self, results: List[Dict[str, Any]], api_client: RandomImageAPI) -> None:
        """逐条发送（不使用合并转发时的备选方案）。"""
        async with skip_vlm_for_images():
            for result in results:
                info_text: str = self._format_result(result)
                await self.send_text(info_text)
                image_url: str = str(result.get("image_url", "") or "")
                if image_url:
                    try:
                        img_b64: Optional[str] = await api_client.download_image_base64(image_url)
                        if img_b64:
                            await self.send_image(img_b64)
                        else:
                            await self.send_custom("imageurl", image_url)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(f"发送图片失败: {exc}")

    def _parse_args(
        self,
        args_str: str,
        allow_r18: bool,
        default_strategy: str,
        default_num: int,
        max_num: int,
        exclude_ai: bool = True,
        trigger_word: str = "",
        quality_samples: int = 10,
    ) -> Tuple[Dict[str, Any], int]:
        """
        解析用户输入的参数。

        返回：(API 请求参数字典, 请求数量)
        """
        params: Dict[str, Any] = {"format": "json"}
        # 如果配置了默认排除 AI，先设置上，用户输入 "ai" 可以覆盖
        if exclude_ai:
            params["ai_type"] = "0"
        # 触发词属于涩图系列时，自动请求 R18（需管理员开启 allow_r18）
        _setu_triggers: frozenset[str] = frozenset(
            {
                "\u6da9\u56fe",
                "\u6765\u5f20\u6da9\u56fe",
                "\u6765\u70b9\u6da9\u56fe",
                "\u968f\u673a\u6da9\u56fe",
                "setu",
                "\u6765\u5f20setu",
                "\u6765\u70b9setu",
            }
        )
        if allow_r18 and trigger_word in _setu_triggers:
            params["r18"] = "1"
            params["ai_type"] = "0"
        request_count: int = default_num
        tokens: List[str] = args_str.split()

        included_tags: List[str] = []
        excluded_tags: List[str] = []

        for token in tokens:
            lower: str = token.lower()

            # 数量（纯数字）
            if re.match(r"^\d+$", token):
                num: int = int(token)
                if 1 <= num <= max_num:
                    request_count = num
                continue

            # 排除标签 -#xxx
            if token.startswith("-#") and len(token) > 2:
                excluded_tags.append(token[2:])
                continue

            # 包含标签 #xxx
            if token.startswith("#") and len(token) > 1:
                included_tags.append(token[1:])
                continue

            # R18
            if lower == "r18":
                if allow_r18:
                    params["r18"] = "1"
                continue

            # AI 过滤
            if lower == "noai":
                params["ai_type"] = "0"
                continue
            if lower == "ai":
                params["ai_type"] = "1"
                continue

            # 方向
            if token in ("横屏", "横图"):
                params["orientation"] = "landscape"
                continue
            if token in ("竖屏", "竖图"):
                params["orientation"] = "portrait"
                continue
            if token in ("方形", "方图"):
                params["orientation"] = "square"
                continue

            # 画师 uid:xxx
            uid_match = re.match(r"^uid[：:](\d+)$", token)
            if uid_match:
                params["user_id"] = uid_match.group(1)
                continue

            # 收藏阈值 收藏>xxx
            bookmark_match = re.match(r"^收藏[>＞](\d+)$", token)
            if bookmark_match:
                params["min_bookmarks"] = bookmark_match.group(1)
                continue

            # 浏览阈值 浏览>xxx
            view_match = re.match(r"^浏览[>＞](\d+)$", token)
            if view_match:
                params["min_views"] = view_match.group(1)
                continue

            # 宽度阈值 宽度>xxx
            width_match = re.match(r"^宽度[>＞](\d+)$", token)
            if width_match:
                params["min_width"] = width_match.group(1)
                continue

            # 高度阈值 高度>xxx
            height_match = re.match(r"^高度[>＞](\d+)$", token)
            if height_match:
                params["min_height"] = height_match.group(1)
                continue

            # 像素阈值 像素>xxx
            pixels_match = re.match(r"^像素[>＞](\d+)$", token)
            if pixels_match:
                params["min_pixels"] = pixels_match.group(1)
                continue
            # 策略
            if lower in ("quality", "random"):
                params["strategy"] = lower
                continue

            # 作品类型
            if lower in ("illust", "manga", "ugoira", "插画", "漫画", "动图"):
                type_map: Dict[str, str] = {
                    "illust": "illust",
                    "插画": "illust",
                    "manga": "manga",
                    "漫画": "manga",
                    "ugoira": "ugoira",
                    "动图": "ugoira",
                }
                params["illust_type"] = type_map.get(lower, "any")
                continue

        # 设置默认策略
        if "strategy" not in params:
            params["strategy"] = default_strategy
        # quality 策略时附加采样数量（限制服务端采样范围，减少响应时间）
        if params.get("strategy") == "quality" and quality_samples > 0:
            params["quality_samples"] = str(quality_samples)

        # 标签参数
        if included_tags:
            params["included_tags"] = included_tags
        if excluded_tags:
            params["excluded_tags"] = excluded_tags

        return params, request_count

    def _format_result(self, result: Dict[str, Any]) -> str:
        """格式化图片信息为展示文本。"""
        title: str = str(result.get("title", "未知"))
        user_name: str = str(result.get("user_name", "未知"))
        width: int = int(result.get("width", 0) or 0)
        height: int = int(result.get("height", 0) or 0)
        bookmarks: int = int(result.get("bookmarks", 0) or 0)
        views: int = int(result.get("views", 0) or 0)
        ai_type: int = int(result.get("ai_type", 0) or 0)
        tags: List[str] = result.get("tags", [])
        illust_id: str = str(result.get("illust_id", ""))

        ai_label: str = "AI 绘图" if ai_type == 1 else "人工绘图"
        tags_str: str = "、".join(tags[:8]) if tags else "无"

        lines: List[str] = [
            f"🎨 {title}",
            f"👤 画师：{user_name}",
            f"📐 尺寸：{width}x{height}",
            f"❤️ 收藏：{bookmarks} | 👁️ 浏览：{views}",
            f"🤖 类型：{ai_label}",
            f"🏷️ 标签：{tags_str}",
            f"🔗 作品：pixiv.net/artworks/{illust_id}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        """安全转换为 int。"""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_bool(value: Any) -> bool:
        """安全转换为 bool。"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)

    def _safe_user_id(self) -> str:
        """安全获取用户 ID。"""
        try:
            if (
                hasattr(self.message, "message_info")
                and hasattr(self.message.message_info, "user_info")
                and self.message.message_info.user_info is not None
                and hasattr(self.message.message_info.user_info, "user_id")
            ):
                return str(self.message.message_info.user_info.user_id)
        except Exception:  # noqa: BLE001
            pass
        return "unknown"


# ============================================================
# 插件主类
# ============================================================
@register_plugin
class RandomImagePlugin(BasePlugin):
    """Random-image-api-for-MaiBot 插件。"""

    @property
    def plugin_name(self) -> str:
        """插件唯一标识。"""
        return "random_image_api"

    @property
    def enable_plugin(self) -> bool:
        """插件默认启用状态。"""
        return True

    @property
    def dependencies(self) -> list[str]:
        """插件依赖的其他 MaiBot 插件。"""
        return []

    @property
    def python_dependencies(self) -> list[str]:
        """插件所需的 Python 依赖。"""
        return ["Pillow"]

    @property
    def config_file_name(self) -> str:
        """插件配置文件名。"""
        return "config.toml"

    @classproperty
    def config_schema(cls) -> Dict[str, Any]:
        """
        插件配置结构（系统将自动生成 config.toml）。

        配置分为三个部分：
        - plugin: 插件总开关
        - api: API 连接相关配置
        - features: 功能行为配置
        """
        return {
            # ========== 插件基础配置 ==========
            "plugin": {
                # 是否启用这个插件
                # - true: 插件正常运行
                # - false: 插件完全禁用，所有功能都不生效
                "enabled": ConfigField(
                    type=bool,
                    default=False,
                    description="是否启用插件",
                ),
                # 配置文件版本号（用于未来配置迁移）
                "config_version": ConfigField(
                    type=str,
                    default="1.0.0",
                    description="配置版本号（请勿手动修改）",
                ),
            },
            # ========== 组件开关 ==========
            "components": {
                # 是否启用随机图片命令
                # - true: 用户可以使用 "随机图片" 等触发词
                # - false: 命令不会被注册，用户无法触发
                "enable_command": ConfigField(
                    type=bool,
                    default=True,
                    description="是否启用随机图片命令",
                ),
            },
            # ========== API 连接配置 ==========
            "api": {
                # Random-image-api 服务地址
                # - 默认使用公开服务 https://i.mukyu.ru
                # - 如果你自建了服务，改成你自己的地址
                # - 末尾不要加 /
                "base_url": ConfigField(
                    type=str,
                    default="https://i.mukyu.ru",
                    description="API 服务地址（末尾不加 /）",
                ),
                # 请求超时时间（秒）
                # - 太短容易超时，太长会让用户等太久
                # - 建议 15-60 秒
                "timeout": ConfigField(
                    type=int,
                    default=30,
                    description="请求超时时间（秒），建议 15-60",
                ),
            },
            # ========== 图片尺寸筛选 ==========
            # ========== 图片尺寸筛选 ==========
            "filter": {
                # 最小宽度（像素）
                # - 0 表示不限制，可配合压缩功能使用
                "min_width": ConfigField(
                    type=int,
                    default=0,
                    description="最小宽度（像素），0 表示不限制",
                ),
                # 最小高度（像素）
                # - 0 表示不限制
                "min_height": ConfigField(
                    type=int,
                    default=0,
                    description="最小高度（像素），0 表示不限制",
                ),
                # 最小总像素数
                # - 0 表示不限制
                # - 例：1920x1080 = 2073600，与 min_width/min_height 同时生效
                "min_pixels": ConfigField(
                    type=int,
                    default=0,
                    description="最小总像素数，0 表示不限制",
                ),
            },
            # ========== 功能配置 ==========
            "features": {
                # 默认获取图片数量
                # - 用户不指定数量时，默认获取几张
                # - 范围：1-10
                "default_num": ConfigField(
                    type=int,
                    default=1,
                    description="默认获取图片数量（1-10）",
                ),
                # 单次最大获取数量
                # - 用户最多一次能要几张图
                # - 范围：1-10，设太大会给 API 服务器压力
                "max_num": ConfigField(
                    type=int,
                    default=10,
                    description="单次最大获取数量（1-10）",
                ),
                # 命令冷却时间（秒）
                # - 防止用户刷屏
                # - 设为 0 表示不限制
                "cooldown_seconds": ConfigField(
                    type=int,
                    default=10,
                    description="命令冷却时间（秒），0=不限制",
                ),
                # R18 管理员总开关
                # - false: 即使用户输入 r18 也不会生效（安全）
                # - true: 用户输入 r18 时才会请求 R18 内容
                # ⚠️ 请确保符合当地法律法规再开启
                "allow_r18": ConfigField(
                    type=bool,
                    default=False,
                    description="是否允许 R18 内容（请谨慎开启）",
                ),
                # 是否默认排除 AI 绘图作品
                # - true: 默认只返回人类画师的作品（推荐）
                # - false: 不过滤，AI 和人类作品都会返回
                # - 用户输入 "ai" 可以临时覆盖，只看 AI 作品
                # - 用户输入 "noai" 可以临时强制排除 AI 作品
                "exclude_ai": ConfigField(
                    type=bool,
                    default=True,
                    description="是否默认排除 AI 绘图作品（推荐开启）",
                ),
                # \u9ed8\u8ba4\u83b7\u53d6\u7b56\u7565
                # - "quality": \u8d28\u91cf\u4f18\u5148\uff08\u4f1a\u91c7\u6837\u591a\u5f20\u56fe\u7b5b\u9009\u9ad8\u6536\u85cf\uff0c\u54cd\u5e94\u8f83\u6162\uff09
                # - "random": \u7eaf\u968f\u673a\uff08\u4e0d\u91c7\u6837\uff0c\u54cd\u5e94\u6700\u5feb\uff09
                "default_strategy": ConfigField(
                    type=str,
                    default="random",
                    description="\u9ed8\u8ba4\u7b56\u7565\uff1aquality\uff08\u8d28\u91cf\u4f18\u5148\uff09\u6216 random\uff08\u7eaf\u968f\u673a\uff0c\u66f4\u5feb\uff09",
                ),
                # quality \u7b56\u7565\u7684\u91c7\u6837\u6570\u91cf\uff081-1000\uff09
                # - \u6570\u5b57\u8d8a\u5c0f\u54cd\u5e94\u8d8a\u5feb\uff0c\u4f46\u8d28\u91cf\u8d8a\u4f4e
                # - \u8bbe 10 \u8868\u793a\u4ece 10 \u5f20\u91cc\u6311\u6700\u9ad8\u6536\u85cf\u7684\uff0c\u5efa\u8bae 10-50
                # - \u53ea\u5728 strategy=quality \u65f6\u751f\u6548
                "quality_samples": ConfigField(
                    type=int,
                    default=10,
                    description="quality \u7b56\u7565\u91c7\u6837\u6570\uff081-1000\uff09\uff0c\u6570\u5b57\u8d8a\u5c0f\u54cd\u5e94\u8d8a\u5feb\uff0c\u5efa\u8bae 10-50",
                ),
                # 是否使用合并转发（聊天记录）格式发送
                # - true: 文字和图片打包成聊天记录一起发（推荐）
                # - false: 文字和图片分开发送
                "use_forward_message": ConfigField(
                    type=bool,
                    default=True,
                    description="是否使用合并转发格式发送（推荐开启）",
                ),
            },
            # ========== 图片压缩配置 ==========
            "image_compress": {
                # 是否启用图片压缩
                # - true: 下载图片后自动压缩再发送（推荐开启）
                # - false: 发送原图，大图会很慢
                # 压缩使用 Pillow 库，会自动安装
                "enabled": ConfigField(
                    type=bool,
                    default=True,
                    description="是否启用图片压缩（推荐开启，大幅提升发送速度）",
                ),
                # 图片最大宽度（像素）
                # - 超过这个宽度的图片会等比缩小
                # - 在手机 QQ 上看，1280 已经非常清晰
                # - 数字越小图片越小、发送越快，但清晰度会降低
                # - 建议范围：800-2560
                "max_image_width": ConfigField(
                    type=int,
                    default=1280,
                    description="图片最大宽度（像素），超过会等比缩小，建议 800-2560",
                ),
                # JPEG 压缩质量（1-100）
                # - 数字越大越清晰，但体积越大
                # - 85 是公认的甜点值：肉眼看不出区别，体积压掉 80-90%
                # - 建议范围：70-95
                "jpeg_quality": ConfigField(
                    type=int,
                    default=85,
                    description="JPEG 压缩质量（1-100），85 为推荐值",
                ),
            },
        }

    def get_plugin_components(
        self,
    ) -> List[
        Union[
            Tuple[ActionInfo, Type[BaseAction]],
            Tuple[CommandInfo, Type[BaseCommand]],
            Tuple[EventHandlerInfo, Type[BaseEventHandler]],
            Tuple[ToolInfo, Type[BaseTool]],
        ]
    ]:
        """返回插件组件列表（根据配置决定是否注册命令）。"""
        components: List[
            Union[
                Tuple[ActionInfo, Type[BaseAction]],
                Tuple[CommandInfo, Type[BaseCommand]],
                Tuple[EventHandlerInfo, Type[BaseEventHandler]],
                Tuple[ToolInfo, Type[BaseTool]],
            ]
        ] = []
        enable_command: bool = self.get_config("components.enable_command", True)
        if enable_command:
            components.append((RandomImageCommand.get_command_info(), RandomImageCommand))
        return components
