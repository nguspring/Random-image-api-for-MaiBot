"""随机图片插件（Random-image-api-for-MaiBot）— MaiBot 1.3.x / Plugin SDK 2.x。

从 Random-image-api 拉取 Pixiv 随机图片，功能与原 0.12.x 版本保持一致：
- 关键词触发（随机图片 / 来张图 / 涩图 / setu 等）
- 标签包含与排除、AI 过滤、方向筛选、收藏/浏览/尺寸阈值
- R18 总开关 + 白名单/黑名单
- 全局冷却、多地址回退、图片预下载、Pillow 压缩
- 合并转发或逐条发送

与原版的差异只在宿主接口层：发送、配置、机器人身份全部改走
maibot_sdk 的 PluginContext，不再依赖 src.* 内部模块。
"""

import asyncio
import base64
import json
import logging
import re
import time
from io import BytesIO
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, Union
from urllib import parse, request
from urllib.error import HTTPError, URLError

from maibot_sdk import Command, Field, MaiBotPlugin, PluginConfigBase

logger = logging.getLogger("random_image_api")

# 命令触发词正则，与原版保持一致。
# 注意：Command 组件内部使用 re.search，因此这里必须保留首尾锚点。
COMMAND_PATTERN = (
    r"^(?P<trigger>随机图片|来张图|来点图|来点图片|来张图片|来张涩图|来点涩图|随机涩图|涩图|setu|来张setu|来点setu)"
    r"\s*(?P<args>.*)?$"
)

# 这些触发词默认自动请求 R18（仍需 allow_r18 打开）。
_SETU_TRIGGERS: frozenset[str] = frozenset(
    {"涩图", "来张涩图", "来点涩图", "随机涩图", "setu", "来张setu", "来点setu"}
)

COMMAND_HELP = (
    "随机图片 [数量] [参数]\n"
    "参数说明：\n"
    "  数字(1-10)   获取多张图片\n"
    "  #标签        按标签筛选（可多个）\n"
    "  -#标签       排除标签（可多个）\n"
    "  r18          获取 R18（受总开关与群号/QQ 号名单控制）\n"
    "  noai / ai    排除 AI / 只看 AI\n"
    "  横屏 竖屏 方形  图片方向\n"
    "  uid:数字     指定画师\n"
    "  收藏>数字    最低收藏数\n"
    "  浏览>数字    最低浏览数\n"
    "  quality/random  获取策略\n"
    "示例：随机图片 3 #原神 noai 横屏"
)


# ============================================================
# API 客户端（与原版一致，仅日志来源改为模块级 logger）
# ============================================================
class RandomImageAPI:
    """Random-image-api 的 HTTP 客户端（标准库实现）。"""

    # 图片压缩默认参数，可通过 set_compress_options 覆盖。
    _compress_enabled: bool = False
    _max_image_width: int = 1280
    _jpeg_quality: int = 85

    def __init__(self, base_url: str, timeout: int = 30):
        """初始化 API 客户端。

        Args:
            base_url: API 服务地址，例如 https://i.mukyu.ru。
            timeout: 请求超时时间（秒），会被限制在 5~120 之间。
        """
        self.base_url: str = base_url.rstrip("/")
        self.timeout: int = max(5, min(int(timeout), 120))
        self.user_agent: str = "MaiBot-RandomImageAPI-Plugin/2.0"

    def set_compress_options(self, enabled: bool, max_width: int, quality: int) -> None:
        """设置图片压缩参数。

        Args:
            enabled: 是否启用压缩。
            max_width: 最大宽度（像素），超过会等比缩放。
            quality: JPEG 压缩质量（1-100）。
        """
        self._compress_enabled = enabled
        self._max_image_width = max(100, min(int(max_width), 4096))
        self._jpeg_quality = max(1, min(int(quality), 100))

    async def get_random_image(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """异步获取一张随机图片，失败返回 None。"""
        return await asyncio.to_thread(self._request_sync, dict(params))

    async def get_multiple_images(self, params: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
        """并发获取多张图片并按 illust_id 去重。

        Random-image-api 没有原生 num 参数，因此通过并发多次请求实现。
        """
        count = max(1, min(count, 10))
        tasks: List[asyncio.Task[Optional[Dict[str, Any]]]] = [
            asyncio.create_task(self.get_random_image(params)) for _ in range(count)
        ]
        raw_results: List[Optional[Dict[str, Any]]] = await asyncio.gather(*tasks)
        seen_ids: set[str] = set()
        results: List[Dict[str, Any]] = []
        for raw_result in raw_results:
            if raw_result is None:
                continue
            illust_id: str = str(raw_result.get("illust_id", ""))
            if illust_id and illust_id in seen_ids:
                continue
            seen_ids.add(illust_id)
            results.append(raw_result)
        return results

    def _request_sync(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """同步执行 HTTP 请求（由线程池调用）。"""
        # 固定加上最快的镜像节点参数（re = 大陆访问优先）。
        params.setdefault("pixiv_cat", "1")
        params.setdefault("pximg_mirror_host", "re")
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

            def _absolute_url(value: Any) -> str:
                """把 API 返回的绝对地址或相对路径统一为可下载地址。"""
                raw_url: str = str(value or "").strip()
                if raw_url.startswith(("http://", "https://")):
                    return raw_url
                if raw_url.startswith("/"):
                    return f"{self.base_url}{raw_url}"
                return ""

            # 当前 API 返回多种出图入口，保留全部地址以便线路故障时切换。
            candidate_urls: List[str] = []
            for raw_url in (
                urls_info.get("proxy"),
                urls_info.get("local"),
                urls_info.get("imgproxy"),
                urls_info.get("origin"),
                urls_info.get("legacy_single"),
            ):
                absolute_url: str = _absolute_url(raw_url)
                if absolute_url and absolute_url not in candidate_urls:
                    candidate_urls.append(absolute_url)
            image_url: str = candidate_urls[0] if candidate_urls else ""

            tags_raw: Any = payload.get("tags", [])
            tags: List[str] = [str(tag) for tag in tags_raw] if isinstance(tags_raw, list) else []

            return {
                "illust_id": str(image_info.get("illust_id", "")),
                "page_index": int(image_info.get("page_index", 0) or 0),
                "title": str(image_info.get("title", "未知")),
                "user_name": str(user_info.get("name", "未知")),
                "user_id": str(user_info.get("id", "")),
                "width": int(image_info.get("width", 0) or 0),
                "height": int(image_info.get("height", 0) or 0),
                "x_restrict": int(image_info.get("x_restrict", 0) or 0),
                "bookmarks": int(image_info.get("bookmark_count", 0) or 0),
                "views": int(image_info.get("view_count", 0) or 0),
                "comments": int(image_info.get("comment_count", 0) or 0),
                "ai_type": int(image_info.get("ai_type", 0) or 0),
                "illust_type": int(image_info.get("illust_type", 0) or 0),
                "created_at": str(image_info.get("created_at_pixiv", "") or ""),
                "tags": tags,
                "image_url": image_url,
                "image_urls": candidate_urls,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"解析 Random-image-api 响应失败: {exc}")
            return None

    async def download_image_base64(self, urls: Union[str, List[str]]) -> Optional[str]:
        """按候选地址顺序下载图片并转为 base64 字符串。

        先把图片下载到本地再发送，避免 QQ 平台拉取超时导致“图片已过期”。

        Args:
            urls: 单个图片地址，或按优先级排列的候选地址列表。

        Returns:
            成功时返回 base64 字符串，全部地址失败时返回 None。
        """
        candidates: List[str] = [urls] if isinstance(urls, str) else list(urls)
        return await asyncio.to_thread(self._download_candidates_sync, candidates)

    def _download_candidates_sync(self, urls: List[str]) -> Optional[str]:
        """同步尝试候选图片地址（由线程池调用），成功后返回 base64。"""
        for url in urls:
            if not url:
                continue
            req: request.Request = request.Request(
                url=url,
                method="GET",
                headers={"User-Agent": self.user_agent},
            )
            try:
                with request.urlopen(req, timeout=60) as resp:
                    image_bytes: bytes = resp.read()
                # 图片下载成功后才压缩，避免把错误响应当成图片处理。
                if self._compress_enabled:
                    image_bytes = self._compress_image(image_bytes)
                return base64.b64encode(image_bytes).decode("ascii")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"下载图片失败 ({url}): {exc}")
        return None

    def _compress_image(self, image_bytes: bytes) -> bytes:
        """压缩图片：限制最大宽度并转 JPEG。

        压缩过程出错（例如图片格式不支持）时返回原始数据，不影响发送。

        Args:
            image_bytes: 原始图片字节。

        Returns:
            压缩后的图片字节。
        """
        try:
            from PIL import Image

            img: Image.Image = Image.open(BytesIO(image_bytes))

            # 等比缩放：宽度超过上限时缩小。
            if img.width > self._max_image_width:
                ratio: float = self._max_image_width / img.width
                new_height: int = int(img.height * ratio)
                img = img.resize((self._max_image_width, new_height), Image.Resampling.LANCZOS)

            # 转为 RGB（JPEG 不支持 RGBA 透明通道）。
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")

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
# 配置模型（键名与原 config.toml 完全一致，WebUI 可直接生成表单）
#
# 注意：SDK 2.x 的 WebUI 只读取 json_schema_extra 里的 label / hint，
# description 仅用于生成 config.toml 注释。若不写 label，页面会回退显示
# 英文字段名，因此这里为每个字段显式声明中文 label。
# ============================================================
class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__: ClassVar[str] = "基础设置"

    enabled: bool = Field(
        default=True,
        description="是否启用插件",
        json_schema_extra={"label": "启用插件", "hint": "关闭后插件所有功能停止运行"},
    )
    config_version: str = Field(
        default="1.1.0",
        description="配置版本号（请勿手动修改）",
        json_schema_extra={"label": "配置版本号", "hint": "请勿手动修改，删除后插件会拒绝加载"},
    )


class ComponentsSectionConfig(PluginConfigBase):
    """组件开关。"""

    __ui_label__: ClassVar[str] = "组件开关"

    enable_command: bool = Field(
        default=True,
        description="是否启用随机图片命令",
        json_schema_extra={"label": "启用随机图片命令", "hint": "关闭后所有触发词都不会响应"},
    )


class ApiSectionConfig(PluginConfigBase):
    """API 连接配置。"""

    __ui_label__: ClassVar[str] = "API 配置"

    base_url: str = Field(
        default="https://i.mukyu.ru",
        description="API 服务地址（末尾不加 /）",
        json_schema_extra={"label": "API 服务地址", "hint": "Random-image-api 服务根地址，末尾不要加 /"},
    )
    timeout: int = Field(
        default=30,
        description="请求超时时间（秒），建议 15-60",
        json_schema_extra={"label": "请求超时（秒）", "hint": "建议 15-60；太短容易超时，太长等待时间过久"},
    )


class FilterSectionConfig(PluginConfigBase):
    """图片尺寸筛选。"""

    __ui_label__: ClassVar[str] = "尺寸筛选"

    min_width: int = Field(
        default=0,
        description="最小宽度（像素），0 表示不限制",
        json_schema_extra={"label": "最小宽度（像素）", "hint": "0 表示不限制"},
    )
    min_height: int = Field(
        default=0,
        description="最小高度（像素），0 表示不限制",
        json_schema_extra={"label": "最小高度（像素）", "hint": "0 表示不限制"},
    )
    min_pixels: int = Field(
        default=0,
        description="最小总像素数，0 表示不限制",
        json_schema_extra={"label": "最小总像素", "hint": "0 表示不限制；例如 1920×1080 = 2073600"},
    )


class FeaturesSectionConfig(PluginConfigBase):
    """功能行为配置。"""

    __ui_label__: ClassVar[str] = "功能配置"

    default_num: int = Field(
        default=1,
        description="默认获取图片数量（1-10）",
        json_schema_extra={"label": "默认获取数量", "hint": "用户不指定数量时使用，范围 1-10"},
    )
    max_num: int = Field(
        default=10,
        description="单次最大获取数量（1-10）",
        json_schema_extra={"label": "单次最大数量", "hint": "用户一次最多请求的张数，范围 1-10"},
    )
    cooldown_seconds: int = Field(
        default=10,
        description="命令全局冷却时间（秒），0=不限制；任意用户触发后所有人共享这段冷却",
        json_schema_extra={"label": "全局冷却（秒）", "hint": "0 = 不限制；触发后所有用户共享这段冷却"},
    )
    allow_r18: bool = Field(
        default=False,
        description="R18 总开关；默认关闭，开启后 R18 请求才会生效",
        json_schema_extra={"label": "R18 总开关", "hint": "关闭后所有 R18 请求都会自动回退为非 R18"},
    )
    # 用 Literal 声明可选值，SDK 会自动把该字段渲染成下拉框。
    r18_mode: Literal["blacklist", "whitelist"] = Field(
        default="blacklist",
        description="R18 名单模式：blacklist=默认全局允许，名单内禁用；whitelist=只有名单内允许",
        json_schema_extra={
            "label": "R18 名单模式",
            "hint": "blacklist = 默认允许、名单内禁用；whitelist = 只有名单内允许",
        },
    )
    r18_user_id_list: str = Field(
        default="",
        description="R18 QQ 号名单；blacklist 模式下是禁用名单，whitelist 模式下是允许名单；支持换行/逗号/空格分隔",
        json_schema_extra={
            "label": "R18 QQ 号名单",
            "hint": "blacklist 模式下为禁用名单，whitelist 模式下为允许名单；支持换行 / 逗号 / 空格分隔",
        },
    )
    r18_group_id_list: str = Field(
        default="",
        description="R18 群号名单；blacklist 模式下是禁用名单，whitelist 模式下是允许名单；支持换行/逗号/空格分隔",
        json_schema_extra={
            "label": "R18 群号名单",
            "hint": "blacklist 模式下为禁用名单，whitelist 模式下为允许名单；支持换行 / 逗号 / 空格分隔",
        },
    )
    exclude_ai: bool = Field(
        default=True,
        description="是否默认排除 AI 绘图作品（推荐开启）",
        json_schema_extra={"label": "默认排除 AI 作品", "hint": "开启后默认只看人工绘图；命令里输入 ai 可临时覆盖"},
    )
    # 用 Literal 声明可选值，SDK 会自动把该字段渲染成下拉框。
    default_strategy: Literal["quality", "random"] = Field(
        default="random",
        description="默认策略：quality（质量优先）或 random（纯随机，更快）",
        json_schema_extra={"label": "默认获取策略", "hint": "quality = 质量优先但较慢；random = 纯随机更快"},
    )
    quality_samples: int = Field(
        default=10,
        description="quality 策略采样数（1-1000），数字越小响应越快，建议 10-50",
        json_schema_extra={"label": "质量采样数", "hint": "仅 quality 策略生效，范围 1-1000；越小越快，建议 10-50"},
    )
    use_forward_message: bool = Field(
        default=True,
        description="是否使用合并转发格式发送（推荐开启）",
        json_schema_extra={"label": "使用合并转发", "hint": "开启后图片和文字会打包成聊天记录一起发送"},
    )


class ImageCompressSectionConfig(PluginConfigBase):
    """图片压缩配置。"""

    __ui_label__: ClassVar[str] = "图片压缩"

    enabled: bool = Field(
        default=True,
        description="是否启用图片压缩（推荐开启，大幅提升发送速度）",
        json_schema_extra={"label": "启用图片压缩", "hint": "开启后下载图片会自动压缩，发送更快"},
    )
    max_image_width: int = Field(
        default=1280,
        description="图片最大宽度（像素），超过会等比缩小，建议 800-2560",
        json_schema_extra={"label": "图片最大宽度（像素）", "hint": "超过会等比缩小，建议 800-2560"},
    )
    jpeg_quality: int = Field(
        default=85,
        description="JPEG 压缩质量（1-100），85 为推荐值",
        json_schema_extra={"label": "JPEG 压缩质量", "hint": "1-100，越大越清晰体积越大；85 为推荐值"},
    )


class RandomImageConfig(PluginConfigBase):
    """随机图片插件配置模型。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig, description="插件基础配置")
    components: ComponentsSectionConfig = Field(default_factory=ComponentsSectionConfig, description="组件开关")
    api: ApiSectionConfig = Field(default_factory=ApiSectionConfig, description="API 连接配置")
    filter: FilterSectionConfig = Field(default_factory=FilterSectionConfig, description="图片尺寸筛选")
    features: FeaturesSectionConfig = Field(default_factory=FeaturesSectionConfig, description="功能行为配置")
    image_compress: ImageCompressSectionConfig = Field(
        default_factory=ImageCompressSectionConfig,
        description="图片压缩配置",
    )


# ============================================================
# 插件主类
# ============================================================
class RandomImagePlugin(MaiBotPlugin):
    """Random-image-api-for-MaiBot 插件（SDK 2.x 版）。"""

    config_model = RandomImageConfig

    # 全局冷却状态。插件实例唯一，因此实例级状态等价于原版的类级全局状态。
    _cooldown_until: float = 0.0
    _cooldown_lock: Optional[asyncio.Lock] = None

    async def on_load(self) -> None:
        """插件加载完成后的回调。"""
        logger.info("随机图片插件已加载")

    async def on_unload(self) -> None:
        """插件卸载前的回调，重置冷却状态。"""
        self._cooldown_until = 0.0
        logger.info("随机图片插件已卸载")

    async def on_config_update(self, scope: str, config_data: Dict[str, Any], version: str) -> None:
        """配置热更新回调。

        Runner 在调用本方法前已把最新配置注入插件实例，这里只做日志记录。
        """
        del config_data, version
        if scope == "self":
            logger.info("随机图片插件配置已更新")

    def _get_cooldown_lock(self) -> asyncio.Lock:
        """惰性创建冷却锁，避免在事件循环之外构造 asyncio 原语。"""
        if self._cooldown_lock is None:
            self._cooldown_lock = asyncio.Lock()
        return self._cooldown_lock

    @Command(
        "random_image",
        description="从 Random-image-api 获取随机 Pixiv 图片（支持多图、标签筛选、R18、尺寸筛选）",
        pattern=COMMAND_PATTERN,
    )
    async def handle_random_image(
        self,
        text: str = "",
        stream_id: str = "",
        user_id: str = "",
        group_id: str = "",
        matched_groups: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Tuple[bool, str, bool]:
        """命令执行入口。

        返回值中的第三项为 True 时拦截后续聊天处理，与旧版返回 2 的语义一致。
        """
        del text, kwargs
        if not stream_id:
            return False, "无法获取聊天流信息", True

        cfg = self.config
        if not cfg.components.enable_command:
            return False, "随机图片命令已禁用", True

        groups: Dict[str, str] = matched_groups or {}
        args_str: str = str(groups.get("args", "") or "").strip()
        trigger_word: str = str(groups.get("trigger", "") or "")

        # help 命令。
        if args_str.lower() in ("help", "帮助", "?", "？"):
            await self.ctx.send.text(COMMAND_HELP, stream_id)
            return True, "显示帮助", True

        features = cfg.features
        cooldown: int = features.cooldown_seconds
        allow_r18: bool = features.allow_r18
        r18_mode: str = self._normalize_r18_mode(features.r18_mode)
        r18_user_ids: set[str] = self._parse_id_list(features.r18_user_id_list)
        r18_group_ids: set[str] = self._parse_id_list(features.r18_group_id_list)

        reserved_cooldown_until: float = 0.0
        cooldown_remaining, reserved_cooldown_until = await self._check_and_reserve_global_cooldown(cooldown)
        if cooldown_remaining > 0:
            await self.ctx.send.text(f"全局冷却中，请 {cooldown_remaining} 秒后再试~", stream_id)
            return True, "全局冷却中", True

        r18_allowed: bool = self._is_r18_allowed(
            allow_r18=allow_r18,
            mode=r18_mode,
            configured_user_ids=r18_user_ids,
            configured_group_ids=r18_group_ids,
            user_id=user_id,
            group_id=group_id,
        )

        # 解析用户参数。
        params, request_count = self._parse_args(
            args_str,
            allow_r18=r18_allowed,
            default_strategy=features.default_strategy,
            default_num=features.default_num,
            max_num=features.max_num,
            exclude_ai=features.exclude_ai,
            trigger_word=trigger_word,
            quality_samples=features.quality_samples,
        )

        # 把尺寸筛选配置注入请求参数（用户显式参数已存在时不覆盖）。
        filter_cfg = cfg.filter
        if filter_cfg.min_width > 0:
            params.setdefault("min_width", str(filter_cfg.min_width))
        if filter_cfg.min_height > 0:
            params.setdefault("min_height", str(filter_cfg.min_height))
        if filter_cfg.min_pixels > 0:
            params.setdefault("min_pixels", str(filter_cfg.min_pixels))

        api_client = RandomImageAPI(base_url=cfg.api.base_url, timeout=cfg.api.timeout)
        api_client.set_compress_options(
            cfg.image_compress.enabled,
            cfg.image_compress.max_image_width,
            cfg.image_compress.jpeg_quality,
        )

        await self.ctx.send.text(f"正在获取{request_count}张随机图片，请稍候...", stream_id)

        # 获取图片。
        results: List[Dict[str, Any]]
        if request_count == 1:
            single_result: Optional[Dict[str, Any]] = await api_client.get_random_image(params)
            results = [single_result] if single_result else []
        else:
            results = await api_client.get_multiple_images(params, request_count)

        if not results:
            await self._release_global_cooldown(reserved_cooldown_until)
            await self.ctx.send.text("没有找到符合条件的图片，换个条件试试？", stream_id)
            return True, "未找到图片", True

        await self._activate_global_cooldown(reserved_cooldown_until, cooldown)

        # 发送结果。
        if features.use_forward_message:
            await self._send_as_forward(results, api_client, stream_id)
        else:
            await self._send_as_separate(results, api_client, stream_id)

        return True, f"发送{len(results)}张随机图片成功", True

    async def _send_as_forward(
        self,
        results: List[Dict[str, Any]],
        api_client: RandomImageAPI,
        stream_id: str,
    ) -> None:
        """以合并转发消息形式发送，先并发下载全部图片再统一发送。"""
        bot_qq: str = str(await self.ctx.config.get("bot.qq_account", "") or "")
        bot_name: str = str(await self.ctx.config.get("bot.nickname", "麦麦") or "麦麦")

        image_candidates: List[List[str]] = [
            [str(url) for url in result.get("image_urls", []) if str(url)]
            or [str(result.get("image_url", "") or "")]
            for result in results
        ]

        download_tasks: List[asyncio.Task[Optional[str]]] = []
        for urls in image_candidates:
            if urls and urls[0]:
                download_tasks.append(asyncio.create_task(api_client.download_image_base64(urls)))
            else:
                download_tasks.append(asyncio.create_task(self._empty_image()))
        base64_results: List[Optional[str]] = await asyncio.gather(*download_tasks)

        # 新版 send.forward 使用字典节点：{"user_id", "nickname", "segments"}。
        forward_nodes: List[Dict[str, Any]] = []
        for index, result in enumerate(results):
            info_text: str = self._format_result(result)
            image_b64: Optional[str] = base64_results[index] if index < len(base64_results) else None
            segments: List[Dict[str, Any]] = [{"type": "text", "content": info_text}]
            if image_b64:
                segments.append({"type": "image", "content": image_b64})
            elif image_candidates[index]:
                # 下载失败时不塞图片段，避免把无效 base64 传给平台，改为附加原图链接。
                segments.append({"type": "text", "content": f"🔗 原图：{image_candidates[index][0]}"})
            forward_nodes.append(
                {
                    "user_id": bot_qq,
                    "nickname": bot_name,
                    "segments": segments,
                }
            )

        await self.ctx.send.forward(forward_nodes, stream_id)

    async def _send_as_separate(
        self,
        results: List[Dict[str, Any]],
        api_client: RandomImageAPI,
        stream_id: str,
    ) -> None:
        """逐条发送（关闭合并转发时的备选方案）。"""
        for result in results:
            info_text: str = self._format_result(result)
            await self.ctx.send.text(info_text, stream_id)
            image_candidates: List[str] = [
                str(url) for url in result.get("image_urls", []) if str(url)
            ] or [str(result.get("image_url", "") or "")]
            if not image_candidates[0]:
                continue
            try:
                image_b64: Optional[str] = await api_client.download_image_base64(image_candidates)
                if image_b64:
                    await self.ctx.send.image(image_b64, stream_id)
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"发送图片失败: {exc}")

            # 下载失败时先尝试平台侧拉取，仍失败则退回文本链接。
            try:
                sent = await self.ctx.send.custom("imageurl", image_candidates[0], stream_id)
                if sent:
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"发送 imageurl 消息失败: {exc}")
            await self.ctx.send.text(f"🔗 原图：{image_candidates[0]}", stream_id)

    @staticmethod
    async def _empty_image() -> Optional[str]:
        """占位协程：没有候选地址时返回 None，保持下载任务数量与结果数量一致。"""
        return None

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
        """解析用户输入参数。

        Returns:
            (API 请求参数字典, 请求数量)
        """
        params: Dict[str, Any] = {"format": "json"}
        # 配置了默认排除 AI 时先设置，用户输入 "ai" 可覆盖。
        if exclude_ai:
            params["ai_type"] = "0"
        # 触发词属于涩图系列时自动请求 R18（需管理员开启 allow_r18）。
        if allow_r18 and trigger_word in _SETU_TRIGGERS:
            params["r18"] = "1"
            params["ai_type"] = "0"

        request_count: int = default_num
        tokens: List[str] = args_str.split()

        included_tags: List[str] = []
        excluded_tags: List[str] = []

        for token in tokens:
            lower: str = token.lower()

            # 数量（纯数字）。
            if re.match(r"^\d+$", token):
                num: int = int(token)
                if 1 <= num <= max_num:
                    request_count = num
                continue

            # 排除标签 -#xxx。
            if token.startswith("-#") and len(token) > 2:
                excluded_tags.append(token[2:])
                continue

            # 包含标签 #xxx。
            if token.startswith("#") and len(token) > 1:
                included_tags.append(token[1:])
                continue

            # R18。
            if lower == "r18":
                if allow_r18:
                    params["r18"] = "1"
                continue

            # AI 过滤。
            if lower == "noai":
                params["ai_type"] = "0"
                continue
            if lower == "ai":
                params["ai_type"] = "1"
                continue

            # 方向。
            if token in ("横屏", "横图"):
                params["orientation"] = "landscape"
                continue
            if token in ("竖屏", "竖图"):
                params["orientation"] = "portrait"
                continue
            if token in ("方形", "方图"):
                params["orientation"] = "square"
                continue

            # 画师 uid:xxx。
            uid_match = re.match(r"^uid[：:](\d+)$", token)
            if uid_match:
                params["user_id"] = uid_match.group(1)
                continue

            # 收藏阈值 收藏>xxx。
            bookmark_match = re.match(r"^收藏[>＞](\d+)$", token)
            if bookmark_match:
                params["min_bookmarks"] = bookmark_match.group(1)
                continue

            # 浏览阈值 浏览>xxx。
            view_match = re.match(r"^浏览[>＞](\d+)$", token)
            if view_match:
                params["min_views"] = view_match.group(1)
                continue

            # 宽度阈值 宽度>xxx。
            width_match = re.match(r"^宽度[>＞](\d+)$", token)
            if width_match:
                params["min_width"] = width_match.group(1)
                continue

            # 高度阈值 高度>xxx。
            height_match = re.match(r"^高度[>＞](\d+)$", token)
            if height_match:
                params["min_height"] = height_match.group(1)
                continue

            # 像素阈值 像素>xxx。
            pixels_match = re.match(r"^像素[>＞](\d+)$", token)
            if pixels_match:
                params["min_pixels"] = pixels_match.group(1)
                continue

            # 策略。
            if lower in ("quality", "random"):
                params["strategy"] = lower
                continue

            # 作品类型。
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

        # 设置默认策略。
        if "strategy" not in params:
            params["strategy"] = default_strategy
        # quality 策略时附加采样数量（限制服务端采样范围，减少响应时间）。
        if params.get("strategy") == "quality" and quality_samples > 0:
            params["quality_samples"] = str(quality_samples)

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

    @staticmethod
    def _normalize_r18_mode(value: Any) -> str:
        """规范化 R18 名单模式，只允许 whitelist / blacklist。"""
        mode: str = str(value or "blacklist").strip().lower()
        if mode not in {"whitelist", "blacklist"}:
            return "blacklist"
        return mode

    @staticmethod
    def _parse_id_list(value: Any) -> set[str]:
        """把配置中的群号/QQ 号列表解析成集合。"""
        if isinstance(value, (list, tuple, set)):
            raw_items: List[str] = [str(item).strip() for item in value]
        else:
            raw_text: str = str(value or "")
            raw_items = re.split(r"[\s,，;；]+", raw_text)

        return {item for item in raw_items if item}

    async def _check_and_reserve_global_cooldown(self, cooldown: int) -> Tuple[int, float]:
        """检查并预占全局冷却，返回 (剩余秒数, 预占到的时间戳)。"""
        if cooldown <= 0:
            return 0, 0.0

        async with self._get_cooldown_lock():
            now: float = time.time()
            if now < self._cooldown_until:
                return int(self._cooldown_until - now) + 1, 0.0

            reserved_until: float = now + cooldown
            self._cooldown_until = reserved_until
            return 0, reserved_until

    async def _release_global_cooldown(self, reserved_until: float) -> None:
        """请求失败时释放本次预占的冷却时间。"""
        if reserved_until <= 0:
            return

        async with self._get_cooldown_lock():
            if self._cooldown_until == reserved_until:
                self._cooldown_until = 0.0

    async def _activate_global_cooldown(self, reserved_until: float, cooldown: int) -> None:
        """请求成功后把预占位切换为正式的全局冷却。"""
        if cooldown <= 0 or reserved_until <= 0:
            return

        async with self._get_cooldown_lock():
            if self._cooldown_until == reserved_until:
                self._cooldown_until = time.time() + cooldown

    def _is_r18_allowed(
        self,
        allow_r18: bool,
        mode: str,
        configured_user_ids: set[str],
        configured_group_ids: set[str],
        user_id: str,
        group_id: str,
    ) -> bool:
        """根据总开关和群号/QQ 名单判断当前上下文是否允许 R18。"""
        if not allow_r18:
            return False

        normalized_user_id: str = str(user_id or "")
        normalized_group_id: str = str(group_id or "")
        matched: bool = normalized_user_id in configured_user_ids or (
            bool(normalized_group_id) and normalized_group_id in configured_group_ids
        )

        if mode == "whitelist":
            return matched

        return not matched


def create_plugin() -> RandomImagePlugin:
    """创建插件实例，供 Runner 加载。"""
    return RandomImagePlugin()
