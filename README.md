# Random-image-api-for-MaiBot

从 [Random-image-api](https://github.com/inliver233/Random-image-api) 拉取 Pixiv 随机图片的 MaiBot 插件。

> **当前版本 v1.2.1（MaiBot 1.3.x / Plugin SDK 2.x 适配版）**
>
> 这是一个 breaking change：本版本使用新版插件协议（`manifest_version: 2` + `maibot_sdk`），
> **只能运行在 MaiBot 1.3.x 及以上**。0.12.x 旧版代码保留在 `v1.1.0-maibot-0.12` tag 中。

## ✨ 功能

- 随机获取 Pixiv 高质量图片，一次最多 10 张
- 标签包含 / 排除（`#原神`、`-#AI`）
- AI 绘图过滤（`noai` / `ai`）
- 方向筛选（横屏 / 竖屏 / 方形）
- 画师、收藏、浏览、宽高、像素阈值
- R18 总开关 + 白名单 / 黑名单
- 全局冷却，防止刷屏
- 多线路回退（proxy / local / imgproxy / origin）
- 先下载再发送，避免“图片已过期”
- Pillow 智能压缩，大图发送速度提升数十倍
- 合并转发或逐条发送

## 📦 安装

把插件目录放进：

```
MaiBot/plugins/random_image_api_base64/
├── _manifest.json
├── plugin.py
└── README.md
```

然后：

1. 重启 MaiBot，或在 WebUI 插件管理里重载本插件
2. 首次加载会自动生成 `config.toml`，默认已启用（`plugin.enabled = true`）
3. 需要调整时，编辑 `config.toml` 或在 WebUI 插件配置页修改

插件依赖 `Pillow`，已在 `_manifest.json` 中声明，框架会自动检查。

## ⚙️ 配置说明

配置位于 `config.toml`，也可以在 WebUI 的插件配置页直接修改。
**请勿删除 `plugin.config_version`，否则插件会拒绝加载。**

### 基础开关

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `plugin.enabled` | `true` | 插件总开关 |
| `components.enable_command` | `true` | 是否启用随机图片命令 |

### API 配置

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `api.base_url` | `https://i.mukyu.ru` | Random-image-api 服务地址，末尾不加 `/` |
| `api.timeout` | `30` | 请求超时（秒），建议 15-60 |

### 图片筛选

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `filter.min_width` | `0` | 最小宽度，0=不限制 |
| `filter.min_height` | `0` | 最小高度，0=不限制 |
| `filter.min_pixels` | `0` | 最小总像素，0=不限制 |

### 功能配置

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `features.default_num` | `1` | 默认获取数量（1-10） |
| `features.max_num` | `10` | 单次最大数量（1-10） |
| `features.cooldown_seconds` | `10` | 全局冷却秒数，0=不限制 |
| `features.allow_r18` | `false` | R18 总开关，默认关闭 |
| `features.r18_mode` | `blacklist` | `blacklist`=默认允许、名单内禁用；`whitelist`=只有名单内允许 |
| `features.r18_user_id_list` | `""` | QQ 号名单，支持换行/逗号/空格分隔 |
| `features.r18_group_id_list` | `""` | 群号名单，支持换行/逗号/空格分隔 |
| `features.exclude_ai` | `true` | 默认排除 AI 绘图 |
| `features.default_strategy` | `random` | `quality`=质量优先，`random`=纯随机更快 |
| `features.quality_samples` | `10` | quality 策略采样数（1-1000） |
| `features.use_forward_message` | `true` | 是否使用合并转发格式发送 |

### 图片压缩

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `image_compress.enabled` | `true` | 是否压缩后再发送 |
| `image_compress.max_image_width` | `1280` | 最大宽度，超过等比缩小 |
| `image_compress.jpeg_quality` | `85` | JPEG 质量（1-100） |

## 🎮 使用方法

### 触发词

```
随机图片 / 来张图 / 来点图 / 来点图片 / 来张图片
涩图 / 来张涩图 / 来点涩图 / 随机涩图
setu / 来张setu / 来点setu
```

### 参数速查

| 参数 | 作用 | 示例 |
|---|---|---|
| `数字` | 获取数量（1-10） | `3` |
| `#标签` | 包含标签 | `#原神` |
| `-#标签` | 排除标签 | `-#AI` |
| `r18` | 请求 R18 | `随机图片 r18` |
| `noai` | 只要非 AI 图 | `noai` |
| `ai` | 只要 AI 图 | `ai` |
| `横屏/竖屏/方形` | 方向筛选 | `横屏` |
| `uid:数字` | 指定画师 | `uid:12345678` |
| `收藏>数字` | 收藏下限 | `收藏>1000` |
| `浏览>数字` | 浏览下限 | `浏览>5000` |
| `宽度>数字` | 最小宽度 | `宽度>1920` |
| `高度>数字` | 最小高度 | `高度>1080` |
| `像素>数字` | 最小像素 | `像素>2073600` |
| `quality/random` | 切换策略 | `quality` |
| `插画/漫画/动图` | 作品类型 | `漫画` |
| `help` | 查看帮助 | `随机图片 help` |

### 组合示例

```
来张图 3 #原神 noai 横屏
随机图片 5 #碧蓝航线 收藏>1500
随机图片 #白丝 -#AI uid:12345678
来张涩图 #FGO quality
来张涩图 3 收藏>2000 横屏
```

## 🔞 R18 名单怎么理解

**黑名单模式（默认）**：整体允许 R18，名单里的群 / QQ 号会被禁用并自动回退普通图。适合大部分群都能开、只有少数需要禁用的场景。

**白名单模式**：只有名单里的群 / QQ 号允许 R18。适合只想在少数环境开放。

**总开关关闭**：`allow_r18 = false` 时所有 R18 请求都会自动回退为非 R18，名单不再生效。

> **R18 默认关闭。** 需要时在 `config.toml` 或 WebUI 里把 `features.allow_r18` 改为 `true`，并确保符合当地法律法规。

## ⚠️ 从 0.12.x 迁移

- 新版必须使用 `manifest_version: 2`，旧版清单文件无法直接加载
- 如果从旧配置迁移，`config.toml` 里必须有 `plugin.config_version = "1.1.0"`
- 旧版目录名任意；新版建议保持 `random_image_api_base64`
- 0.12.x 代码见 tag：`v1.1.0-maibot-0.12`

## 🔧 故障排除

### 获取不到图片

1. 打开日志确认 `Random-image-api` 请求是否超时
2. 检查 `api.base_url` 是否可用（浏览器访问 `https://i.mukyu.ru/random?format=json`）
3. 把 `api.timeout` 调大，或更换 `api.base_url`

### 说了“涩图”却没出 R18

按顺序检查：

1. `features.allow_r18` 是否为 `true`
2. `features.r18_mode` 是否正确
3. `r18_group_id_list` / `r18_user_id_list` 是否把当前会话挡掉了

### 发图太慢

- 确认 `image_compress.enabled = true`
- 把 `image_compress.max_image_width` 降到 1080 或更低
- 把 `api.timeout` 保持在 30 左右，避免长时间等待

## 📝 更新日志

### v1.2.1

- 默认关闭 R18（`features.allow_r18 = false`），需要时在配置中手动开启
- 已存在的 `config.toml` 不会被覆盖，升级后仍保留原有设置

### v1.2.0（MaiBot 1.3.x 适配版）

- 迁移到 `manifest_version: 2` 与 `maibot_sdk`
- 命令注册改为 `@Command` 装饰器，触发词与参数完全保持
- 发送改为 `ctx.send.text / image / forward / custom`
- 配置改为 `PluginConfigBase`，WebUI 可直接编辑
- 适配新版合并转发消息格式
- 移除旧版对主进程 VLM 的 monkey-patch（新架构不需要）
- 功能、命令、参数、R18 规则与 0.12.x 版本保持一致

### v1.1.0（0.12.x）

- 适配新版 Random-image-api 图片地址契约
- 多地址回退、全局冷却、R18 白名单/黑名单
- 标签筛选、多图、AI 过滤、合并转发、智能压缩

### v1.0.2

- 修复配置与图片地址兼容问题

### v1.0.1

- 修复类属性访问问题

### v1.0.0

- 首个发布版本

## 免责声明

**请合理使用本插件。** 插件仅负责从公开 API 获取随机图片，不对图片内容负责。
使用者需自行确保使用场景符合当地法律法规与平台规则。
