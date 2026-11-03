"""根据官方 GitHub Releases 生成 AltStore / Feather 使用的 JSON 源。

输入：当前目录下的 config.yaml
输出：output/source.json

环境变量（均为可选）：
    GITHUB_TOKEN  GitHub 访问令牌，可将 API 限额从匿名的 60 次/小时提高到 5000 次/小时
    LOG_LEVEL     日志级别，默认 INFO；排查被跳过的 release 时可设为 DEBUG

运行要求：Python 3.10+（使用了 `X | None` 类型语法）。
"""
import json
import logging
import os
import re
import shutil
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterator, TypedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

logger = logging.getLogger("generate_source")

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

RELEASES_URL = "https://api.github.com/repos/{repo}/releases?per_page={per_page}"
# 单次拉取的 release 数量。预发布、草稿、资源不匹配的条目都会被过滤，
# 因此要比最终保留的版本数多取一些，以保证过滤后仍能凑够 MAX_VERSIONS 个。
PER_PAGE = 60
# 每个应用在源中保留的历史版本数量（含最新版）
MAX_VERSIONS = 5

REQUEST_TIMEOUT = 30  # 单次请求超时（秒）
MAX_ATTEMPTS = 3  # 最大尝试次数（含首次）
BACKOFF_BASE = 2  # 指数退避底数：第 n 次失败后等待 BACKOFF_BASE ** n 秒
# 视为临时故障、值得重试的 HTTP 状态码；404、401 等确定性错误不重试
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_WORKERS = 8  # 并发请求线程数上限

# 源顶层需要原样从配置中复制的字段
FEED_KEYS = ("name", "identifier", "subtitle", "iconURL")
OUTPUT_FILE = Path("output") / "source.json"
ICON_SOURCE_DIR = Path("icons")
ICON_OUTPUT_DIR = OUTPUT_FILE.parent / "icons"

# 解析 release 的 tag，规则如下：
#   - 可带 v / V 前缀，主体为点分数字，如 1.2.3
#   - 允许的后缀只有三种：-release、+<数字>（构建号）、-<数字>（修订号）
#   - 构建号与 -release 不参与版本比较，直接丢弃
#   - 修订号（如 iTorrent 的 v2.2.0-1）会作为额外的数字段保留（2.2.0.1），
#     以避免同一主版本的不同修订互相冲突
TAG_RE = re.compile(
    r"""
    [vV]?
    (?P<num>\d+(?:\.\d+)*)
    (?:
        -release
      | \+\d+
      | -(?P<rev>\d+)
    )?
    """,
    re.VERBOSE | re.IGNORECASE,
)

CONFIG = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

class VersionInfo(TypedDict):
    """AltStore 源中 `versions` 数组的单个元素。"""

    version: str
    date: str
    downloadURL: str
    size: int


@dataclass(frozen=True)
class Source:
    """config.yaml 中 upstream 下的一个上游应用。"""

    key: str  # config.yaml 中 upstream 的唯一标识
    repo: str  # "owner/name" 形式的 GitHub 仓库
    name: str
    bundle_identifier: str
    description: str
    asset_pattern: re.Pattern[str]  # 用于匹配 IPA 文件名的预编译正则

    @property
    def owner(self) -> str:
        return self.repo.partition("/")[0]

    @classmethod
    def from_config(cls, key: str, raw: dict[str, str]) -> "Source":
        """从配置字典构造，并在配置有误时给出明确的错误信息。"""
        try:
            return cls(
                key=key,
                repo=raw["repo"],
                name=raw["name"],
                bundle_identifier=raw["bundleIdentifier"],
                description=raw["description"],
                asset_pattern=re.compile(raw["asset"]),
            )
        except KeyError as exc:
            raise ValueError(f"上游配置缺少字段 {exc}：{raw!r}") from exc
        except re.error as exc:
            raise ValueError(f"上游配置的 asset 不是合法正则（{exc}）：{raw!r}") from exc


# ---------------------------------------------------------------------------
# 网络请求
# ---------------------------------------------------------------------------

def _build_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    if token := os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get_json(url: str) -> Any:
    """GET 并解析 JSON；遇到临时性故障时按指数退避重试。"""
    request = Request(url, headers=_build_headers())
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                return json.load(response)
        except HTTPError as exc:  # HTTPError 是 URLError 的子类，必须先捕获
            if exc.code not in RETRYABLE_STATUS:
                raise
            last_error = exc
        except (URLError, TimeoutError) as exc:
            last_error = exc
        if attempt < MAX_ATTEMPTS:
            delay = BACKOFF_BASE**attempt
            logger.warning("请求失败（%s），%d 秒后重试（%d/%d）：%s",
                           last_error, delay, attempt, MAX_ATTEMPTS, url)
            time.sleep(delay)
    raise RuntimeError(f"请求在 {MAX_ATTEMPTS} 次尝试后仍失败：{url}") from last_error


def fetch_releases(repo: str) -> list[dict[str, Any]]:
    """获取仓库最近的 release 列表（GitHub 按创建时间倒序返回）。"""
    data = _get_json(RELEASES_URL.format(repo=repo, per_page=PER_PAGE))
    # 出错时 GitHub 可能返回 {"message": ...} 之类的字典，需显式校验
    if not isinstance(data, list):
        raise ValueError(f"{repo}：响应格式异常，期望列表，实际为 {type(data).__name__}")
    return data


# ---------------------------------------------------------------------------
# 解析与筛选
# ---------------------------------------------------------------------------

def version_of(tag: str) -> str | None:
    """从 tag 提取版本号，无法识别时返回 None。规则见 TAG_RE 处的说明。"""
    if not (match := TAG_RE.fullmatch(tag)):
        return None
    if revision := match["rev"]:
        return f"{match['num']}.{revision}"
    return match["num"]


def pick_asset(release: dict[str, Any], source: Source) -> dict[str, Any] | None:
    """返回 release 中唯一匹配的合法 asset；不满足条件时返回 None 并记录原因。

    匹配到 0 个或多个都视为异常：前者说明没有 IPA，后者说明正则不够精确，
    此时无法确定该用哪个文件，宁可跳过也不要猜。
    """
    tag = release.get("tag_name", "")
    matched = [a for a in release.get("assets", []) if source.asset_pattern.fullmatch(a["name"])]
    if len(matched) != 1:
        logger.warning("%s@%s：匹配到 %d 个资源（期望 1 个），已跳过", source.repo, tag, len(matched))
        return None
    asset = matched[0]
    if not asset.get("size") or not asset.get("browser_download_url"):
        logger.warning("%s@%s：资源 %s 缺少 size 或下载地址，已跳过", source.repo, tag, asset["name"])
        return None
    return asset


def iter_versions(releases: list[dict[str, Any]], source: Source) -> Iterator[VersionInfo]:
    """按 release 的原有顺序（新到旧）惰性产出合法版本，同一版本号只取第一次出现的。"""
    seen: set[str] = set()
    for release in releases:
        tag = release.get("tag_name", "")
        if release.get("draft") or release.get("prerelease"):
            logger.debug("%s@%s：草稿或预发布，已跳过", source.repo, tag)
            continue
        version = version_of(tag)
        if version is None:
            logger.debug("%s@%s：tag 格式无法识别，已跳过", source.repo, tag)
            continue
        if version in seen:
            logger.debug("%s@%s：版本号 %s 重复，已跳过", source.repo, tag, version)
            continue
        date = release.get("published_at") or release.get("created_at")
        if not date:
            logger.warning("%s@%s：缺少发布时间，已跳过", source.repo, tag)
            continue
        if (asset := pick_asset(release, source)) is None:
            continue
        seen.add(version)
        yield {
            "version": version,
            "date": date[:10],  # ISO 8601 时间戳只保留 YYYY-MM-DD
            "downloadURL": asset["browser_download_url"],
            "size": asset["size"],
        }


# ---------------------------------------------------------------------------
# 组装与输出
# ---------------------------------------------------------------------------

def build_app(source: Source, releases: list[dict[str, Any]]) -> dict[str, Any]:
    """将筛选出的版本列表组装成源中的一个 app 条目。"""
    # islice 配合生成器：凑够 MAX_VERSIONS 个后不再继续解析剩余 release
    versions = list(islice(iter_versions(releases, source), MAX_VERSIONS))
    if not versions:
        raise ValueError(f"{source.repo}：没有找到符合条件的 release")
    newest = versions[0]  # releases 为新到旧顺序，首个即最新版
    return {
        "name": source.name,
        "bundleIdentifier": source.bundle_identifier,
        "developerName": source.owner,
        "subtitle": source.description,
        "localizedDescription": source.description,
        "iconURL": f"{CONFIG['iconBaseURL'].rstrip('/')}/{source.key}.png",
        "version": newest["version"],
        "versionDate": newest["date"],
        "size": newest["size"],
        "downloadURL": newest["downloadURL"],
        "versions": versions,
    }


def load_app(source: Source) -> dict[str, Any]:
    """拉取并构建单个应用（在工作线程中执行）。"""
    app = build_app(source, fetch_releases(source.repo))
    logger.info("%s：最新版本 %s，共 %d 个版本", source.repo, app["version"], len(app["versions"]))
    return app



def publish_icons(sources: list[Source]) -> None:
    """校验并复制静态应用图标，随 source.json 一同发布到 Pages。"""
    ICON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for source in sources:
        icon = ICON_SOURCE_DIR / f"{source.key}.png"
        data = icon.read_bytes()  # 缺少图标立即失败，避免发布无效 iconURL
        if not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) < 24:
            raise ValueError(f"{icon} 不是有效的 PNG")
        width, height = struct.unpack(">II", data[16:24])
        if width != 512 or height != 512:
            raise ValueError(f"{icon} 图标尺寸不符合要求：{width}x{height}")
        shutil.copyfile(icon, ICON_OUTPUT_DIR / icon.name)


def write_feed(feed: dict[str, Any]) -> None:
    """原子写入：先写临时文件再替换，避免中途失败留下残缺的 JSON。"""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = OUTPUT_FILE.with_suffix(".json.tmp")
    tmp_file.write_text(
        json.dumps(feed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp_file.replace(OUTPUT_FILE)


def main() -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    sources = [Source.from_config(key, raw) for key, raw in CONFIG["upstream"].items()]

    apps: list[dict[str, Any]] = []
    failed: list[str] = []
    # 任务以网络 I/O 为主，用线程池并发请求各仓库；
    # 按提交顺序取结果，保证 apps 的顺序与配置文件一致
    with ThreadPoolExecutor(max_workers=max(1, min(MAX_WORKERS, len(sources)))) as pool:
        futures = [(source, pool.submit(load_app, source)) for source in sources]
        for source, future in futures:
            try:
                apps.append(future.result())
            except Exception as exc:  # 单个仓库失败不应中断其余仓库的处理
                failed.append(source.repo)
                logger.error("处理 %s 失败：%s", source.repo, exc)

    # 只要有失败就不写文件：发布缺少某些应用的源会让用户的订阅里凭空少了应用，
    # 保留上一次的正确输出更安全；返回非零退出码让 CI 能感知失败
    if failed:
        logger.error("以下仓库处理失败，未生成源文件：%s", ", ".join(failed))
        return 1

    feed = {key: CONFIG[key] for key in FEED_KEYS}
    feed.update(apps=apps, news=[])
    publish_icons(sources)
    write_feed(feed)
    logger.info("已生成 %s（共 %d 个应用）", OUTPUT_FILE, len(apps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
