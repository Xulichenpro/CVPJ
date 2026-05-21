#!/usr/bin/env bash
# 用法:
#   ./scripts/download_reid_datasets.sh all
#   ./scripts/download_reid_datasets.sh market duke occ
#
# 默认 DATA_ROOT: 若存在 /root/autodl-tmp 则用 /root/autodl-tmp/transreid_data，否则 <仓库>/data
# 覆盖: DATA_ROOT=/your/path ./scripts/download_reid_datasets.sh market
#
# 镜像说明: 对 HuggingFace 的 *\_HF_REL 路径自动尝试 ${HF_MIRROR_HOST:-hf-mirror.com} 与 huggingface.co。
# 追加 HTTP(S) 镜像（空格分隔，在默认 URL 之后尝试）:
#   MARKET_HTTP_EXTRA_URLS  DUKE_HTTP_EXTRA_URLS  MSMT17_HTTP_EXTRA_URLS
#   VERI_HTTP_EXTRA_URLS  VEHICLEID_HTTP_EXTRA_URLS
# Occluded 列表镜像: 环境变量 OCC_LIST_BASES（逗号分隔），默认 ghproxy + GitHub 直连。
# 可选 HF 直链路径: DUKE_HF_REL  VERI_HF_REL  VEHICLEID_HF_REL（datasets/.../resolve/main/xxx.zip）
# 本地 / 单 URL: MSMT17_ZIP / MSMT17_URL  VERI_ZIP / VERI_URL  VEHICLEID_ZIP / VEHICLEID_URL  DUKE_ZIP / DUKE_URL
#
# 依赖: curl 或 wget, unzip, python3；仅回退 Google Drive 时需: pip install gdown

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 脚本在仓库根目录时为 TransReID；在 scripts/ 子目录时为上一级
if [[ "$(basename "$SCRIPT_DIR")" == "scripts" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  REPO_ROOT="$SCRIPT_DIR"
fi
if [[ -z "${DATA_ROOT:-}" ]]; then
  if [[ -d /root/autodl-tmp ]]; then
    DATA_ROOT="/root/autodl-tmp/transreid_data"
  else
    DATA_ROOT="${REPO_ROOT}/data"
  fi
fi
MARKET_GDRIVE_ID="${MARKET_GDRIVE_ID:-0B8-rUzbwVRk0c054eEozWG9COHM}"
# HuggingFace datasets 相对路径（不含域名）→ 由 hf_expand_urls 展开为镜像 + 官方
MARKET_HF_REL="${MARKET_HF_REL:-datasets/aveocr/Market-1501-v15.09.15.zip/resolve/main/Market-1501-v15.09.15.zip}"
DUKE_URL_PRIMARY="${DUKE_URL_PRIMARY:-http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip}"
DUKE_GDRIVE_ID_FALLBACK="${DUKE_GDRIVE_ID_FALLBACK:-1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O}"
# MSMT17 整包（约 2.4GB，TransReID 需解压后含 train/test 与 list_*.txt；常见为 MSMT17_V1 目录）
MSMT17_HF_REL="${MSMT17_HF_REL:-datasets/xianpeijie/MSMT17_V1/resolve/main/MSMT17_V1.zip}"
# Occluded 划分列表：国内优先 ghproxy 再直连 GitHub
OCC_LIST_BASES_DEFAULT="https://ghproxy.net/https://raw.githubusercontent.com/lightas/Occluded-DukeMTMC-Dataset/master/Occluded_Duke,https://raw.githubusercontent.com/lightas/Occluded-DukeMTMC-Dataset/master/Occluded_Duke"
# VeRi / VehicleID 社区镜像（HF 路径）；若 404 可仅使用 VERI_URL / VEHICLEID_URL 或 *_HTTP_EXTRA_URLS
VERI_HF_REL="${VERI_HF_REL:-}"
VEHICLEID_HF_REL="${VEHICLEID_HF_REL:-}"
HF_MIRROR_HOST="${HF_MIRROR_HOST:-hf-mirror.com}"
HF_ORIGIN_HOST="${HF_ORIGIN_HOST:-huggingface.co}"
# 可选：HuggingFace 上的 Duke zip 相对路径（无则只用官方 HTTP + Drive 直链 + gdown）
DUKE_HF_REL="${DUKE_HF_REL:-}"
log() { printf '\n[%s] %s\n' "$(date -Iseconds)" "$*"; }
die() { log "ERROR: $*"; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# $1 = HuggingFace 路径（datasets/.../resolve/.../file），输出镜像站与官方站各一条 URL
hf_expand_urls() {
  local rel="$1"
  [[ -z "${rel// }" ]] && return 0
  rel="${rel#https://${HF_MIRROR_HOST}/}"
  rel="${rel#https://${HF_ORIGIN_HOST}/}"
  echo "https://${HF_MIRROR_HOST}/${rel}"
  echo "https://${HF_ORIGIN_HOST}/${rel}"
}

# 依次尝试 URL 下载为 zip 并通过 unzip -t；成功返回 0
download_zip_first_ok() {
  local dest="$1" url
  shift
  rm -f "$dest" 2>/dev/null || true
  for url in "$@"; do
    [[ -z "$url" ]] && continue
    if download_to "$url" "$dest" 2>/dev/null && unzip -t "$dest" &>/dev/null; then
      return 0
    fi
    rm -f "$dest"
  done
  return 1
}
ensure_gdown() {
  if have gdown; then return 0; fi
  log "安装 gdown (pip) 用于 Google Drive..."
  python3 -m pip install -q --user gdown || python3 -m pip install -q gdown
}
download_to() {
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  if have curl; then curl -fL --retry 3 --connect-timeout 30 --max-time 7200 -o "$dest" "$url"
  elif have wget; then wget -O "$dest" "$url"
  else die "需要 curl 或 wget"; fi
}
# 注意: gdown 对 -O xxx.zip 常再追加 .zip，得到 xxx.zip.zip；解压原路径会报 No zipfiles found
gdown_file() {
  local id="$1" dest="$2"
  ensure_gdown
  local url="https://drive.google.com/uc?id=${id}"
  local ddir tmp cand
  ddir="$(dirname "$dest")"
  mkdir -p "$ddir"
  tmp="$(mktemp "${ddir}/.gdown_XXXXXX")"
  rm -f "$dest" "${dest}.zip.zip" "${dest}.ZIP.ZIP" 2>/dev/null || true
  if ! python3 -m gdown --fuzzy "$url" -O "$tmp"; then
    rm -f "$tmp" "${tmp}.zip"
    return 1
  fi
  if [[ -f "${tmp}.zip" ]]; then mv -f "${tmp}.zip" "$dest"
  elif [[ -f "$tmp" ]]; then mv -f "$tmp" "$dest"
  else
    cand="$(find "$ddir" -maxdepth 1 -type f \( -iname '*.zip' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"
    if [[ -n "${cand:-}" && -s "$cand" ]]; then mv -f "$cand" "$dest"
    else
      log "gdown: 未找到有效输出文件"
      rm -f "$tmp" "${tmp}.zip"
      return 1
    fi
  fi
  rm -f "$tmp" "${tmp}.zip" 2>/dev/null || true
  if ! unzip -t "$dest" &>/dev/null; then
    log "gdown: 文件不是合法 zip（可能为 HTML 提示页或下载不完整）。前 120 字节:"
    head -c 120 "$dest" | cat -v || true
    return 1
  fi
  return 0
}

# MSMT17 解压后顶层目录可能是 MSMT17 或 MSMT17_V1
_msmt17_from_tmp() {
  local tmp="$1" dest="$2"
  rm -rf "$dest"
  if [[ -d "${tmp}/MSMT17" ]]; then mv "${tmp}/MSMT17" "$dest"
  elif [[ -d "${tmp}/MSMT17_V1" ]]; then mv "${tmp}/MSMT17_V1" "$dest"
  else
    local inner
    inner="$(find "$tmp" -mindepth 1 -maxdepth 1 -type d ! -name '.*' | head -1)"
    [[ -n "$inner" ]] || die "MSMT17 zip 异常"
    mv "$inner" "$dest"
  fi
}

unzip_quiet() { unzip -q -o "$1" -d "$2"; }
install_market1501() {
  local dest="${DATA_ROOT}/market1501"
  if [[ -d "${dest}/bounding_box_train" ]]; then log "market1501 已存在，跳过"; return 0; fi
  local tmp="${DATA_ROOT}/.tmp_market"
  rm -rf "$tmp" && mkdir -p "$tmp"
  local zipf="${tmp}/Market-1501.zip"
  if [[ -n "${MARKET_URL:-}" ]]; then
    log "下载 Market-1501 (MARKET_URL)..."
    download_to "$MARKET_URL" "$zipf"
  elif [[ "${MARKET_USE_GDRIVE:-0}" == "1" ]]; then
    log "下载 Market-1501 (Google Drive，国内可能很慢)..."
    gdown_file "$MARKET_GDRIVE_ID" "$zipf"
  else
    local -a urls=()
    if [[ -n "${MARKET_HF_URL:-}" ]]; then
      urls+=("$MARKET_HF_URL")
    else
      mapfile -t urls < <(hf_expand_urls "${MARKET_HF_REL}")
    fi
    if [[ -n "${MARKET_HTTP_EXTRA_URLS:-}" ]]; then read -r -a _mex <<< "${MARKET_HTTP_EXTRA_URLS:-}"; urls+=("${_mex[@]}"); fi
    local u ok=0
    for u in "${urls[@]}"; do
      [[ -z "$u" ]] && continue
      log "下载 Market-1501 (镜像/HTTP, ~153MB)..."
      if download_to "$u" "$zipf" 2>/dev/null; then ok=1; break; fi
      rm -f "$zipf"
    done
    if [[ "$ok" != "1" ]]; then
      log "HTTP 源失败，回退 Google Drive（若极慢可 Ctrl+C 后本机/网盘下载再: MARKET_URL=...）"
      gdown_file "$MARKET_GDRIVE_ID" "$zipf"
    fi
  fi
  unzip_quiet "$zipf" "$tmp"
  rm -rf "$dest"
  if [[ -d "${tmp}/Market-1501-v15.09.15" ]]; then mv "${tmp}/Market-1501-v15.09.15" "$dest"
  else
    local inner; inner="$(find "$tmp" -mindepth 1 -maxdepth 1 -type d ! -name '.*' | head -1)"
    [[ -n "$inner" ]] || die "Market zip 目录结构异常"
    mv "$inner" "$dest"
  fi
  rm -rf "$tmp"
  log "Market-1501 -> ${dest}"
}
get_duke_zip() {
  local tmp="${DATA_ROOT}/.tmp_duke_zip"
  mkdir -p "$tmp"
  local zipf="${tmp}/DukeMTMC-reID.zip"
  if [[ -s "$zipf" ]] && unzip -t "$zipf" &>/dev/null; then echo "$zipf"; return 0; fi
  rm -f "$zipf" "${zipf}.zip" "${tmp}/.gdown_"* 2>/dev/null || true
  if [[ -n "${DUKE_ZIP:-}" && -f "${DUKE_ZIP}" ]]; then
    log "复制 DukeMTMC-reID.zip (DUKE_ZIP)..."
    cp -f "${DUKE_ZIP}" "$zipf"
  elif [[ -n "${DUKE_URL:-}" ]]; then
    log "下载 DukeMTMC-reID (DUKE_URL)..."
    download_to "$DUKE_URL" "$zipf" || die "DUKE_URL 下载失败"
  else
    local -a duke_urls=()
    duke_urls+=("$DUKE_URL_PRIMARY")
    duke_urls+=("https://drive.google.com/uc?export=download&id=${DUKE_GDRIVE_ID_FALLBACK}&confirm=t")
    if [[ -n "${DUKE_HF_REL:-}" ]]; then
      mapfile -t _dh < <(hf_expand_urls "$DUKE_HF_REL")
      duke_urls+=("${_dh[@]}")
    fi
    if [[ -n "${DUKE_HTTP_EXTRA_URLS:-}" ]]; then read -r -a _dex <<< "${DUKE_HTTP_EXTRA_URLS:-}"; duke_urls+=("${_dex[@]}"); fi
    log "下载 DukeMTMC-reID (多镜像 HTTP / Drive 直链)..."
    if ! download_zip_first_ok "$zipf" "${duke_urls[@]}"; then
      rm -f "$zipf"
      log "HTTP 均失败，改用 Google Drive gdown（国内无代理常失败）..."
      if ! gdown_file "$DUKE_GDRIVE_ID_FALLBACK" "$zipf"; then
        die "Duke 下载失败。可将 DukeMTMC-reID.zip 上传到实例后: DUKE_ZIP=/path/DukeMTMC-reID.zip bash $0 duke ；或 DUKE_URL=... / DUKE_HF_REL=datasets/.../resolve/main/DukeMTMC-reID.zip / DUKE_HTTP_EXTRA_URLS='url ...'"
      fi
    fi
  fi
  unzip -t "$zipf" &>/dev/null || die "Duke zip 校验失败（文件损坏或非 zip）"
  echo "$zipf"
}
install_dukemtmcreid() {
  local dest="${DATA_ROOT}/dukemtmcreid"
  if [[ -d "${dest}/bounding_box_train" ]]; then log "dukemtmcreid 已存在，跳过"; return 0; fi
  local zipf; zipf="$(get_duke_zip)"
  local tmp="${DATA_ROOT}/.tmp_duke"
  rm -rf "$tmp" && mkdir -p "$tmp"
  unzip_quiet "$zipf" "$tmp"
  rm -rf "$dest"
  if [[ -d "${tmp}/DukeMTMC-reID" ]]; then mv "${tmp}/DukeMTMC-reID" "$dest"
  else
    local inner; inner="$(find "$tmp" -mindepth 1 -maxdepth 1 -type d ! -name '.*' | head -1)"
    [[ -n "$inner" ]] || die "Duke zip 目录结构异常"
    mv "$inner" "$dest"
  fi
  rm -rf "$tmp"
  log "DukeMTMC-reID -> ${dest}"
}
build_occluded_duke() {
  local dest="${DATA_ROOT}/Occluded_Duke"
  if [[ -d "${dest}/bounding_box_train" ]] && [[ -d "${dest}/query" ]]; then log "Occluded_Duke 已存在，跳过"; return 0; fi
  local zipf; zipf="$(get_duke_zip)"
  export OCC_LIST_BASES="${OCC_LIST_BASES:-${OCC_LIST_BASES_DEFAULT}}"
  python3 - <<'PY' "$zipf" "$DATA_ROOT"
import os, shutil, sys, urllib.request
from zipfile import ZipFile
duke_zip, data_root = sys.argv[1], sys.argv[2]
target = os.path.join(data_root, "Occluded_Duke")
origin = os.path.join(target, "DukeMTMC-reID")
bases = [b.strip() for b in os.environ.get("OCC_LIST_BASES", "").split(",") if b.strip()]
if not bases:
    raise SystemExit("OCC_LIST_BASES empty")

def fetch_list(name):
    dst = os.path.join(target, f"{name}.list")
    last = None
    for base in bases:
        url = f"{base.rstrip('/')}/{name}.list"
        try:
            os.makedirs(target, exist_ok=True)
            urllib.request.urlretrieve(url, dst)
            with open(dst, "r") as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            last = e
    raise SystemExit("fetch %s.list failed: %s" % (name, last))

def gen(split, folder_name):
    imgs = fetch_list(split)
    src_split = os.path.join(origin, folder_name)
    dst_split = os.path.join(target, folder_name)
    os.makedirs(dst_split, exist_ok=True)
    for img in imgs:
        p1 = os.path.join(src_split, img)
        src = p1 if os.path.isfile(p1) else os.path.join(origin, "bounding_box_test", img)
        shutil.copy2(src, os.path.join(dst_split, img))
    print(folder_name, len(imgs))

def main():
    if os.path.isdir(target):
        shutil.rmtree(target)
    os.makedirs(target, exist_ok=True)
    with ZipFile(duke_zip) as z:
        z.extractall(path=target)
    if not os.path.isdir(origin):
        raise SystemExit("missing %s" % origin)
    gen("train", "bounding_box_train")
    gen("gallery", "bounding_box_test")
    gen("query", "query")
    shutil.rmtree(origin)
    for x in ("train.list", "gallery.list", "query.list"):
        try:
            os.remove(os.path.join(target, x))
        except OSError:
            pass
    print("Occluded_Duke ->", target)

main()
PY
  log "Occluded_Duke -> ${dest}"
}
install_msmt17() {
  local dest="${DATA_ROOT}/MSMT17"
  if [[ -d "${dest}/train" && -d "${dest}/test" && -f "${dest}/list_train.txt" ]]; then
    log "MSMT17 已存在，跳过"; return 0
  fi
  if [[ -n "${MSMT17_ZIP:-}" && -f "${MSMT17_ZIP}" ]]; then
    local tmp="${DATA_ROOT}/.tmp_msmt17"
    rm -rf "$tmp" && mkdir -p "$tmp"
    unzip_quiet "${MSMT17_ZIP}" "$tmp"
    _msmt17_from_tmp "$tmp" "$dest"
    rm -rf "$tmp"
    log "MSMT17 -> ${dest}"; return 0
  fi
  if [[ -n "${MSMT17_URL:-}" ]]; then
    local tmp="${DATA_ROOT}/.tmp_msmt17_dl" zipf ex
    mkdir -p "$tmp"
    zipf="${tmp}/MSMT17_dl.zip"
    download_to "$MSMT17_URL" "$zipf"
    ex="${DATA_ROOT}/.tmp_msmt17"
    rm -rf "$ex" && mkdir -p "$ex"
    unzip_quiet "$zipf" "$ex"
    _msmt17_from_tmp "$ex" "$dest"
    rm -rf "$tmp" "$ex"
    log "MSMT17 -> ${dest}"; return 0
  fi
  local -a urls=()
  mapfile -t urls < <(hf_expand_urls "${MSMT17_HF_REL:-}")
  if [[ -n "${MSMT17_HTTP_EXTRA_URLS:-}" ]]; then read -r -a _mx <<< "${MSMT17_HTTP_EXTRA_URLS:-}"; urls+=("${_mx[@]}"); fi
  if [[ ${#urls[@]} -eq 0 ]]; then
    log "MSMT17 跳过（设置 MSMT17_ZIP / MSMT17_URL，或配置 MSMT17_HF_REL / MSMT17_HTTP_EXTRA_URLS）。"
    return 0
  fi
  log "MSMT17: 尝试 HuggingFace 镜像 (约 2.4GB，耗时较长)..."
  local dld="${DATA_ROOT}/.tmp_msmt17_hf" zipf="${DATA_ROOT}/.tmp_msmt17_hf/MSMT17_V1.zip" ex="${DATA_ROOT}/.tmp_msmt17"
  mkdir -p "$dld"
  rm -f "$zipf"
  if ! download_zip_first_ok "$zipf" "${urls[@]}"; then
    rm -rf "$dld"
    log "MSMT17: 镜像下载失败，跳过。"
    return 0
  fi
  rm -rf "$ex" && mkdir -p "$ex"
  unzip_quiet "$zipf" "$ex"
  _msmt17_from_tmp "$ex" "$dest"
  rm -rf "$ex" "$dld"
  log "MSMT17 -> ${dest}"
}
install_veri() {
  local dest="${DATA_ROOT}/VeRi"
  if [[ -d "${dest}/image_train" ]]; then log "VeRi 已存在，跳过"; return 0; fi
  local src="${VERI_ZIP:-${VERI_URL:-}}"
  local -a urls=()
  if [[ -n "${VERI_HF_REL:-}" ]]; then mapfile -t urls < <(hf_expand_urls "$VERI_HF_REL"); fi
  if [[ -n "${VERI_HTTP_EXTRA_URLS:-}" ]]; then read -r -a _vx <<< "${VERI_HTTP_EXTRA_URLS:-}"; urls+=("${_vx[@]}"); fi

  local tmp="${DATA_ROOT}/.tmp_veri" zipf="${DATA_ROOT}/.tmp_veri/arch.zip"
  rm -rf "$tmp" && mkdir -p "$tmp"

  if [[ -n "$src" ]]; then
    [[ -f "$src" ]] && cp -f "$src" "$zipf" || download_to "$src" "$zipf"
  elif [[ ${#urls[@]} -gt 0 ]]; then
    log "VeRi: 尝试镜像下载 (zip)..."
    download_zip_first_ok "$zipf" "${urls[@]}" || die "VeRi 镜像均失败（请设 VERI_ZIP / VERI_URL 或检查 VERI_HF_REL）"
  else
    log "VeRi 跳过（设置 VERI_ZIP / VERI_URL 或 VERI_HF_REL / VERI_HTTP_EXTRA_URLS）"; return 0
  fi
  unzip_quiet "$zipf" "$tmp"
  rm -rf "$dest"
  if [[ -d "${tmp}/VeRi" ]]; then mv "${tmp}/VeRi" "$dest"
  else
    local inner
    inner="$(find "$tmp" -mindepth 1 -maxdepth 1 -type d ! -name '.*' | head -1)"
    [[ -n "$inner" ]] || die "VeRi 压缩包结构异常"; mv "$inner" "$dest"
  fi
  rm -rf "$tmp"
  log "VeRi -> ${dest}"
}
install_vehicleid() {
  local dest="${DATA_ROOT}/VehicleID_V1.0"
  if [[ -d "${dest}/image" ]]; then log "VehicleID_V1.0 已存在，跳过"; return 0; fi
  local src="${VEHICLEID_ZIP:-${VEHICLEID_URL:-}}"
  local -a urls=()
  if [[ -n "${VEHICLEID_HF_REL:-}" ]]; then mapfile -t urls < <(hf_expand_urls "$VEHICLEID_HF_REL"); fi
  if [[ -n "${VEHICLEID_HTTP_EXTRA_URLS:-}" ]]; then read -r -a _vid <<< "${VEHICLEID_HTTP_EXTRA_URLS:-}"; urls+=("${_vid[@]}"); fi

  local tmp="${DATA_ROOT}/.tmp_vid" zipf="${DATA_ROOT}/.tmp_vid/arch.zip"
  rm -rf "$tmp" && mkdir -p "$tmp"

  if [[ -n "$src" ]]; then
    [[ -f "$src" ]] && cp -f "$src" "$zipf" || download_to "$src" "$zipf"
  elif [[ ${#urls[@]} -gt 0 ]]; then
    log "VehicleID: 尝试镜像下载 (zip)..."
    download_zip_first_ok "$zipf" "${urls[@]}" || die "VehicleID 镜像均失败（请设 VEHICLEID_ZIP / URL 或 VEHICLEID_HF_REL）"
  else
    log "VehicleID 跳过（设置 VEHICLEID_ZIP / VEHICLEID_URL 或 VEHICLEID_HF_REL / VEHICLEID_HTTP_EXTRA_URLS）"; return 0
  fi
  unzip_quiet "$zipf" "$tmp"
  rm -rf "$dest"
  if [[ -d "${tmp}/VehicleID_V1.0" ]]; then mv "${tmp}/VehicleID_V1.0" "$dest"
  else
    local inner
    inner="$(find "$tmp" -mindepth 1 -maxdepth 1 -type d ! -name '.*' | head -1)"
    [[ -n "$inner" ]] || die "VehicleID 压缩包结构异常"; mv "$inner" "$dest"
  fi
  rm -rf "$tmp"
  log "VehicleID_V1.0 -> ${dest}"
}
usage() {
  cat <<EOF
用法: $0 [market|duke|occ|msmt17|veri|vehicleid|all ...]
当前 DATA_ROOT=${DATA_ROOT}
镜像: Market/Duke/MSMT17 默认多源；VeRi/VehicleID 需配置 VERI_HF_REL 等或 EXTRA URL。
EOF
}
mkdir -p "$DATA_ROOT"
log "REPO_ROOT=${REPO_ROOT}  DATA_ROOT=${DATA_ROOT}"
[[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]] && { usage; exit 0; }
parts=("$@")
[[ "${parts[0]}" == "all" ]] && parts=(market duke occ msmt17 veri vehicleid)
for p in "${parts[@]}"; do
  case "$p" in
    market)    install_market1501 ;;
    duke)      install_dukemtmcreid ;;
    occ)       build_occluded_duke ;;
    msmt17)    install_msmt17 ;;
    veri)      install_veri ;;
    vehicleid) install_vehicleid ;;
    *) die "未知参数: $p" ;;
  esac
done
log "完成。训练时在 yml 里设置 DATASETS.ROOT_DIR 为: ${DATA_ROOT}"