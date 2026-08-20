import re
import shutil
import tarfile
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs

import requests
import numpy as np
import xarray as xr
from PIL import Image
from scipy.interpolate import griddata

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# 1. 配置
# ============================================================

# ScienceDB 导出的下载链接 txt
INPUT_TXT = Path(
    r"C:\Users\Administrator\Downloads\696756084735475712.txt"
)

# ZIP / TAR 临时下载目录
DOWNLOAD_DIR = Path(
    r"D:\dataset\zip"
)

# NC 临时解压目录
# 可以和 DOWNLOAD_DIR 是同一个根目录
TEMP_ROOT = Path(
    r"D:\dataset\zip"
)

# 最终只保留 UTC 2020 华北 PM2.5
OUTPUT_DIR = Path(
    r"D:\dataset\utc_2020_huabei"
)

DOWNLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEMP_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 时间配置
# ============================================================

TARGET_UTC_YEAR = 2020

# 为了得到完整 UTC 2020：
#
# UTC 2020-01-01 00:00
# 对应北京时间 2020-01-01 08:00
#
# UTC 2020-12-31 23:00
# 对应北京时间 2021-01-01 07:00
#
# 因此必须下载北京时间：
#
# 2020-01-01
#     ↓
# 2021-01-01
#
DOWNLOAD_START_DATE = datetime(
    2020, 1, 1
)

DOWNLOAD_END_DATE = datetime(
    2021, 1, 1
)


# ============================================================
# 3. 华北区域配置
# ============================================================

TARGET_LON_MIN = 112.6
TARGET_LON_MAX = 123.0

TARGET_LAT_MIN = 33.5
TARGET_LAT_MAX = 42.0

GRID_SIZE = 64


# 建立 64 × 64 目标经纬度网格
target_lon = np.linspace(
    TARGET_LON_MIN,
    TARGET_LON_MAX,
    GRID_SIZE
)

target_lat = np.linspace(
    TARGET_LAT_MIN,
    TARGET_LAT_MAX,
    GRID_SIZE
)

target_lon2d, target_lat2d = np.meshgrid(
    target_lon,
    target_lat
)


# ============================================================
# 4. 创建网络 Session
# ============================================================

def create_session():

    session = requests.Session()

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504
        ],
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(
        max_retries=retry
    )

    session.mount(
        "http://",
        adapter
    )

    session.mount(
        "https://",
        adapter
    )

    session.headers.update({
        "User-Agent":
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36"
    })

    return session


SESSION = create_session()


# ============================================================
# 5. 生成需要下载的日期
# ============================================================

def generate_required_dates():

    result = []

    current = DOWNLOAD_START_DATE

    while current <= DOWNLOAD_END_DATE:

        result.append(
            current.strftime("%Y%m%d")
        )

        current += timedelta(
            days=1
        )

    return result


# ============================================================
# 6. 从 URL 中提取压缩文件信息
# ============================================================

def parse_archive_from_url(url):
    """
    从 ScienceDB URL 中解析：

        日期
        文件名
        扩展格式

    支持：

        .zip
        .tar
        .tar.gz
        .tgz
    """

    parsed = urlparse(
        url
    )

    params = parse_qs(
        parsed.query
    )

    filenames = params.get(
        "fileName",
        []
    )

    filename = (
        filenames[0]
        if filenames
        else ""
    )

    # --------------------------------------------------------
    # 注意 tar.gz 必须放在 tar 前面
    # --------------------------------------------------------

    pattern = re.compile(
        r"CN-Reanalysis"
        r"(\d{8})"
        r"\."
        r"(tar\.gz|tgz|zip|tar)"
        r"$",
        re.IGNORECASE
    )

    match = pattern.search(
        filename
    )

    # --------------------------------------------------------
    # 如果 fileName 没取到，再从 path 中取
    # --------------------------------------------------------

    if match is None:

        paths = params.get(
            "path",
            []
        )

        if paths:

            filename = Path(
                paths[0]
            ).name

            match = pattern.search(
                filename
            )

    # --------------------------------------------------------
    # 最后直接搜索整个 URL
    # --------------------------------------------------------

    if match is None:

        general_pattern = re.compile(
            r"CN-Reanalysis"
            r"(\d{8})"
            r"\."
            r"(tar\.gz|tgz|zip|tar)",
            re.IGNORECASE
        )

        match = general_pattern.search(
            url
        )

        if match:

            date_str = match.group(1)
            extension = match.group(2).lower()

            filename = (
                f"CN-Reanalysis"
                f"{date_str}."
                f"{extension}"
            )

            return (
                date_str,
                filename,
                extension
            )

    if match is None:

        return None

    date_str = match.group(
        1
    )

    extension = match.group(
        2
    ).lower()

    try:

        datetime.strptime(
            date_str,
            "%Y%m%d"
        )

    except ValueError:

        return None

    return (
        date_str,
        filename,
        extension
    )


# ============================================================
# 7. 读取 ScienceDB 下载链接
# ============================================================

def load_download_urls():

    if not INPUT_TXT.exists():

        raise FileNotFoundError(
            f"找不到链接文件：{INPUT_TXT}"
        )

    print("=" * 80)
    print("读取 ScienceDB 下载链接")
    print("=" * 80)

    print(
        f"链接文件：{INPUT_TXT}"
    )

    urls = None

    # --------------------------------------------------------
    # 尝试不同编码
    # --------------------------------------------------------

    for encoding in [
        "utf-8",
        "utf-8-sig",
        "gbk"
    ]:

        try:

            with open(
                INPUT_TXT,
                "r",
                encoding=encoding
            ) as f:

                urls = [
                    line.strip()
                    for line in f
                    if line.strip()
                ]

            break

        except UnicodeDecodeError:
            continue

    if urls is None:

        raise RuntimeError(
            "无法读取下载链接 txt。"
        )

    print(
        f"TXT 中总链接数："
        f"{len(urls)}"
    )

    url_map = {}

    extension_count = {}

    duplicate_dates = {}

    # ========================================================
    # 解析每一个真实 URL
    # ========================================================

    for url in urls:

        if not url.startswith(
            ("http://", "https://")
        ):
            continue

        result = parse_archive_from_url(
            url
        )

        if result is None:
            continue

        date_str, filename, extension = result

        # ----------------------------------------------------
        # 同一个日期如果存在多个链接
        # 记录下来，但默认使用第一个
        # ----------------------------------------------------

        if date_str in url_map:

            duplicate_dates.setdefault(
                date_str,
                []
            ).append(
                url
            )

            continue

        url_map[
            date_str
        ] = url

        extension_count[
            extension
        ] = (
            extension_count.get(
                extension,
                0
            )
            + 1
        )

    print(
        f"成功解析日压缩文件："
        f"{len(url_map)}"
    )

    # ========================================================
    # 年份统计
    # ========================================================

    year_count = {}

    for date_str in url_map:

        year = date_str[:4]

        year_count[
            year
        ] = (
            year_count.get(
                year,
                0
            )
            + 1
        )

    print()
    print(
        "download.txt 年份统计："
    )

    for year in sorted(
        year_count
    ):

        print(
            f"  {year}: "
            f"{year_count[year]} 个日文件"
        )

    # ========================================================
    # 文件格式统计
    # ========================================================

    print()
    print(
        "压缩文件格式统计："
    )

    for extension in sorted(
        extension_count
    ):

        print(
            f"  .{extension}: "
            f"{extension_count[extension]} 个"
        )

    if duplicate_dates:

        print()
        print(
            f"发现 {len(duplicate_dates)} 个重复日期，"
            f"程序默认使用每个日期遇到的第一个链接。"
        )

    return url_map


# ============================================================
# 8. 检查需要的 367 个链接是否完整
# ============================================================

def check_required_urls(
        url_map
):

    required_dates = generate_required_dates()

    missing = [
        date_str
        for date_str in required_dates
        if date_str not in url_map
    ]

    print()
    print("=" * 80)
    print("下载链接完整性检查")
    print("=" * 80)

    print(
        f"理论需要："
        f"{len(required_dates)} 个日文件"
    )

    print(
        f"已有链接："
        f"{len(required_dates) - len(missing)}"
    )

    print(
        f"缺失链接："
        f"{len(missing)}"
    )

    if missing:

        print()
        print(
            "缺少以下日期："
        )

        for date_str in missing[:100]:

            print(
                f"  {date_str}"
            )

        if len(missing) > 100:

            print(
                "  ..."
            )

        raise RuntimeError(
            "\nScienceDB 链接文件中缺少完整的 "
            "UTC 2020 所需数据。\n"
            "程序停止，不会猜测 fileId。"
        )

    print()
    print(
        "367 个日期链接全部存在，可以开始下载。"
    )


# ============================================================
# 9. 从 URL 获取真实文件名
# ============================================================

def get_filename_from_url(
        url,
        date_str
):

    result = parse_archive_from_url(
        url
    )

    if result is None:

        raise RuntimeError(
            f"无法解析 {date_str} 的压缩文件名。"
        )

    _, filename, _ = result

    return filename


# ============================================================
# 10. 判断是否为有效压缩包
# ============================================================

def is_valid_archive(
        file_path
):

    file_path = Path(
        file_path
    )

    # ZIP
    try:

        if zipfile.is_zipfile(
            file_path
        ):

            return True

    except Exception:
        pass

    # TAR / TAR.GZ / TGZ
    try:

        if tarfile.is_tarfile(
            file_path
        ):

            return True

    except Exception:
        pass

    return False


# ============================================================
# 11. 解压文件
# ============================================================

def extract_archive(
        archive_path,
        extract_dir
):

    archive_path = Path(
        archive_path
    )

    extract_dir = Path(
        extract_dir
    )

    extract_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"正在解压："
        f"{archive_path.name}"
    )

    # ========================================================
    # ZIP
    # ========================================================

    if zipfile.is_zipfile(
        archive_path
    ):

        print(
            "压缩格式：ZIP"
        )

        with zipfile.ZipFile(
            archive_path,
            "r"
        ) as archive:

            archive.extractall(
                extract_dir
            )

        print(
            "ZIP 解压完成。"
        )

        return

    # ========================================================
    # TAR / TAR.GZ / TGZ
    # ========================================================

    if tarfile.is_tarfile(
        archive_path
    ):

        print(
            "压缩格式：TAR"
        )

        with tarfile.open(
            archive_path,
            "r:*"
        ) as archive:

            archive.extractall(
                extract_dir
            )

        print(
            "TAR 解压完成。"
        )

        return

    raise RuntimeError(
        f"无法识别压缩文件："
        f"{archive_path}"
    )


# ============================================================
# 12. 下载某一天
# ============================================================

def download_one_day(
        date_str,
        url
):

    filename = get_filename_from_url(
        url,
        date_str
    )

    archive_path = (
        DOWNLOAD_DIR
        / filename
    )

    part_path = (
        DOWNLOAD_DIR
        / f"{filename}.part"
    )

    print()
    print("=" * 80)
    print(
        f"开始下载：{date_str}"
    )
    print("=" * 80)

    parsed = urlparse(
        url
    )

    params = parse_qs(
        parsed.query
    )

    file_ids = params.get(
        "fileId",
        []
    )

    file_id = (
        file_ids[0]
        if file_ids
        else "未知"
    )

    print(
        f"文件：{filename}"
    )

    print(
        f"fileId：{file_id}"
    )

    # --------------------------------------------------------
    # 删除以前失败留下的文件
    # --------------------------------------------------------

    if part_path.exists():

        part_path.unlink()

    if archive_path.exists():

        archive_path.unlink()

    try:

        with SESSION.get(
            url,
            stream=True,
            timeout=(30, 600)
        ) as response:

            print(
                f"HTTP状态："
                f"{response.status_code}"
            )

            response.raise_for_status()

            total_size = int(
                response.headers.get(
                    "Content-Length",
                    0
                )
            )

            if total_size > 0:

                print(
                    f"文件大小："
                    f"{total_size / 1024 / 1024:.2f} MB"
                )

            downloaded = 0
            last_percent = -1

            with open(
                part_path,
                "wb"
            ) as f:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if not chunk:
                        continue

                    f.write(
                        chunk
                    )

                    downloaded += len(
                        chunk
                    )

                    if total_size > 0:

                        percent = int(
                            downloaded
                            / total_size
                            * 100
                        )

                        if percent != last_percent:

                            print(
                                f"\r下载："
                                f"{percent:3d}%  "
                                f"{downloaded / 1024 / 1024:.1f}"
                                f"/"
                                f"{total_size / 1024 / 1024:.1f}"
                                f" MB",
                                end=""
                            )

                            last_percent = percent

            print()

        # ====================================================
        # 文件检查
        # ====================================================

        if not part_path.exists():

            print(
                "下载失败：临时文件不存在。"
            )

            return None

        if part_path.stat().st_size == 0:

            print(
                "下载失败：文件大小为0。"
            )

            part_path.unlink(
                missing_ok=True
            )

            return None

        # ====================================================
        # ZIP / TAR 有效性检查
        # ====================================================

        if not is_valid_archive(
            part_path
        ):

            print()
            print(
                "错误：服务器返回的不是有效 "
                "ZIP/TAR/TAR.GZ/TGZ 压缩包。"
            )

            try:

                with open(
                    part_path,
                    "rb"
                ) as f:

                    head = f.read(
                        500
                    )

                print(
                    "文件前500字节："
                )

                print(
                    head
                )

            except Exception:
                pass

            part_path.unlink(
                missing_ok=True
            )

            return None

        # ====================================================
        # 校验成功后正式改名
        # ====================================================

        part_path.rename(
            archive_path
        )

        print(
            f"下载成功："
            f"{archive_path.name}"
        )

        return archive_path

    except KeyboardInterrupt:

        print()
        print(
            "下载被用户中断。"
        )

        if part_path.exists():

            part_path.unlink()

        raise

    except Exception as e:

        print(
            f"下载失败：{repr(e)}"
        )

        if part_path.exists():

            part_path.unlink()

        if archive_path.exists():

            archive_path.unlink()

        return None


# ============================================================
# 13. 从 NC 文件名提取北京时间
# ============================================================

def extract_beijing_datetime(
        filename
):

    filename = Path(
        filename
    ).name

    # 例如：
    #
    # CN-Reanalysis2020101800.nc
    #
    # 提取：
    #
    # 2020101800

    match = re.search(
        r"(\d{10})",
        filename
    )

    if match is None:

        return None

    try:

        return datetime.strptime(
            match.group(1),
            "%Y%m%d%H"
        )

    except ValueError:

        return None


# ============================================================
# 14. 计算某个北京时间日期应该生成哪些 UTC 2020 文件
# ============================================================

def expected_utc_files_for_beijing_day(
        date_str
):
    """
    用于断点续传。

    例如北京时间 2020-01-02：
        00~23
    转为 UTC：
        2020-01-01 16
            ~
        2020-01-02 15

    返回属于 UTC 2020 的文件名。
    """

    day = datetime.strptime(
        date_str,
        "%Y%m%d"
    )

    result = []

    for hour in range(
        24
    ):

        bj_time = (
            day
            + timedelta(
                hours=hour
            )
        )

        utc_time = (
            bj_time
            - timedelta(
                hours=8
            )
        )

        if (
            utc_time.year
            == TARGET_UTC_YEAR
        ):

            result.append(
                utc_time.strftime(
                    "%Y%m%d%H"
                )
            )

    return result


# ============================================================
# 15. 判断一天是否已经完整处理
# ============================================================

def is_day_already_complete(
        date_str
):

    expected_names = (
        expected_utc_files_for_beijing_day(
            date_str
        )
    )

    # 如果当天没有需要保留的 UTC 2020 数据
    if not expected_names:

        return True

    for name in expected_names:

        output_path = (
            OUTPUT_DIR
            / f"{name}.png"
        )

        if not output_path.exists():

            return False

    return True


# ============================================================
# 16. 处理一个 NC
# ============================================================

def process_single_nc(
        nc_file
):

    nc_file = Path(
        nc_file
    )

    bj_time = extract_beijing_datetime(
        nc_file.name
    )

    if bj_time is None:

        print(
            f"无法识别时间："
            f"{nc_file.name}"
        )

        return False

    # ========================================================
    # 北京时间 -> UTC
    # ========================================================

    utc_time = (
        bj_time
        - timedelta(
            hours=8
        )
    )

    # ========================================================
    # 只保留 UTC 2020
    # ========================================================

    if utc_time.year != TARGET_UTC_YEAR:

        print(
            f"跳过："
            f"{bj_time.strftime('%Y%m%d%H')} BJ"
            f" -> "
            f"{utc_time.strftime('%Y%m%d%H')} UTC"
        )

        return True

    utc_name = utc_time.strftime(
        "%Y%m%d%H"
    )

    output_path = (
        OUTPUT_DIR
        / f"{utc_name}.png"
    )

    # --------------------------------------------------------
    # 已经存在则跳过
    # --------------------------------------------------------

    if output_path.exists():

        print(
            f"已存在："
            f"{output_path.name}"
        )

        return True

    print(
        f"{bj_time.strftime('%Y%m%d%H')} BJ"
        f" -> "
        f"{utc_name} UTC"
    )

    try:

        # ====================================================
        # 读取 NC
        # ====================================================

        with xr.open_dataset(
            nc_file
        ) as data:

            required = [
                "pm25",
                "lon2d",
                "lat2d"
            ]

            for variable in required:

                if variable not in data.variables:

                    print(
                        f"缺少变量："
                        f"{variable}"
                    )

                    print(
                        "实际变量："
                        f"{list(data.variables)}"
                    )

                    return False

            pm25 = (
                data["pm25"]
                .squeeze()
                .values
            )

            lon = (
                data["lon2d"]
                .squeeze()
                .values
            )

            lat = (
                data["lat2d"]
                .squeeze()
                .values
            )

        # ====================================================
        # 数据类型
        # ====================================================

        pm25 = np.asarray(
            pm25,
            dtype=np.float64
        )

        lon = np.asarray(
            lon,
            dtype=np.float64
        )

        lat = np.asarray(
            lat,
            dtype=np.float64
        )

        if pm25.ndim != 2:

            print(
                f"PM2.5维度错误："
                f"{pm25.shape}"
            )

            return False

        if lon.ndim != 2:

            print(
                f"lon2d维度错误："
                f"{lon.shape}"
            )

            return False

        if lat.ndim != 2:

            print(
                f"lat2d维度错误："
                f"{lat.shape}"
            )

            return False

        # ====================================================
        # 与你的原始代码保持一致：
        #
        # flatten
        # ->
        # griddata
        # ->
        # nearest
        # ====================================================

        pm_flat = pm25.flatten()
        lon_flat = lon.flatten()
        lat_flat = lat.flatten()

        valid = (
            np.isfinite(pm_flat)
            &
            np.isfinite(lon_flat)
            &
            np.isfinite(lat_flat)
        )

        pm_flat = pm_flat[
            valid
        ]

        lon_flat = lon_flat[
            valid
        ]

        lat_flat = lat_flat[
            valid
        ]

        if len(pm_flat) == 0:

            print(
                "没有有效 PM2.5 数据。"
            )

            return False

        # ====================================================
        # 64×64 最近邻空间重采样
        # ====================================================

        grid_values = griddata(
            (
                lon_flat,
                lat_flat
            ),
            pm_flat,
            (
                target_lon2d,
                target_lat2d
            ),
            method="nearest"
        )

        # ====================================================
        # 纬度翻转
        #
        # target_lat：
        # 33.5 -> 42
        #
        # flip 后：
        #
        # row=0 北
        # row=63 南
        # ====================================================

        grid_values = np.flipud(
            grid_values
        )

        # ====================================================
        # 无效值处理
        # ====================================================

        if np.isnan(
            grid_values
        ).any():

            nan_count = int(
                np.isnan(
                    grid_values
                ).sum()
            )

            print(
                f"警告：存在 "
                f"{nan_count} 个 NaN"
            )

            grid_values = np.nan_to_num(
                grid_values,
                nan=0.0
            )

        # ====================================================
        # uint16 范围
        # ====================================================

        grid_values = np.clip(
            grid_values,
            0,
            65535
        )

        image_array = grid_values.astype(
            np.uint16
        )

        # ====================================================
        # 保存 16 位 PNG
        # ====================================================

        image = Image.fromarray(
            image_array
        )

        image.save(
            output_path,
            format="PNG"
        )

        if not output_path.exists():

            print(
                f"保存失败："
                f"{output_path}"
            )

            return False

        print(
            f"保存："
            f"{output_path.name}  "
            f"范围="
            f"{image_array.min()}~"
            f"{image_array.max()}"
        )

        return True

    except Exception as e:

        print(
            f"处理失败："
            f"{nc_file.name}"
        )

        print(
            repr(e)
        )

        return False


# ============================================================
# 17. 处理某一天
# ============================================================

def process_one_day(
        date_str,
        url
):

    print()
    print()
    print("#" * 80)
    print(
        f"处理日期："
        f"{date_str}"
    )
    print("#" * 80)

    # ========================================================
    # 如果当天已经处理完整：
    # 直接跳过下载
    # ========================================================

    if is_day_already_complete(
        date_str
    ):

        expected_names = (
            expected_utc_files_for_beijing_day(
                date_str
            )
        )

        print(
            f"{date_str} 对应的 UTC 2020 "
            f"数据已经全部存在，跳过下载。"
        )

        print(
            f"对应需要的影像数量："
            f"{len(expected_names)}"
        )

        return True

    # ========================================================
    # 下载
    # ========================================================

    archive_path = download_one_day(
        date_str,
        url
    )

    if archive_path is None:

        return False

    # ========================================================
    # 临时解压目录
    # ========================================================

    extract_dir = (
        TEMP_ROOT
        / f"temp_{date_str}"
    )

    if extract_dir.exists():

        shutil.rmtree(
            extract_dir
        )

    extract_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        # ====================================================
        # 解压
        # ========================================================

        extract_archive(
            archive_path,
            extract_dir
        )

        # ====================================================
        # 搜索 NC
        # ========================================================

        nc_files = list(
            extract_dir.rglob(
                "*.nc"
            )
        )

        nc_files.sort(
            key=lambda p:
                extract_beijing_datetime(
                    p.name
                )
                or datetime.max
        )

        print(
            f"NC数量："
            f"{len(nc_files)}"
        )

        if not nc_files:

            print(
                "压缩包中没有找到 NC 文件。"
            )

            return False

        # ====================================================
        # 检查真实日期
        #
        # 防止再次出现：
        #
        # 请求 20200101
        # 实际下载 20190111
        # ========================================================

        actual_dates = set()

        for nc_file in nc_files:

            dt = extract_beijing_datetime(
                nc_file.name
            )

            if dt is not None:

                actual_dates.add(
                    dt.strftime(
                        "%Y%m%d"
                    )
                )

        print(
            f"压缩包真实数据日期："
            f"{sorted(actual_dates)}"
        )

        if actual_dates != {
            date_str
        }:

            print()
            print(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )

            print(
                "错误：下载文件和请求日期不一致！"
            )

            print(
                f"请求日期："
                f"{date_str}"
            )

            print(
                f"实际日期："
                f"{sorted(actual_dates)}"
            )

            print(
                "该压缩包不会被保存。"
            )

            print(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )

            return False

        # ====================================================
        # 正常每天应为 24 个 NC
        # ========================================================

        if len(nc_files) != 24:

            print()
            print(
                f"警告：{date_str} "
                f"不是24个 NC，"
                f"实际为 {len(nc_files)} 个。"
            )

        # ====================================================
        # 逐小时处理
        # ========================================================

        success_count = 0

        failed_files = []

        for index, nc_file in enumerate(
            nc_files,
            start=1
        ):

            print(
                f"[{index:02d}/"
                f"{len(nc_files):02d}] ",
                end=""
            )

            success = process_single_nc(
                nc_file
            )

            if success:

                success_count += 1

            else:

                failed_files.append(
                    nc_file.name
                )

        print()
        print(
            f"{date_str}："
            f"{success_count}/"
            f"{len(nc_files)} 成功"
        )

        if failed_files:

            print(
                "失败 NC："
            )

            for filename in failed_files:

                print(
                    f"  {filename}"
                )

        # ====================================================
        # 再检查该日期真正需要保存的 UTC 文件
        # ========================================================

        expected_names = (
            expected_utc_files_for_beijing_day(
                date_str
            )
        )

        missing_outputs = []

        for name in expected_names:

            output_path = (
                OUTPUT_DIR
                / f"{name}.png"
            )

            if not output_path.exists():

                missing_outputs.append(
                    name
                )

        if missing_outputs:

            print()
            print(
                f"{date_str} 处理后仍缺少 "
                f"{len(missing_outputs)} 个 UTC 文件："
            )

            for name in missing_outputs:

                print(
                    f"  {name}"
                )

            return False

        return True

    finally:

        # ====================================================
        # 删除解压后的原始 NC
        # ========================================================

        if extract_dir.exists():

            try:

                shutil.rmtree(
                    extract_dir
                )

                print(
                    f"已删除临时解压目录："
                    f"{extract_dir.name}"
                )

            except Exception as e:

                print(
                    f"删除临时目录失败："
                    f"{e}"
                )

        # ====================================================
        # 删除当天压缩包
        #
        # 无论 ZIP / TAR 都删除
        # ========================================================

        if archive_path.exists():

            try:

                archive_path.unlink()

                print(
                    f"已删除原始压缩包："
                    f"{archive_path.name}"
                )

            except Exception as e:

                print(
                    f"删除压缩包失败："
                    f"{e}"
                )


# ============================================================
# 18. 最终 UTC 2020 完整性检查
# ============================================================

def check_final_dataset():

    print()
    print("=" * 80)
    print(
        "检查 UTC 2020 PM2.5 数据完整性"
    )
    print("=" * 80)

    expected = []

    current = datetime(
        2020,
        1,
        1,
        0
    )

    end = datetime(
        2020,
        12,
        31,
        23
    )

    while current <= end:

        expected.append(
            current.strftime(
                "%Y%m%d%H"
            )
        )

        current += timedelta(
            hours=1
        )

    # --------------------------------------------------------
    # 只统计符合 YYYYMMDDHH.png 的文件
    # --------------------------------------------------------

    existing = set()

    for png_path in OUTPUT_DIR.glob(
        "*.png"
    ):

        if re.fullmatch(
            r"\d{10}",
            png_path.stem
        ):

            existing.add(
                png_path.stem
            )

    expected_set = set(
        expected
    )

    missing = [
        timestamp
        for timestamp in expected
        if timestamp not in existing
    ]

    extra = sorted(
        existing
        - expected_set
    )

    print(
        f"理论 UTC 2020 数量："
        f"{len(expected)}"
    )

    print(
        f"当前符合时间命名的 PNG："
        f"{len(existing)}"
    )

    print(
        f"缺失："
        f"{len(missing)}"
    )

    if extra:

        print(
            f"UTC 2020 范围外额外 PNG："
            f"{len(extra)}"
        )

    # ========================================================
    # 完整
    # ========================================================

    if not missing:

        print()
        print(
            "=" * 80
        )

        print(
            "UTC 2020 PM2.5 数据完整！"
        )

        print(
            "开始：2020010100.png"
        )

        print(
            "结束：2020123123.png"
        )

        print(
            "数量：8784 张"
        )

        print(
            "尺寸：64 × 64"
        )

        print(
            f"目录：{OUTPUT_DIR}"
        )

        print(
            "=" * 80
        )

    # ========================================================
    # 不完整
    # ========================================================

    else:

        print()
        print(
            f"缺失 {len(missing)} 个小时。"
        )

        print(
            "前100个缺失时间："
        )

        for timestamp in missing[:100]:

            print(
                timestamp
            )

        if len(missing) > 100:

            print(
                "..."
            )

    return missing


# ============================================================
# 19. 主程序
# ============================================================

def main():

    print("=" * 80)

    print(
        "CN-Reanalysis UTC 2020 "
        "PM2.5 完整年度下载与处理"
    )

    print("=" * 80)

    print(
        f"ScienceDB链接文件："
        f"{INPUT_TXT}"
    )

    print(
        f"临时压缩包目录："
        f"{DOWNLOAD_DIR}"
    )

    print(
        f"最终输出目录："
        f"{OUTPUT_DIR}"
    )

    print()

    print(
        "目标区域："
    )

    print(
        f"经度："
        f"{TARGET_LON_MIN} ~ "
        f"{TARGET_LON_MAX}"
    )

    print(
        f"纬度："
        f"{TARGET_LAT_MIN} ~ "
        f"{TARGET_LAT_MAX}"
    )

    print(
        f"尺寸："
        f"{GRID_SIZE} × "
        f"{GRID_SIZE}"
    )

    print()

    print(
        "时间转换："
    )

    print(
        "UTC = 北京时间 - 8小时"
    )

    print()

    # ========================================================
    # Step 1：读取所有真实 URL
    # ========================================================

    url_map = load_download_urls()

    # ========================================================
    # Step 2：检查 367 天链接
    # ========================================================

    check_required_urls(
        url_map
    )

    # ========================================================
    # Step 3：生成需要下载的日期
    # ========================================================

    required_dates = (
        generate_required_dates()
    )

    print()
    print("=" * 80)

    print(
        f"需要处理北京时间日文件："
        f"{len(required_dates)} 个"
    )

    print(
        f"第一个："
        f"{required_dates[0]}"
    )

    print(
        f"最后一个："
        f"{required_dates[-1]}"
    )

    print("=" * 80)

    failed_days = []

    # ========================================================
    # Step 4：一天一天处理
    # ========================================================

    for index, date_str in enumerate(
        required_dates,
        start=1
    ):

        print()
        print()
        print(
            "=" * 80
        )

        print(
            f"年度进度："
            f"[{index}/{len(required_dates)}]"
        )

        print(
            "=" * 80
        )

        try:

            # ------------------------------------------------
            # 已完成则直接跳过
            # ------------------------------------------------

            if is_day_already_complete(
                date_str
            ):

                print(
                    f"{date_str} 已经完整处理，"
                    f"跳过下载。"
                )

                continue

            success = process_one_day(
                date_str,
                url_map[
                    date_str
                ]
            )

            if not success:

                failed_days.append(
                    date_str
                )

        except KeyboardInterrupt:

            print()
            print(
                "用户停止程序。"
            )

            break

        except Exception as e:

            print()
            print(
                f"{date_str} 出现异常："
                f"{repr(e)}"
            )

            failed_days.append(
                date_str
            )

    # ========================================================
    # Step 5：失败日期
    # ========================================================

    print()
    print("=" * 80)

    if failed_days:

        print(
            f"本次运行失败日期："
            f"{len(failed_days)}"
        )

        for date_str in failed_days:

            print(
                f"  {date_str}"
            )

        print()
        print(
            "重新运行本程序即可继续，"
            "已经完整处理的日期不会重新下载。"
        )

    else:

        print(
            "本次运行没有发现失败日期。"
        )

    # ========================================================
    # Step 6：检查最终完整性
    # ========================================================

    check_final_dataset()


# ============================================================
# 20. 程序入口
# ============================================================

if __name__ == "__main__":

    main()