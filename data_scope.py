import os
import re
import cv2
import numpy as np
import pandas as pd


# ============================================================
# 配置
# ============================================================
PRESS_DIR = r"D:\dataset\press"
TEMP_DIR = r"D:\dataset\TEMP_daily"
WIND_DIR = r"D:\dataset\WDSP_daily"

START_YEAR = 2013
END_YEAR = 2018

# 结果保存位置
OUTPUT_CSV = r"D:\dataset\range_2013_2018.csv"


# ============================================================
# 工具函数
# ============================================================
def extract_year(filename):
    """
    从文件名开头提取年份。

    支持：
        2013010103.png
        20130101.csv
    """
    basename = os.path.basename(filename)

    match = re.match(r"(\d{4})", basename)
    if match is None:
        return None

    return int(match.group(1))


def valid_year(filename):
    """判断文件是否属于 2013-2018。"""
    year = extract_year(filename)

    if year is None:
        return False

    return START_YEAR <= year <= END_YEAR


# ============================================================
# 气压 PNG 统计
# ============================================================
def analyze_pressure(folder):
    print("\n" + "=" * 70)
    print("开始统计气压数据")
    print("=" * 70)

    files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".png") and valid_year(f)
    ])

    if len(files) == 0:
        raise RuntimeError(
            f"在 {folder} 中没有找到 {START_YEAR}-{END_YEAR} 年的 PNG 文件。"
        )

    global_min = float("inf")
    global_max = float("-inf")

    min_file = None
    max_file = None
    min_location = None
    max_location = None

    total_pixels = 0
    valid_pixels = 0

    yearly_stats = {}

    for i, path in enumerate(files):

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"[WARNING] 无法读取：{path}")
            continue

        img = img.astype(np.float64)

        total_pixels += img.size

        valid_mask = np.isfinite(img)
        values = img[valid_mask]

        if values.size == 0:
            continue

        valid_pixels += values.size

        current_min = float(values.min())
        current_max = float(values.max())

        # ----------------------------------------------------
        # 全局最小值
        # ----------------------------------------------------
        if current_min < global_min:
            global_min = current_min
            min_file = path

            pos = np.unravel_index(
                np.nanargmin(img),
                img.shape
            )
            min_location = pos

        # ----------------------------------------------------
        # 全局最大值
        # ----------------------------------------------------
        if current_max > global_max:
            global_max = current_max
            max_file = path

            pos = np.unravel_index(
                np.nanargmax(img),
                img.shape
            )
            max_location = pos

        # ----------------------------------------------------
        # 按年份统计
        # ----------------------------------------------------
        year = extract_year(path)

        if year not in yearly_stats:
            yearly_stats[year] = {
                "min": float("inf"),
                "max": float("-inf"),
                "count": 0,
                "files": 0,
            }

        yearly_stats[year]["min"] = min(
            yearly_stats[year]["min"],
            current_min
        )

        yearly_stats[year]["max"] = max(
            yearly_stats[year]["max"],
            current_max
        )

        yearly_stats[year]["count"] += values.size
        yearly_stats[year]["files"] += 1

        if (i + 1) % 1000 == 0:
            print(
                f"气压处理进度："
                f"{i + 1}/{len(files)}"
            )

    print("\n气压统计完成")
    print(f"文件数量      : {len(files)}")
    print(f"总像素数量    : {total_pixels:,}")
    print(f"有效像素数量  : {valid_pixels:,}")

    print(f"\nPressure Min : {global_min}")
    print(f"对应文件      : {min_file}")
    print(f"像素位置      : {min_location}")

    print(f"\nPressure Max : {global_max}")
    print(f"对应文件      : {max_file}")
    print(f"像素位置      : {max_location}")

    return {
        "name": "Pressure",
        "min": global_min,
        "max": global_max,
        "min_file": min_file,
        "max_file": max_file,
        "count": valid_pixels,
        "files": len(files),
        "yearly": yearly_stats,
    }


# ============================================================
# CSV 气象变量统计
# ============================================================
def analyze_csv_variable(folder, variable_name):
    """
    CSV 格式假设为：

        longitude, latitude, value

    与当前模型中的：
        .iloc[:, 2]

    保持一致。
    """

    print("\n" + "=" * 70)
    print(f"开始统计 {variable_name}")
    print("=" * 70)

    files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".csv") and valid_year(f)
    ])

    if len(files) == 0:
        raise RuntimeError(
            f"在 {folder} 中没有找到 "
            f"{START_YEAR}-{END_YEAR} 年 CSV 文件。"
        )

    global_min = float("inf")
    global_max = float("-inf")

    min_file = None
    max_file = None

    min_row = None
    max_row = None

    total_values = 0
    valid_values = 0

    yearly_stats = {}

    for i, path in enumerate(files):

        try:
            df = pd.read_csv(
                path,
                header=None
            )
        except Exception as e:
            print(
                f"[WARNING] 读取失败：{path}\n"
                f"原因：{e}"
            )
            continue

        if df.shape[1] < 3:
            print(
                f"[WARNING] 列数不足3列：{path}, "
                f"shape={df.shape}"
            )
            continue

        # ----------------------------------------------------
        # 第三列就是模型实际使用的气象值
        # ----------------------------------------------------
        values = pd.to_numeric(
            df.iloc[:, 2],
            errors="coerce"
        ).to_numpy(dtype=np.float64)

        total_values += len(values)

        valid_mask = np.isfinite(values)
        valid = values[valid_mask]

        if valid.size == 0:
            continue

        valid_values += valid.size

        current_min = float(valid.min())
        current_max = float(valid.max())

        # ----------------------------------------------------
        # 全局最小值
        # ----------------------------------------------------
        if current_min < global_min:
            global_min = current_min
            min_file = path

            valid_indices = np.where(valid_mask)[0]
            local_idx = np.argmin(valid)

            original_idx = valid_indices[local_idx]

            min_row = df.iloc[original_idx].tolist()

        # ----------------------------------------------------
        # 全局最大值
        # ----------------------------------------------------
        if current_max > global_max:
            global_max = current_max
            max_file = path

            valid_indices = np.where(valid_mask)[0]
            local_idx = np.argmax(valid)

            original_idx = valid_indices[local_idx]

            max_row = df.iloc[original_idx].tolist()

        # ----------------------------------------------------
        # 每年统计
        # ----------------------------------------------------
        year = extract_year(path)

        if year not in yearly_stats:
            yearly_stats[year] = {
                "min": float("inf"),
                "max": float("-inf"),
                "count": 0,
                "files": 0,
            }

        yearly_stats[year]["min"] = min(
            yearly_stats[year]["min"],
            current_min
        )

        yearly_stats[year]["max"] = max(
            yearly_stats[year]["max"],
            current_max
        )

        yearly_stats[year]["count"] += valid.size
        yearly_stats[year]["files"] += 1

        if (i + 1) % 500 == 0:
            print(
                f"{variable_name}处理进度："
                f"{i + 1}/{len(files)}"
            )

    print(f"\n{variable_name}统计完成")
    print(f"文件数量      : {len(files)}")
    print(f"原始数值数量  : {total_values:,}")
    print(f"有效数值数量  : {valid_values:,}")

    print(f"\n{variable_name} Min : {global_min}")
    print(f"对应文件      : {min_file}")
    print(f"对应数据行    : {min_row}")

    print(f"\n{variable_name} Max : {global_max}")
    print(f"对应文件      : {max_file}")
    print(f"对应数据行    : {max_row}")

    return {
        "name": variable_name,
        "min": global_min,
        "max": global_max,
        "min_file": min_file,
        "max_file": max_file,
        "count": valid_values,
        "files": len(files),
        "yearly": yearly_stats,
    }


# ============================================================
# 打印年度结果
# ============================================================
def print_yearly_stats(result):

    print("\n" + "-" * 70)
    print(f"{result['name']} 各年份数值范围")
    print("-" * 70)

    print(
        f"{'Year':<8}"
        f"{'Min':>15}"
        f"{'Max':>15}"
        f"{'Files':>12}"
        f"{'Values':>18}"
    )

    for year in range(START_YEAR, END_YEAR + 1):

        if year not in result["yearly"]:
            continue

        item = result["yearly"][year]

        print(
            f"{year:<8}"
            f"{item['min']:>15.6f}"
            f"{item['max']:>15.6f}"
            f"{item['files']:>12}"
            f"{item['count']:>18,}"
        )


# ============================================================
# 保存 CSV
# ============================================================
def save_results(results, output_path):

    rows = []

    # 总体
    for result in results:

        rows.append({
            "Variable": result["name"],
            "Period": f"{START_YEAR}-{END_YEAR}",
            "Min": result["min"],
            "Max": result["max"],
            "Files": result["files"],
            "Valid_values": result["count"],
        })

        # 每年
        for year in sorted(result["yearly"].keys()):

            item = result["yearly"][year]

            rows.append({
                "Variable": result["name"],
                "Period": str(year),
                "Min": item["min"],
                "Max": item["max"],
                "Files": item["files"],
                "Valid_values": item["count"],
            })

    df = pd.DataFrame(rows)

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\n统计结果已保存：\n{output_path}"
    )


# ============================================================
# 主程序
# ============================================================
def main():

    print("=" * 70)
    print(
        f"统计时间范围："
        f"{START_YEAR}-01-01 ~ {END_YEAR}-12-31"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Pressure
    # --------------------------------------------------------
    pressure_result = analyze_pressure(
        PRESS_DIR
    )

    # --------------------------------------------------------
    # 2. Temperature
    # --------------------------------------------------------
    temperature_result = analyze_csv_variable(
        TEMP_DIR,
        "Temperature"
    )

    # --------------------------------------------------------
    # 3. Wind Speed
    # --------------------------------------------------------
    wind_result = analyze_csv_variable(
        WIND_DIR,
        "Wind Speed"
    )

    results = [
        pressure_result,
        temperature_result,
        wind_result
    ]

    # --------------------------------------------------------
    # 年度统计
    # --------------------------------------------------------
    for result in results:
        print_yearly_stats(result)

    # --------------------------------------------------------
    # 最终汇总
    # --------------------------------------------------------
    print("\n")
    print("=" * 70)
    print("2013-2018 总体数值范围")
    print("=" * 70)

    for result in results:

        print(
            f"{result['name']:<15}: "
            f"min = {result['min']:.6f}, "
            f"max = {result['max']:.6f}"
        )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------
    save_results(
        results,
        OUTPUT_CSV
    )


if __name__ == "__main__":
    main()