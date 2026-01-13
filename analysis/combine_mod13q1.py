import os
import numpy as np
from osgeo import gdal
import glob
from tqdm import tqdm
from datetime import datetime
import re
import argparse
from collections import defaultdict

def parse_filename_and_date(filename):
    """
    解析文件名，提取前缀和日期
    
    参数:
        filename: 文件名，格式如 MOD13Q1_2005_01_01.tif
    
    返回:
        (prefix, date_obj, is_valid): 
        - prefix: 文件前缀，如 "MOD13Q1"
        - date_obj: 日期对象，datetime
        - is_valid: 是否为有效格式
    """
    # 匹配格式: MOD13Q1_yyyy_mm_dd.tif 或 MOD13Q1_yyyy_mm_dd.tiff
    pattern = r'^(MOD13Q1)_(\d{4})_(\d{2})_(\d{2})\.(tif|tiff)$'
    match = re.match(pattern, filename)
    
    if match:
        prefix = match.group(1)
        year = int(match.group(2))
        month = int(match.group(3))
        day = int(match.group(4))
        try:
            date_obj = datetime(year, month, day)
            return prefix, date_obj, True
        except ValueError:
            return None, None, False
    else:
        return None, None, False

def get_tif_info(tif_path):
    """
    获取TIF文件的基本信息
    
    参数:
        tif_path: TIF文件路径
    
    返回:
        (x_size, y_size, num_bands, data_type, geo_transform, projection)
    """
    src_ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
    if src_ds is None:
        raise Exception(f"无法打开TIF文件: {tif_path}")
    
    x_size = src_ds.RasterXSize
    y_size = src_ds.RasterYSize
    num_bands = src_ds.RasterCount
    data_type = src_ds.GetRasterBand(1).DataType
    geo_transform = src_ds.GetGeoTransform()
    projection = src_ds.GetProjection()
    
    src_ds = None
    return x_size, y_size, num_bands, data_type, geo_transform, projection

def read_tif_bands(tif_path):
    """
    读取TIF文件的所有波段数据
    
    参数:
        tif_path: TIF文件路径
    
    返回:
        bands_data: numpy数组，形状为 (num_bands, y_size, x_size)
    """
    src_ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
    if src_ds is None:
        raise Exception(f"无法打开TIF文件: {tif_path}")
    
    num_bands = src_ds.RasterCount
    y_size = src_ds.RasterYSize
    x_size = src_ds.RasterXSize
    
    # 读取所有波段
    bands_data = []
    for band_idx in range(1, num_bands + 1):
        band = src_ds.GetRasterBand(band_idx)
        data = band.ReadAsArray()
        bands_data.append(data)
    
    src_ds = None
    # 转换为numpy数组，形状为 (num_bands, y_size, x_size)
    bands_array = np.array(bands_data)
    return bands_array

def validate_data_consistency(files_by_year):
    """
    验证数据一致性：检查文件数量、波段数、空间尺寸等
    
    参数:
        files_by_year: 按年份分组的文件列表字典
    
    返回:
        (is_valid, error_messages): (是否有效, 错误消息列表)
    """
    error_messages = []
    
    if len(files_by_year) == 0:
        error_messages.append("错误: 没有找到有效格式的TIF文件")
        return False, error_messages
    
    # 获取所有年份
    years = sorted(files_by_year.keys())
    
    # 1. 检查每个年份内的文件数量是否一致
    file_counts = [len(files_by_year[year]) for year in years]
    if len(set(file_counts)) > 1:
        count_info = ", ".join([f"{year}: {count}个文件" for year, count in zip(years, file_counts)])
        error_messages.append(f"错误: 不同年份的文件数量不一致 - {count_info}")
    
    # 2. 检查所有文件的波段数是否一致（跨年份也要检查）
    band_counts = {}
    for year in years:
        year_files = files_by_year[year]
        for tif_path, date_obj, filename in year_files:
            try:
                _, _, num_bands, _, _, _ = get_tif_info(tif_path)
                if year not in band_counts:
                    band_counts[year] = []
                band_counts[year].append((filename, num_bands))
            except Exception as e:
                error_messages.append(f"错误: 无法读取文件 {filename} 的信息: {str(e)}")
                return False, error_messages
    
    # 检查每个年份内的波段数是否一致
    for year in years:
        bands_in_year = [bands for _, bands in band_counts[year]]
        if len(set(bands_in_year)) > 1:
            band_info = ", ".join([f"{fname}: {bands}波段" for fname, bands in band_counts[year]])
            error_messages.append(f"错误: {year}年内的文件波段数不一致 - {band_info}")
    
    # 检查跨年份的波段数是否一致
    if len(band_counts) > 1:
        first_year_bands = band_counts[years[0]][0][1]  # 第一个文件的波段数
        for year in years[1:]:
            year_bands = band_counts[year][0][1]
            if year_bands != first_year_bands:
                error_messages.append(f"错误: 跨年份波段数不一致 - {years[0]}年: {first_year_bands}波段, {year}年: {year_bands}波段")
    
    # 3. 检查所有文件的空间尺寸是否一致（跨年份也要检查）
    spatial_sizes = {}
    for year in years:
        year_files = files_by_year[year]
        for tif_path, date_obj, filename in year_files:
            try:
                x_size, y_size, _, _, _, _ = get_tif_info(tif_path)
                if year not in spatial_sizes:
                    spatial_sizes[year] = []
                spatial_sizes[year].append((filename, x_size, y_size))
            except Exception as e:
                error_messages.append(f"错误: 无法读取文件 {filename} 的空间尺寸: {str(e)}")
                return False, error_messages
    
    # 检查每个年份内的空间尺寸是否一致
    for year in years:
        sizes_in_year = [(x, y) for _, x, y in spatial_sizes[year]]
        if len(set(sizes_in_year)) > 1:
            size_info = ", ".join([f"{fname}: {x}x{y}" for fname, x, y in spatial_sizes[year]])
            error_messages.append(f"错误: {year}年内的文件空间尺寸不一致 - {size_info}")
    
    # 检查跨年份的空间尺寸是否一致
    if len(spatial_sizes) > 1:
        first_year_size = spatial_sizes[years[0]][0][1:]  # (x_size, y_size)
        for year in years[1:]:
            year_size = spatial_sizes[year][0][1:]
            if year_size != first_year_size:
                error_messages.append(f"错误: 跨年份空间尺寸不一致 - {years[0]}年: {first_year_size[0]}x{first_year_size[1]}, {year}年: {year_size[0]}x{year_size[1]}")
    
    # 4. 检查是否有缺失的日期（检查日期是否连续，MOD13Q1通常是16天间隔）
    # 这里我们检查每个年份内的日期是否有明显缺失
    for year in years:
        year_files = files_by_year[year]
        dates = sorted([date_obj for _, date_obj, _ in year_files])
        
        # 检查是否有重复日期
        date_strings = [d.strftime('%Y-%m-%d') for d in dates]
        if len(date_strings) != len(set(date_strings)):
            duplicates = [d for d in date_strings if date_strings.count(d) > 1]
            error_messages.append(f"错误: {year}年存在重复日期: {', '.join(set(duplicates))}")
    
    is_valid = len(error_messages) == 0
    return is_valid, error_messages

def combine_tifs_by_year(input_folder, output_folder, prefix="MOD13Q1_sask", compress="DEFLATE", compress_level=6):
    """
    按年份将TIF文件按通道拼接
    
    参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        prefix: 输出文件前缀，默认为 "MOD13Q1_sask"
        compress: 压缩方式，可选: "NONE"（无压缩）、"LZW"（平衡）、"DEFLATE"（最佳压缩，默认）
        compress_level: 压缩级别，1-9（仅对DEFLATE有效，默认6，平衡速度和压缩比）
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有TIF文件
    all_tif_files = glob.glob(os.path.join(input_folder, '*.tif')) + \
                    glob.glob(os.path.join(input_folder, '*.tiff'))
    
    if len(all_tif_files) == 0:
        print(f"错误: 在 {input_folder} 中没有找到TIF文件")
        return
    
    print(f"找到 {len(all_tif_files)} 个TIF文件")
    
    # 解析文件名并按年份分组
    files_by_year = defaultdict(list)
    invalid_count = 0
    
    for tif_path in all_tif_files:
        filename = os.path.basename(tif_path)
        prefix_name, date_obj, is_valid = parse_filename_and_date(filename)
        
        if not is_valid:
            invalid_count += 1
            continue
        
        year = date_obj.year
        files_by_year[year].append((tif_path, date_obj, filename))
    
    if invalid_count > 0:
        print(f"跳过 {invalid_count} 个无效格式的文件")
    
    if len(files_by_year) == 0:
        print("错误: 没有找到有效格式的TIF文件")
        return
    
    print(f"找到 {len(files_by_year)} 个年份的数据")
    
    # ========== 运行前全面检查 ==========
    print("\n" + "=" * 60)
    print("开始数据一致性检查...")
    print("=" * 60)
    
    is_valid, error_messages = validate_data_consistency(files_by_year)
    
    if not is_valid:
        print("\n❌ 数据检查失败，发现以下问题：")
        for i, error_msg in enumerate(error_messages, 1):
            print(f"  {i}. {error_msg}")
        print("\n请修复这些问题后重新运行。")
        return
    
    print("✓ 所有检查通过！")
    print("  - 每个年份的文件数量一致")
    print("  - 所有文件的波段数一致")
    print("  - 所有文件的空间尺寸一致")
    print("  - 没有发现重复日期")
    print("=" * 60 + "\n")
    
    # 处理每个年份
    for year in sorted(files_by_year.keys()):
        print(f"\n处理年份: {year}")
        year_files = files_by_year[year]
        
        # 按日期排序
        year_files.sort(key=lambda x: x[1])
        
        print(f"  找到 {len(year_files)} 个文件")
        
        # 检查第一个文件获取基本信息
        first_file_path = year_files[0][0]
        x_size, y_size, num_bands, data_type, geo_transform, projection = get_tif_info(first_file_path)
        
        print(f"  空间尺寸: {x_size} x {y_size}")
        print(f"  每个文件波段数: {num_bands}")
        print(f"  数据类型: {gdal.GetDataTypeName(data_type)}")
        
        # 注意: 数据一致性已在运行前检查中验证，此处不再重复检查
        
        # 计算总波段数
        total_bands = num_bands * len(year_files)
        print(f"  总波段数: {total_bands} ({len(year_files)} 个文件 × {num_bands} 个波段)")
        
        # 创建输出文件
        output_filename = f"{prefix}_{year}.tiff"
        output_path = os.path.join(output_folder, output_filename)
        
        # 创建输出TIF文件
        driver = gdal.GetDriverByName('GTiff')
        
        # 根据文件大小选择创建选项
        estimated_size_mb = (x_size * y_size * total_bands * (gdal.GetDataTypeSize(data_type) // 8)) / (1024 * 1024)
        creation_options = ['TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512']
        
        if estimated_size_mb > 4096:
            creation_options.append('BIGTIFF=YES')
            print(f"  启用BIGTIFF支持（估计大小: {estimated_size_mb/1024:.2f} GB）")
        
        # 添加压缩选项
        if compress.upper() == "NONE":
            print("  压缩: 无压缩（最快）")
        elif compress.upper() == "LZW":
            creation_options.append('COMPRESS=LZW')
            print("  压缩: LZW（平衡速度和压缩比）")
        elif compress.upper() == "DEFLATE":
            creation_options.append('COMPRESS=DEFLATE')
            # 压缩级别 1-9，默认6（平衡速度和压缩比）
            compress_level = max(1, min(9, compress_level))
            creation_options.append(f'ZLEVEL={compress_level}')
            
            # 根据数据类型选择PREDICTOR
            # PREDICTOR=2 用于整数数据，PREDICTOR=3 用于浮点数据
            if data_type in [gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, 
                            gdal.GDT_UInt32, gdal.GDT_Int32]:
                creation_options.append('PREDICTOR=2')
            elif data_type in [gdal.GDT_Float32, gdal.GDT_Float64]:
                creation_options.append('PREDICTOR=3')
            
            print(f"  压缩: DEFLATE（级别{compress_level}，最佳压缩比）")
        else:
            raise ValueError(f"不支持的压缩方式: {compress}，请选择 NONE、LZW 或 DEFLATE")
        
        out_ds = driver.Create(
            output_path,
            x_size,
            y_size,
            total_bands,
            data_type,
            options=creation_options
        )
        
        if out_ds is None:
            raise Exception(f"无法创建输出文件: {output_path}")
        
        # 设置地理信息
        out_ds.SetGeoTransform(geo_transform)
        out_ds.SetProjection(projection)
        
        # 读取并写入每个文件的所有波段
        current_band = 1
        for tif_path, date_obj, filename in tqdm(year_files, desc=f"  处理 {year} 年文件"):
            # 读取所有波段
            bands_data = read_tif_bands(tif_path)
            
            # 写入每个波段
            for band_idx in range(num_bands):
                out_band = out_ds.GetRasterBand(current_band)
                
                # 写入数据
                out_band.WriteArray(bands_data[band_idx])
                
                # 从源文件复制NoData值（如果有）
                src_ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
                if src_ds:
                    src_band = src_ds.GetRasterBand(band_idx + 1)
                    nodata_value = src_band.GetNoDataValue()
                    if nodata_value is not None:
                        out_band.SetNoDataValue(nodata_value)
                    src_ds = None
                
                current_band += 1
        
        # 刷新缓存并关闭
        out_ds.FlushCache()
        out_ds = None
        
        print(f"  ✓ 完成: {output_filename} ({total_bands} 个波段)")
    
    print(f"\n处理完成！结果保存在: {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='按年份将MOD13Q1 TIF文件按通道拼接',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认DEFLATE压缩（推荐，最佳压缩比）
  python combine_mod13q1.py \\
    --input_folder /path/to/input/folder \\
    --output_folder /path/to/output/folder \\
    --prefix MOD13Q1_sask
  
  # 使用LZW压缩（平衡速度和压缩比）
  python combine_mod13q1.py \\
    --input_folder /path/to/input/folder \\
    --output_folder /path/to/output/folder \\
    --compress LZW
  
  # 使用DEFLATE最高压缩级别（最慢但文件最小）
  python combine_mod13q1.py \\
    --input_folder /path/to/input/folder \\
    --output_folder /path/to/output/folder \\
    --compress DEFLATE --compress_level 9
  
  # 不使用压缩（最快但文件最大）
  python combine_mod13q1.py \\
    --input_folder /path/to/input/folder \\
    --output_folder /path/to/output/folder \\
    --compress NONE

功能说明:
  - 读取输入文件夹中格式为 MOD13Q1_YYYY_MM_DD.tif 的文件
  - 按年份分组，按日期顺序将每个年份内的所有TIF文件按通道拼接
  - 输出文件格式: {prefix}_YYYY.tiff
  - 例如: MOD13Q1_2005_01_01.tif (10波段) + MOD13Q1_2005_01_20.tif (10波段) 
         -> MOD13Q1_sask_2005.tiff (20波段: 前10个是01-01的，后10个是01-20的)
  - 支持压缩选项以减少文件大小（DEFLATE通常可减少50-80%的文件大小）
        """
    )
    
    parser.add_argument(
        '--input_folder',
        type=str,
        default='/mnt/storage/benchmark_datasets/MODIS/sask/drivers/MOD13Q1_raw',
        help='输入文件夹路径（包含按日期命名的TIF文件）'
    )
    
    parser.add_argument(
        '--output_folder',
        type=str,
        default='/mnt/storage/benchmark_datasets/MODIS/sask/drivers/MOD13Q1_combine',
        help='输出文件夹路径'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='MOD13Q1_sask',
        help='输出文件前缀（默认: MOD13Q1_sask）'
    )
    
    parser.add_argument(
        '--compress',
        type=str,
        choices=['NONE', 'LZW', 'DEFLATE'],
        default='DEFLATE',
        help='压缩方式: NONE（无压缩，最快）、LZW（平衡）、DEFLATE（最佳压缩比，默认）'
    )
    
    parser.add_argument(
        '--compress_level',
        type=int,
        default=6,
        choices=range(1, 10),
        metavar='[1-9]',
        help='压缩级别，1-9（仅对DEFLATE有效，默认6，平衡速度和压缩比。1最快但压缩比低，9最慢但压缩比最高）'
    )
    
    args = parser.parse_args()
    
    # 验证文件夹是否存在
    if not os.path.isdir(args.input_folder):
        print(f"错误: 输入文件夹不存在: {args.input_folder}")
        exit(1)
    
    print("=" * 60)
    print("MOD13Q1 TIF文件按年份通道拼接工具")
    print("=" * 60)
    print(f"输入文件夹: {args.input_folder}")
    print(f"输出文件夹: {args.output_folder}")
    print(f"输出前缀: {args.prefix}")
    print(f"压缩方式: {args.compress}")
    if args.compress == 'DEFLATE':
        print(f"压缩级别: {args.compress_level}")
    print("=" * 60)
    
    combine_tifs_by_year(args.input_folder, args.output_folder, args.prefix, 
                         args.compress, args.compress_level)

