import os
import numpy as np
from osgeo import gdal, ogr
import glob
from tqdm import tqdm
from datetime import datetime
import re
import argparse
from multiprocessing import Pool, cpu_count

def parse_filename_and_date(filename):
    """
    解析文件名，提取前缀和日期
    
    参数:
        filename: 文件名，格式如 ERA5_complement_2024_12_31_tile03.tif
    
    返回:
        (prefix, date_str, tile_str, is_valid): 
        - prefix: 文件前缀，如 "ERA5_complement"
        - date_str: 日期字符串，如 "2024_12_31"
        - tile_str: tile字符串，如 "tile03"
        - is_valid: 是否为有效格式
    """
    # 匹配格式: xxxx_yyyy_mm_dd_tilexx.tif
    pattern = r'^(.+)_(\d{4})_(\d{2})_(\d{2})_(tile\d{2})\.(tif|tiff)$'
    match = re.match(pattern, filename)
    
    if match:
        prefix = match.group(1)
        year = match.group(2)
        month = match.group(3)
        day = match.group(4)
        tile_str = match.group(5)
        date_str = f"{year}_{month}_{day}"
        return prefix, date_str, tile_str, True
    else:
        return None, None, None, False

def is_date_in_range(date_str, start_date, end_date):
    """
    检查日期是否在指定范围内
    
    参数:
        date_str: 日期字符串，格式 "yyyy_mm_dd"
        start_date: 开始日期，datetime对象
        end_date: 结束日期，datetime对象
    
    返回:
        bool: 是否在范围内
    """
    try:
        date_obj = datetime.strptime(date_str, '%Y_%m_%d')
        return start_date <= date_obj <= end_date
    except ValueError:
        return False

def generate_output_filename(prefix, date_str):
    """
    生成输出文件名，格式: xxxx_yyyy_mm_dd.tif
    
    参数:
        prefix: 文件前缀
        date_str: 日期字符串，格式 "yyyy_mm_dd"
    
    返回:
        输出文件名
    """
    return f"{prefix}_{date_str}.tif"

def mask_tif_with_gdalwarp(tif_path, shapefile_path, output_path):
    """
    使用gdalwarp进行掩膜（最快的方法，使用GDAL的底层优化）
    
    参数:
        tif_path: 输入TIF文件路径
        shapefile_path: Shapefile文件路径
        output_path: 输出TIF文件路径
    """
    import subprocess
    
    # 使用gdalwarp命令行工具（最快的方法，使用GDAL的C++底层优化）
    # -cutline: 使用shapefile作为裁剪边界
    # -crop_to_cutline: 裁剪到cutline的边界
    # -dstnodata: 设置NoData值
    # -co: 创建选项（无压缩，大块，多线程）
    # -multi: 多线程处理
    # -wo: warp选项（使用多线程）
    # -wm: 内存限制（针对几十GB数据增加到4GB）
    cmd = [
        'gdalwarp',
        '-cutline', shapefile_path,
        '-crop_to_cutline',
        '-dstnodata', '0',
        '-co', 'TILED=YES',
        '-co', 'BLOCKXSIZE=512',  # 使用512的块大小，平衡内存和速度
        '-co', 'BLOCKYSIZE=512',
        '-co', 'BIGTIFF=YES',  # 支持>4GB文件
        '-co', 'NUM_THREADS=ALL_CPUS',  # 创建时使用多线程
        '-multi',  # 多线程warp
        '-wo', 'NUM_THREADS=ALL_CPUS',  # warp操作使用所有CPU核心
        '-wm', '4096',  # 增大工作内存到4GB以处理超大文件
        '-overwrite',
        '-q',  # 静默模式
        tif_path,
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        raise Exception(f"gdalwarp失败: {result.stderr}")

def mask_tif_with_shapefile(tif_path, mask_array, geo_transform, projection, output_path):
    """
    使用预生成的掩膜数组掩膜tif文件（优化版本，避免重复栅格化）
    注意：这个方法比gdalwarp慢，但提供了更多控制
    
    参数:
        tif_path: 输入TIF文件路径
        mask_array: 预生成的掩膜数组
        geo_transform: 地理变换参数
        projection: 投影信息
        output_path: 输出TIF文件路径
    """
    # 打开TIF文件
    src_ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
    if src_ds is None:
        raise Exception(f"无法打开TIF文件: {tif_path}")
    
    # 获取TIF文件信息
    x_size = src_ds.RasterXSize
    y_size = src_ds.RasterYSize
    num_bands = src_ds.RasterCount
    
    # 验证尺寸是否匹配
    if mask_array.shape != (y_size, x_size):
        raise Exception(f"掩膜尺寸 {mask_array.shape} 与TIF尺寸 {(y_size, x_size)} 不匹配")
    
    # 获取数据类型
    data_type = src_ds.GetRasterBand(1).DataType
    
    # 针对几十GB的超大数据优化分块策略
    pixel_size = gdal.GetDataTypeSize(data_type) // 8  # 每个像素的字节数
    estimated_size_mb = (x_size * y_size * num_bands * pixel_size) / (1024 * 1024)
    
    print(f"  文件估计大小: {estimated_size_mb:.1f} MB ({estimated_size_mb/1024:.2f} GB)")
    
    # 对于几十GB的数据，必须使用分块处理
    # 阈值设为1GB，超过1GB的文件都分块处理
    use_chunking = estimated_size_mb > 1024
    
    if use_chunking:
        # 根据文件大小动态调整块大小
        if estimated_size_mb > 10240:  # >10GB
            block_size = 2048  # 使用2048x2048，平衡内存和I/O
        elif estimated_size_mb > 5120:  # 5-10GB
            block_size = 4096  # 使用4096x4096
        else:  # 1-5GB
            block_size = 8192  # 使用8192x8192
        print(f"  使用分块处理，块大小: {block_size}x{block_size}")
    else:
        block_size = max(x_size, y_size)  # 一次性处理
        print("  使用一次性处理")
    
    # 创建输出文件 - 针对超大数据优化
    driver = gdal.GetDriverByName('GTiff')
    creation_options = [
        'TILED=YES',
        'BLOCKXSIZE=1024',  # 使用大块减少I/O
        'BLOCKYSIZE=1024'
    ]
    
    # 对于超大文件（>4GB），启用BIGTIFF
    if estimated_size_mb > 4096:
        creation_options.append('BIGTIFF=YES')
        print("  启用BIGTIFF支持")
    
    # 对于超大文件，使用LZW压缩减少磁盘写入（LZW速度快且有效）
    # 对于中等文件，不压缩更快
    if estimated_size_mb > 10240:  # >10GB使用LZW压缩
        creation_options.append('COMPRESS=LZW')
        print("  启用LZW压缩以减少磁盘I/O")
    # 否则不压缩，速度最快
    
    out_ds = driver.Create(
        output_path,
        x_size,
        y_size,
        num_bands,
        data_type,
        options=creation_options
    )
    
    if out_ds is None:
        raise Exception(f"无法创建输出文件: {output_path}")
    
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(projection)
    
    # 处理每个波段
    for band_idx in range(1, num_bands + 1):
        src_band = src_ds.GetRasterBand(band_idx)
        out_band = out_ds.GetRasterBand(band_idx)
        
        # 获取NoData值
        nodata_value = src_band.GetNoDataValue()
        if nodata_value is None:
            nodata_value = 0
        out_band.SetNoDataValue(nodata_value)
        
        # 复制颜色表（如果存在）
        color_table = src_band.GetColorTable()
        if color_table is not None:
            out_band.SetColorTable(color_table)
        
        if use_chunking:
            # 分块处理超大文件（避免一次性加载整个数组到内存）
            total_blocks = ((y_size + block_size - 1) // block_size) * ((x_size + block_size - 1) // block_size)
            block_count = 0
            
            for y_offset in range(0, y_size, block_size):
                y_block_size = min(block_size, y_size - y_offset)
                for x_offset in range(0, x_size, block_size):
                    x_block_size = min(block_size, x_size - x_offset)
                    block_count += 1
                    
                    # 每处理10个块显示一次进度
                    if block_count % 10 == 0:
                        progress = (block_count / total_blocks) * 100
                        print(f"    波段 {band_idx}/{num_bands} 进度: {progress:.1f}% ({block_count}/{total_blocks}块)")
                    
                    # 读取数据块
                    data = src_band.ReadAsArray(
                        xoff=x_offset, yoff=y_offset,
                        win_xsize=x_block_size, win_ysize=y_block_size
                    )
                    
                    # 读取对应的掩膜块
                    mask_block = mask_array[y_offset:y_offset+y_block_size, 
                                           x_offset:x_offset+x_block_size]
                    
                    # 应用掩膜（直接在原数组上修改，避免copy）
                    # 先检查是否需要修改（如果掩膜块全为1，可以跳过）
                    if not np.all(mask_block == 1):
                        # 直接在data上应用掩膜，避免copy
                        data[mask_block != 1] = nodata_value
                    
                    # 写入数据块
                    out_band.WriteArray(data, xoff=x_offset, yoff=y_offset)
                    
                    # 显式删除数据块以释放内存
                    del data, mask_block
        else:
            # 一次性处理（最快）
            data = src_band.ReadAsArray()
            
            # 应用掩膜（直接在原数组上修改，避免copy）
            if not np.all(mask_array == 1):
                data[mask_array != 1] = nodata_value
            
            # 写入输出
            out_band.WriteArray(data)
            
            # 释放内存
            del data
    
    # 关闭数据集
    out_ds.FlushCache()
    src_ds = None
    out_ds = None

def create_mask_from_shapefile(shapefile_path, sample_tif_path):
    """
    预先从shapefile创建掩膜数组（只执行一次）
    
    参数:
        shapefile_path: Shapefile文件路径
        sample_tif_path: 示例TIF文件路径（用于获取尺寸和地理变换）
    
    返回:
        (mask_array, geo_transform, projection)
    """
    # 打开示例TIF文件获取尺寸和地理信息
    sample_ds = gdal.Open(sample_tif_path, gdal.GA_ReadOnly)
    if sample_ds is None:
        raise Exception(f"无法打开示例TIF文件: {sample_tif_path}")
    
    geo_transform = sample_ds.GetGeoTransform()
    projection = sample_ds.GetProjection()
    x_size = sample_ds.RasterXSize
    y_size = sample_ds.RasterYSize
    sample_ds = None
    
    # 打开Shapefile
    shp_ds = ogr.Open(shapefile_path)
    if shp_ds is None:
        raise Exception(f"无法打开Shapefile: {shapefile_path}")
    
    layer = shp_ds.GetLayer()
    
    # 创建内存中的掩膜栅格
    mem_driver = gdal.GetDriverByName('MEM')
    mask_ds = mem_driver.Create('', x_size, y_size, 1, gdal.GDT_Byte)
    mask_ds.SetGeoTransform(geo_transform)
    mask_ds.SetProjection(projection)
    
    # 栅格化shapefile（只执行一次）
    gdal.RasterizeLayer(mask_ds, [1], layer, burn_values=[1])
    mask_array = mask_ds.GetRasterBand(1).ReadAsArray()
    
    # 清理
    shp_ds = None
    mask_ds = None
    
    return mask_array, geo_transform, projection

def process_single_file(args):
    """
    处理单个文件的包装函数，用于多进程
    
    参数:
        args: (tif_path, prefix, date_str, shapefile_path, output_folder)
    
    返回:
        (success, tif_name, output_filename, error_msg)
    """
    tif_path, prefix, date_str, shapefile_path, output_folder = args
    tif_name = os.path.basename(tif_path)
    output_filename = generate_output_filename(prefix, date_str)
    output_path = os.path.join(output_folder, output_filename)
    
    try:
        # 使用gdalwarp命令行工具，这是最快的方法
        mask_tif_with_gdalwarp(tif_path, shapefile_path, output_path)
        return (True, tif_name, output_filename, None)
    except Exception as e:
        return (False, tif_name, output_filename, str(e))

def batch_mask_tifs(tif_folder, shapefile_folder, output_folder, num_workers=None):
    """
    批量处理：使用shapefile掩膜多个tif文件
    
    参数:
        tif_folder: TIF文件所在文件夹
        shapefile_folder: Shapefile文件所在文件夹
        output_folder: 输出文件夹
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 日期范围设置
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2004, 12, 31)
    
    # 获取所有TIF文件
    all_tif_files = glob.glob(os.path.join(tif_folder, '*.tif')) + \
                    glob.glob(os.path.join(tif_folder, '*.tiff'))
    
    if len(all_tif_files) == 0:
        print(f"警告: 在 {tif_folder} 中没有找到TIF文件")
        return
    
    # 过滤文件：只处理tile00且在日期范围内的文件
    tif_files = []
    filtered_count = 0
    for tif_path in all_tif_files:
        filename = os.path.basename(tif_path)
        prefix, date_str, tile_str, is_valid = parse_filename_and_date(filename)
        
        if not is_valid:
            filtered_count += 1
            continue
        
        # 只处理tile00
        if tile_str != 'tile00':
            filtered_count += 1
            continue
        
        # 检查日期范围
        if not is_date_in_range(date_str, start_date, end_date):
            filtered_count += 1
            continue
        
        tif_files.append((tif_path, prefix, date_str))
    
    print(f"找到 {len(all_tif_files)} 个TIF文件")
    print(f"过滤后剩余 {len(tif_files)} 个文件（tile00且在2005-01-01到2015-12-31范围内）")
    print(f"跳过 {filtered_count} 个文件")
    
    if len(tif_files) == 0:
        print("警告: 没有符合条件的TIF文件需要处理")
        return
    
    # 获取所有Shapefile文件
    shp_files = glob.glob(os.path.join(shapefile_folder, '*.shp'))
    
    if len(shp_files) == 0:
        print(f"警告: 在 {shapefile_folder} 中没有找到Shapefile文件")
        return
    
    # 根据数据大小动态设置进程数
    if num_workers is None:
        # 检查第一个文件的大小来估算数据规模
        sample_tif = gdal.Open(tif_files[0][0], gdal.GA_ReadOnly)
        if sample_tif:
            x_size = sample_tif.RasterXSize
            y_size = sample_tif.RasterYSize
            num_bands = sample_tif.RasterCount
            data_type = sample_tif.GetRasterBand(1).DataType
            pixel_size = gdal.GetDataTypeSize(data_type) // 8
            estimated_size_mb = (x_size * y_size * num_bands * pixel_size) / (1024 * 1024)
            sample_tif = None
            
            # 根据文件大小调整进程数
            if estimated_size_mb > 10240:  # >10GB，使用单进程避免内存问题
                num_workers = 1
                print(f"检测到超大文件（{estimated_size_mb/1024:.1f} GB），使用单进程处理以避免内存问题")
            elif estimated_size_mb > 5120:  # 5-10GB，使用2个进程
                num_workers = 2
                print(f"检测到大文件（{estimated_size_mb/1024:.1f} GB），使用2个进程")
            elif estimated_size_mb > 2048:  # 2-5GB，使用3个进程
                num_workers = 3
            else:  # <2GB，使用4个进程
                num_workers = min(cpu_count() // 2, 4)
        else:
            num_workers = 2  # 默认使用2个进程
    
    # 如果只有一个shapefile，用它掩膜所有tif
    if len(shp_files) == 1:
        shapefile_path = shp_files[0]
        print(f"使用单个Shapefile掩膜所有TIF文件: {os.path.basename(shapefile_path)}")
        
        print(f"使用 {num_workers} 个进程并行处理")
        print("使用gdalwarp命令行工具进行掩膜（最快方法）")
        
        # 准备多进程参数（使用gdalwarp，直接传递shapefile路径）
        process_args = [
            (tif_path, prefix, date_str, shapefile_path, output_folder)
            for tif_path, prefix, date_str in tif_files
        ]
        
        # 使用多进程处理
        success_count = 0
        fail_count = 0
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_file, process_args),
                total=len(process_args),
                desc="处理TIF文件"
            ))
        
        # 统计结果
        for success, tif_name, output_filename, error_msg in results:
            if success:
                success_count += 1
            else:
                fail_count += 1
                print(f"✗ 处理失败 {tif_name}: {error_msg}")
        
        print(f"\n处理完成: 成功 {success_count} 个, 失败 {fail_count} 个")
    
    # 如果有多个shapefile，尝试按名称匹配（简化：使用第一个shapefile）
    else:
        print(f"找到 {len(shp_files)} 个Shapefile，使用第一个Shapefile处理所有文件...")
        
        # 使用第一个shapefile（简化处理）
        shapefile_path = shp_files[0]
        
        print(f"使用 {num_workers} 个进程并行处理")
        print("使用gdalwarp命令行工具进行掩膜（最快方法）")
        
        # 准备多进程参数（使用gdalwarp，直接传递shapefile路径）
        process_args = [
            (tif_path, prefix, date_str, shapefile_path, output_folder)
            for tif_path, prefix, date_str in tif_files
        ]
        
        # 使用多进程处理
        success_count = 0
        fail_count = 0
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_file, process_args),
                total=len(process_args),
                desc="处理TIF文件"
            ))
        
        # 统计结果
        for success, tif_name, output_filename, error_msg in results:
            if success:
                success_count += 1
            else:
                fail_count += 1
                print(f"✗ 处理失败 {tif_name}: {error_msg}")
        
        print(f"\n处理完成: 成功 {success_count} 个, 失败 {fail_count} 个")
    
    print(f"\n处理完成！结果保存在: {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='使用Shapefile掩膜TIF文件工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python mask_sask.py \\
    --tif_folder /path/to/tif/folder \\
    --shapefile_folder /path/to/shapefile/folder \\
    --output_folder /path/to/output/folder

功能说明:
  - 只处理tile00的TIF文件
  - 只处理日期在2005-01-01到2015-12-31范围内的文件
  - 输出文件名格式: xxxx_yyyy_mm_dd.tif (去掉tile部分)
        """
    )
    
    parser.add_argument(
        '--tif_folder',
        type=str,
        required=True,
        help='TIF文件所在文件夹路径'
    )
    
    parser.add_argument(
        '--shapefile_folder',
        type=str,
        default='/mnt/storage/benchmark_datasets/MODIS/sask/Saskatchewan_boundary',
        help='Shapefile文件所在文件夹路径'
    )
    
    parser.add_argument(
        '--output_folder',
        type=str,
        required=True,
        help='输出文件夹路径'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='并行处理的进程数（默认：自动检测，I/O密集型任务使用较少进程）'
    )
    
    args = parser.parse_args()
    
    # 验证文件夹是否存在
    if not os.path.isdir(args.tif_folder):
        print(f"错误: TIF文件夹不存在: {args.tif_folder}")
        exit(1)
    
    if not os.path.isdir(args.shapefile_folder):
        print(f"错误: Shapefile文件夹不存在: {args.shapefile_folder}")
        exit(1)
    
    print("=" * 60)
    print("TIF文件Shapefile掩膜工具")
    print("=" * 60)
    print(f"TIF文件夹: {args.tif_folder}")
    print(f"Shapefile文件夹: {args.shapefile_folder}")
    print(f"输出文件夹: {args.output_folder}")
    if args.num_workers is not None:
        print(f"并行进程数: {args.num_workers}")
    print("=" * 60)
    
    batch_mask_tifs(args.tif_folder, args.shapefile_folder, args.output_folder, args.num_workers)