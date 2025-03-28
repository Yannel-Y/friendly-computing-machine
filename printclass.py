import os
from collections import defaultdict

def count_classes_in_yolo_labels(folder_path):
    # 初始化一个字典来统计每个类别的数量
    class_count = defaultdict(int)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                for line in file:
                    # 提取类别编号（每行的第一个值）
                    class_id = int(line.strip().split()[0])
                    class_count[class_id] += 1

    return class_count

def main():
    # 指定包含 YOLO 标注文件的文件夹路径
    folder_path = "/root/autodl-tmp/sqwu/yolo/yolov8/data/labels/val"  # 替换为你的文件夹路径

    # 统计类别数量
    class_count = count_classes_in_yolo_labels(folder_path)

    # 输出结果
    print("类别统计结果：")
    for class_id, count in sorted(class_count.items()):
        print(f"类别 {class_id}: {count} 个实例")

if __name__ == "__main__":
    main()