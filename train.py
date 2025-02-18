from ultralytics import YOLO

def main():
    # Load a pretrained model
    model = YOLO("yolo11n.pt")  # 加载预训练模型

    # 训练参数优化
    results = model.train(
        data="/kaggle/working/apexcode/data.yaml",
        epochs=50,              # 由于GPU资源有限，训练周期适当减少
        imgsz=512,              # 更小的图像尺寸，加速训练且适合T4 GPU
        batch=32,               # 减小批次大小以适应T4 GPU显存
        device='0,1',           # 使用双GPU（T4*2）
        workers=4,              # 设置合适的数据加载进程数，避免过多占用CPU资源
        amp=True,               # 启用混合精度训练以加速计算
        optimizer='AdamW',      # 使用AdamW优化器
        lr0=0.01,               # 初始学习率
        cache='disk',           # 使用磁盘缓存，避免占用过多内存
        single_cls=False,       # 如果是单类别分类可启用
        pretrained=True,        # 使用预训练权重
        close_mosaic=10,        # 最后10个epoch关闭Mosaic增强
        val=False,              # 关闭训练中验证
        save_period=10,         # 减少模型保存频率
        dropout=0.1,            # 启用Dropout防止过拟合
    )

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
