import torchvision.transforms as transforms
from dataset.cocodataset import CoCoDataset
from utils.cutout import SLCutoutPIL
from torchvision.transforms.autoaugment import AutoAugment
import os.path as osp
# 引入 ResNet50 仓库的数据集类
from utilities.mimic import mimic
from utilities.nih import nihchest
def get_datasets(args):
    #对输入数据标准化的两种方式。如果开启它，模型输入的是[0,1]范围内的数据。如果关闭它模型输入的就是经过imageNet统计数据标准化后的数据
    if args.orid_norm:
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])
        # print("mean=[0, 0, 0], std=[1, 1, 1]")
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        # print("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

    train_data_transform_list = [transforms.Resize((args.img_size, args.img_size)),
                                            AutoAugment(),
                                               transforms.ToTensor(),
                                               normalize]
    try:
        # for q2l_infer scripts
        if args.cutout:
            print("Using Cutout!!!")
            train_data_transform_list.insert(1, SLCutoutPIL(n_holes=args.n_holes, length=args.length))
    except Exception as e:
        Warning(e)
    train_data_transform = transforms.Compose(train_data_transform_list)

    test_data_transform = transforms.Compose([
                                            transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor(),
                                            normalize])
    

    if args.dataname == 'coco' or args.dataname == 'coco14':
        # ! config your data path here.
        dataset_dir = args.dataset_dir
        train_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'train2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_train2014.json'),
            input_transform=train_data_transform,
            labels_path='data/coco/train_label_vectors_coco14.npy',
        )
        val_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'val2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_val2014.json'),
            input_transform=test_data_transform,
            labels_path='data/coco/val_label_vectors_coco14.npy',
        )
    # 新增 MIMIC 支持
    elif args.dataname == 'mimic':
        train_dataset = mimic(root=args.dataset_dir, mode='train', transform=train_data_transform)
        if args.evaluate:
            print("!!! [Evaluation Mode] Loading TEST dataset (cxr14_test.csv) !!!")
            # 如果是 -e 模式，让 val_dataset 实际上加载测试集
            val_dataset = mimic(root=args.dataset_dir, mode='test', transform=test_data_transform)
        else:
            # 如果是训练模式，加载正常的验证集
            val_dataset = mimic(root=args.dataset_dir, mode='valid', transform=test_data_transform)
        args.num_class = train_dataset.get_number_classes() # 自动设置类别数 (13)
    # 新增 NIH 支持
    elif args.dataname == 'nih':
        train_dataset = nihchest(root=args.dataset_dir, mode='train', transform=train_data_transform)
        if args.evaluate:
            print("!!! [Evaluation Mode] Loading TEST dataset (cxr14_test.csv) !!!")
            # 如果是 -e 模式，让 val_dataset 实际上加载测试集
            val_dataset = nihchest(root=args.dataset_dir, mode='test', transform=test_data_transform)
        else:
            # 如果是训练模式，加载正常的验证集
            val_dataset = nihchest(root=args.dataset_dir, mode='valid', transform=test_data_transform)
        args.num_class = train_dataset.get_number_classes() # 自动设置类别数 (14)
    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)

    print("len(train_dataset):", len(train_dataset)) 
    print("len(val_dataset):", len(val_dataset))
    return train_dataset, val_dataset
