import os
from utils import build_vocab

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
basic_dir = 'XXX'

class Args:  # 每次对于文件地址的改变，只需要针对数据集的名称更改text_data_dir就好，即twitter2015或者twitter2017
    text_data_dir = os.path.join(basic_dir, 'data/twitter2015')  # 储存文本文件'train', 'dev', 'test'的文件夹
    img_path = os.path.join(text_data_dir, 'ner_img')  # 是一个文件夹地址，包含了所有图像，绝对路径
    obj_num = 5  # 每张图片中检测目标数量的上限
    img_feature_rcnn = os.path.join(text_data_dir, 'output/img_features/faster_rcnn.pt')  # 提取的(数据集的)图片的目标图像特征
    img_object_mrcnn = os.path.join(text_data_dir, 'output/img_features/object_label')
    img_object_label_num = 85
    img_coco_detectron_model = os.path.join(basic_dir, 'data_preprocess/img_prep')
    token_aug_vec_addr = os.path.join(text_data_dir, 'output/aug/tweet_aug.npy')
    # *******************************************************************************************************
    char_input_dim = 128
    max_wordlen = 12
    rcnn_input_dim = 2048

    epoch_num = 75

    resize = 1 # 不对错误图片使用resize

    #***************************************以下进行对比实验**********************************#
    use_char_cnn = 1 # 不使用cnn编码                                                                      (已被证实CNN无效)

    aug_before = 1 # 在编码后解码前，对token进行增强　　　　　　　　　　　　　　　　　　　　　　　　　　　(已证实为最佳方式，不再测试)
    # aug_before = 2 # 不使用任何增强

    aug_gate = 1 # aug_before = 1的情况下，进行简单的增强特征串联/相加；否则不起作用

    ffusion_method = 2 # 2表示特征融合的方式为门+串接　　　　　　　　　　　　　　　　　　　　　　　(已证实为最佳特征融合方式，不再测试)

    # *************************以下为对比实验需求**************************#
    # ffusion_method = 7 # 7表示只融合obj的特征                                                   (测试-resnet特征的效果)
    # ffusion_method = 8 # 8表示只融合全图的特征                                                 (测试-mrcnn_object的效果)

    main_cat = 0  # 最后的特征串联使用main

    alpha_seg = 1  # 不使用alpha特征变换

    seg_task = 0 # 添加原论文的辅助任务
    # seg_task = 1 # 不添加原论文的辅助任务                                                   (测试不加entity-segment任务的效果)

    added_output_dir = 'mode_' + str(obj_num)+str(epoch_num)+str(resize)+ str(use_char_cnn)+ str(aug_before) +str(aug_gate)+ str(ffusion_method)+str(main_cat)+str(alpha_seg)+str(seg_task)
    word_dict, char_dict = build_vocab(text_data_dir)  # build_vocab构造的词汇字典、字符字典

    aug_dim = 200
    token_aug_num = 4
