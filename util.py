import math
import torch
import cv2
import os
import numpy as np


# for test data path loading
def get_all_file_paths(directory):
    file_list = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_list.append(file_path)
    return file_list


def get_all_file_paths_asym(rgb_dir, gt_dir):
    rgb_file_list = []
    gt_numbers = []
    
    for filename in os.listdir(rgb_dir):
        file_path = os.path.join(rgb_dir, filename)
        if os.path.isfile(file_path):
            rgb_file_list.append(file_path)
        gt_number = filename.split("_")[0]
        gt_numbers.append(gt_number)  

    gt_file_list = []
    for filename in os.listdir(gt_dir):
        gt_id = filename.split(".")[0]
        match_count = gt_numbers.count(gt_id)  
        for _ in range(match_count): 
            file_path = os.path.join(gt_dir, filename)
            gt_file_list.append(file_path)

    return rgb_file_list, gt_file_list




# for input rgb loading
def rgb_loader(index, rgb_path, device):
    if 8 <= index <= 16:
                rgb = np.load(rgb_path)
    else:
        rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)  # 720*480*3 or 480*720*3 tensor

    if 5 <= index <= 7:
        rgb_tensor = torch.tensor(rgb, dtype=torch.float32, requires_grad=True).permute(2, 0, 1).to(device).unsqueeze(0) / 65535.0
    else:
        rgb_tensor = torch.tensor(rgb, dtype=torch.float32, requires_grad=True).permute(2, 0, 1).to(device).unsqueeze(0) / 255.0
    
    return rgb_tensor




# for applying TF
def extract_y(i):
    
    y = None
    if isinstance(i, np.ndarray):
        if len(i.shape) == 4:
            y = 0.299 * i[:, :, :, 2] + 0.587 * i[:, :, :, 1] + 0.114 * i[:, :, :, 0]
        elif len(i.shape) == 3:
            y = 0.299 * i[:, :, 2] + 0.587 * i[:, :, 1] + 0.114 * i[:, :, 0]

    elif isinstance(i, torch.Tensor):
        if i.ndim == 4:
            y = 0.299 * i[:, 2, :, :] + 0.587 * i[:, 1, :, :] + 0.114 * i[:, 0, :, :]
        elif i.ndim == 3:
            y = 0.299 * i[2, :, :] + 0.587 * i[1, :, :] + 0.114 * i[0, :, :]

    return y

def new_Y_gen(y_tensor, tf):

    y_tensor = y_tensor * 255.0
    b, _, h, w = y_tensor.shape
    u = torch.arange(0, 256, device=y_tensor.device).view(256, 1) 
    d = u - y_tensor.reshape(b, 1, -1) 
    weight = torch.clamp(1 - torch.abs(d), min=0)  
    weight = weight.permute(0, 2, 1)  
    tf = tf.unsqueeze(2) 
    new_y = torch.matmul(weight, tf).squeeze(2)  
    new_y = new_y.view(b, 1, h, w)

    return new_y / 255.0

def apply_tf(rgb, U, C):

    tf = (U @ C).T * 255.0
    y = extract_y(rgb).unsqueeze(1)
    new_y = new_Y_gen(y, tf)
    scale_Y = new_y / (y + 1e-6)  
    y_enhanced = rgb * scale_Y

    return y_enhanced, tf

def apply_tf_test(rgb, U, C):

    tf = (U @ C).T * 255.0 
    tf = torch.cummax(tf, dim=1)[0] 
    tf = torch.clamp(tf, min=0, max=255)
    y = extract_y(rgb).unsqueeze(1)
    new_y = new_Y_gen(y, tf)
    scale_Y = new_y / (y + 1e-6) 
    y_enhanced = rgb * scale_Y

    return y_enhanced




# for applying ccm
def apply_ccm(y_enhanced, ccm_pred):
    
    a, b, c, d, e, f = ccm_pred[:, 0], ccm_pred[:, 1], ccm_pred[:, 2], ccm_pred[:, 3], ccm_pred[:, 4], ccm_pred[:, 5]
    ccm = torch.stack([
        torch.stack([1 - a - b, a, b], dim=-1),
        torch.stack([c, 1 - c - d, d], dim=-1),
        torch.stack([e, f, 1 - e - f], dim=-1)
    ], dim=1)

    # a = ccm.transpose(1, 2)
    y_enhanced_reshape = y_enhanced.permute(0, 2, 3, 1).reshape(y_enhanced.shape[0], -1, 3)  
    pred = torch.bmm(y_enhanced_reshape, ccm.transpose(1, 2))
    pred = pred.reshape(y_enhanced.shape[0], y_enhanced.shape[2], y_enhanced.shape[3], 3).permute(0, 3, 1, 2) 

    return pred

def apply_ccm_test(y_enhanced, ccm_pred):
    
    a, b, c, d, e, f = ccm_pred[:, 0], ccm_pred[:, 1], ccm_pred[:, 2], ccm_pred[:, 3], ccm_pred[:, 4], ccm_pred[:, 5]
    ccm = torch.stack([
        torch.stack([1 - a - b, a, b], dim=-1),
        torch.stack([c, 1 - c - d, d], dim=-1),
        torch.stack([e, f, 1 - e - f], dim=-1)
    ], dim=1)

    y_enhanced_reshape = y_enhanced.permute(0, 2, 3, 1).reshape(y_enhanced.shape[0], -1, 3)
    pred = torch.bmm(y_enhanced_reshape, ccm.transpose(1, 2)) 
    pred = pred.reshape(y_enhanced.shape[0], y_enhanced.shape[2], y_enhanced.shape[3], 3).permute(0, 3, 1, 2) 
    pred = torch.clamp(pred, 0.0, 1.0)

    return pred




# for test result saving
def save_out_img(prefix, img, psnr, save_folder):
    dump(img, prefix, f"_psnr={psnr:.2f}", save_folder)

def dump(img, prefix, postfix, save_folder='output', order='bgr', bpp=8):
    img_file_name = f"{os.path.join(save_folder, prefix)}{postfix}.jpg"
    save_image(img_file_name, img, order, bpp)
    
def save_image(filename, rgb, order='bgr', bpp=8):
    depth = rgb.shape[-1]
    img = rgb / (2 ** (bpp - 8))
    if depth == 3:
        if order == 'rgb':
            img = img[..., ::-1].copy()

    cv2.imwrite(filename, img)




# for image filename 
def prefix_extract(index, name_only):
    
    prefix = ""
    if 0 <= index < 5:
        prefix = name_only.split("-")[0]
    elif 5 <= index < 8:
        prefix = name_only.split(".tif")[0]
    elif 8 <= index <= 13:
        prefix = name_only.split("_resized")[0]
    elif 14 <= index <= 16 :
        prefix = name_only.split(".npy")[0]
    elif index == 17 or 22 <=  index <= 25 or index == 27:
        prefix = name_only.split(".jpg")[0]
    elif 18 <= index <= 21 or index == 26:
        prefix = name_only.split(".png")[0]
    elif index == 28:
        prefix = name_only.split(".jpg")[0]
    elif index == 29:
        prefix = name_only.split(".JPG")[0]
        
    return prefix




# psnr, mse(for WB task)
def psnr_by_index(out, gt, index, prefix):
    
    psnr = 0
    if index == 28:
        metadata_dict = {}
        if index == 28:
            with open('/hdd1/jwkim/multistyle_all/28/metadata.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    file_name, metadata = line.strip().split(": ", 1)
                    metadata_dict[file_name] = float(metadata)
        
        area = metadata_dict.get(prefix)
        mse = cal_mse(out*255.0, gt*255.0, area)
        psnr = 10 * math.log10(65025.0 / (mse + 0.00001))
    elif index == 29: 
        mse = cal_mse(out*255.0, gt*255.0, 58373)
        psnr = 10 * math.log10(65025.0 / (mse + 0.00001))
    else: 
        psnr = cal_psnr(out, gt)
    
    return psnr

def cal_psnr(pred, gt):

    mse_mean = np.mean((pred - gt) ** 2)  # 0~1 scale
    psnr = 20 * math.log10(1.0 / math.sqrt(mse_mean + 0.00001))

    return psnr

def cal_mse(source, target, color_chart_area):
    source = np.reshape(source, [-1, 1]).astype(np.float64)
    target = np.reshape(target, [-1, 1]).astype(np.float64)
    mse = np.sum((source-target)**2)
    return mse / ((np.shape(source)[0]) - color_chart_area)














