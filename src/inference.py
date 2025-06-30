import torch
import torchvision.transforms.functional as F
from util import *
from tqdm import tqdm

from util import *
from model import *


def test(test_loader, test_num_list, num_style, u_mat, save_dir):

    # model setting
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = model_total(num_style)
    model = model.to(device)
    
    # load ckpt 
    ckpt_path = '/your_directory/epoch=80.pth'
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location='cuda:2')
    model.load_state_dict(checkpoint['model_state_dict'])

    # test per expert
    for index in range(29, 30):
        avg_psnr = test_model(model, device, test_loader, test_num_list, u_mat, index, save_dir)
        print(f'Expert_{index}__avg_psnr : %.2f' % (avg_psnr))



def test_model(model, device, test_loader, test_num_list, u_mat, index, save_dir):
    model.eval()
    total_psnr = 0
    
    with torch.no_grad():
        u_mat_tensor = torch.tensor(u_mat, dtype=torch.float32, requires_grad=True).to(device)  
        test_loader = test_loader[test_num_list[index] : test_num_list[index + 1]]
        
        for (rgb_path, gt_path) in tqdm(test_loader, total=len(test_loader), desc=f"Test Index {index}"):

            # prefix = image filename
            name_only = rgb_path.split("/")[7]
            prefix = prefix_extract(index, name_only)

            # input img & gt road
            rgb_tensor = rgb_loader(index, rgb_path, device)
            gt = cv2.imread(gt_path, cv2.IMREAD_COLOR) / 255.0
            
            # coef & ccm predicition
            rgb_network = F.resize(rgb_tensor, [224, 224])
            style_list = [index]
            a, b, coef_pred, ccm_pred, tf = model(rgb_network, style_list, u_mat_tensor)
            
            # tf & ccm apply
            y_enhan = apply_tf_test(rgb_tensor, u_mat_tensor, coef_pred)
            ccm_applied = apply_ccm_test(y_enhan, ccm_pred)
            out = ccm_applied.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

            psnr = psnr_by_index(out, gt, index, prefix)
            total_psnr += psnr
            
            save_folder = save_dir + f'{index}/'
            if not os.path.exists(save_folder):
                os.makedirs(save_folder, exist_ok=True)
            save_out_img(prefix, out*255, psnr, save_folder)

        # cal and return the average psnr
        avg_psnr = round(total_psnr / len(test_loader), 5)

    return avg_psnr





