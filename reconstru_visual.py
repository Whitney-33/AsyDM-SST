import torch
from torch import nn
import time
import numpy as np
import torch.utils.data as Data
import os
from torch.utils.data import TensorDataset
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
import deal_sst_util
import numpy as np
import imageio
import matplotlib.ticker as ticker
import matplotlib.pyplot as pyplot
from PIL import Image
import cv2
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.__version__)
print(torch.cuda.is_available())
print("using {} device.".format(device))
def create_file(train_type,batch_num, N_S_ratio,mask_type, cor_rate):
    if not os.path.exists('./picture'):
        os.mkdir('./picture')
    if os.path.exists('./picture/{}_{}_{}'.format(N_S_ratio,mask_type, cor_rate)):
        if os.path.exists('./picture/{}_{}_{}/{}'.format(N_S_ratio,mask_type, cor_rate,train_type)):
            if os.path.exists('./picture/{}_{}_{}/{}/batch_{}'.format(N_S_ratio,mask_type, cor_rate,train_type, batch_num)):
                return 1
            else:
                os.mkdir(r'./picture/{}_{}_{}/{}/batch_{}'.format(N_S_ratio,mask_type, cor_rate,train_type, batch_num))
        else:
            os.mkdir(r'./picture/{}_{}_{}/{}'.format(N_S_ratio,mask_type, cor_rate,train_type))
            os.mkdir(r'./picture/{}_{}_{}/{}/batch_{}'.format(N_S_ratio,mask_type, cor_rate, train_type, batch_num))
    else:
        os.mkdir(r'./picture/{}_{}_{}'.format(N_S_ratio,mask_type, cor_rate))
        os.mkdir(r'./picture/{}_{}_{}/{}'.format(N_S_ratio,mask_type, cor_rate, train_type))
        os.mkdir(r'./picture/{}_{}_{}/{}/batch_{}'.format(N_S_ratio,mask_type, cor_rate, train_type, batch_num))



def visua_and_save(train_type,epoch,batch_num,data_input, path, N_S_ratio, mask_type, cor_rate):
    create_file(train_type,batch_num,N_S_ratio,mask_type, cor_rate)
    data = data_input.cpu().detach().numpy()

    # print("data.shape",data.shape)
    mask = deal_sst_util.read_cache_all('./data/{}_{:.0f}'.format(mask_type,cor_rate) + '.h5') #mask 84 64 64
    mask = np.array(mask[0])
    for i in range(mask.shape[0]): 
        for j in range(mask.shape[1]):
            if mask[i][j] ==1:
                mask[i][j] = 0
            elif mask[i][j] ==0:
                mask[i][j] = 255
    cv2.imwrite("./mask_image.png", mask)


    for i in range(data.shape[0]):

        y = torch.FloatTensor(data[i]).to(device)
        mask = torch.FloatTensor(data[i] != 0).to(device)

        for j in range(y.shape[0]):
            if j== 7:
                filename= 'average'
            else:
                filename ='dayily'
            data2 = y[j]
            data2 = torch.squeeze(data2)
            data2 = torch.squeeze(data2).cpu().detach().numpy()
            if path == "miss0" or path == "miss1" or path == "miss2" or path == "miss3"  or \
                    path == "miss4" or path == "miss5" or path == "miss6" or \
                path == "valid_miss0" or path == "valid_miss1" or path == "valid_miss2" or path == "valid_miss3" or  \
                    path == "valid_miss4" or path == "valid_miss5" or path == "valid_miss6"  :
                mask = cv2.imread("./mask_image.png")
                mask_ = cv2.resize(mask, dsize=(data2.shape[0], data2.shape[1]), dst=None, fx=2, fy=2,
                                   interpolation=cv2.INTER_NEAREST)
                mask_ = mask_[:, :, 0]
                mask_ = mask_ > 220
                plt.figure()
                ax = sns.heatmap(data2, cmap='jet', square=True, mask=mask_, vmin=-1, vmax=1,
                                 annot_kws={"fontsize": 30})  
                plt.xlabel("°W", fontsize=30, style="normal", labelpad=-5.0, rotation=0, x=1.11)  # fontweight ='bold',
                plt.ylabel("°N", fontsize=30, style="normal", labelpad=-33.0, rotation=1, y=1.01)  # fontweight ='bold',

                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
                name_list = ('70', '69.5', '69', '68.5', '68', '67.5', '67')
                plt.xticks(np.arange(0, 65, 64 / 6), name_list,
                           rotation=0, fontsize=13)  
                name_list = ('26', '25.5', '25', '24.5', '24', '23.5', '23')
                plt.yticks(np.arange(0, 65, 64 / 6), name_list, fontsize=13)

                plt.savefig('./picture/{}_{}_{}/{}/batch_{}/'.format(N_S_ratio, mask_type, cor_rate,train_type,batch_num)+str(epoch)  +'-'+ path +'.png', dpi=300)
                plt.clf()
                plt.cla()
                plt.close("all")

                # plt.show()
            elif path== "valid_min_rmse_recons" or path== "valid_min_mse_recons" or path== "valid_min_mae_recons" or path== "valid_min_r2_recons"  :
                plt.figure()
                ax = sns.heatmap(data2, cmap='jet', square=True, vmin=-1, vmax=1,
                                 annot_kws={"fontsize": 30}) 
                plt.xlabel("°W", fontsize=30, style="normal", labelpad=-5.0, rotation=0, x=1.11)  # fontweight ='bold',
                plt.ylabel("°N", fontsize=30, style="normal", labelpad=-33.0, rotation=1, y=1.01)  # fontweight ='bold',

                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
                name_list = ('70', '69.5', '69', '68.5', '68', '67.5', '67')
                plt.xticks(np.arange(0, 65, 64 / 6), name_list,
                           rotation=0, fontsize=13) 
                name_list = ('26', '25.5', '25', '24.5', '24', '23.5', '23')
                plt.yticks(np.arange(0, 65, 64 / 6), name_list, fontsize=13)
                plt.savefig('./picture/{}_{}_{}/{}/batch_{}/'.format(N_S_ratio , mask_type, cor_rate,train_type,batch_num)+ str(epoch)+ '-'+ path + '.png', dpi=300)
                plt.clf()
                plt.cla()
                plt.close("all")

                # plt.show()
            else:
                plt.figure()
                ax = sns.heatmap(data2, cmap='jet', square=True, vmin=-1, vmax=1,
                                 annot_kws={"fontsize": 30}) 

                plt.xlabel("°W", fontsize=30, style="normal", labelpad=-5.0, rotation=0, x=1.11)  # fontweight ='bold',
                plt.ylabel("°N", fontsize=30, style="normal", labelpad=-33.0, rotation=1, y=1.01)  # fontweight ='bold',

                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
                name_list = ('70', '69.5', '69', '68.5', '68', '67.5', '67')
                plt.xticks(np.arange(0, 65, 64 / 6), name_list,
                           rotation=0, fontsize=13)  
                name_list = ('26', '25.5', '25', '24.5', '24', '23.5', '23')
                plt.yticks(np.arange(0, 65, 64 / 6), name_list, fontsize=13)
                plt.savefig('./picture/{}_{}_{}/{}/batch_{}/'.format(N_S_ratio, mask_type, cor_rate,train_type,batch_num)+str(epoch) +'-'+ path + '.png', dpi=300)

                plt.clf()
                plt.cla()
                plt.close("all")
                # plt.show()

    plt.close()
