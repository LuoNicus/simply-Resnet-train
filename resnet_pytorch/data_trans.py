import numpy as np
import cv2



def img_process(img, out_shape=(640, 480)):  # 图像转灰度并调整大小
    img = cv2.resize(src=img, dsize=out_shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape((out_shape[1], out_shape[0], 1))
    return img




def USM_Shapen(src, alpha=5, sigma=5):  # USM锐化图片
    blur_img = cv2.GaussianBlur(src, (0, 0), sigma)
    usm = cv2.addWeighted(src, (1 + alpha), blur_img, -alpha, 0)
    return usm


def noise_cancelling(src, h=8):  # 图像降噪并反相
    img = cv2.fastNlMeansDenoising(src, h=h)
    img = 255 - img
    return img




def data_trainser(path=".", output_path=".",img_shape=(640, 480), t_shape=3):  
    import os
    import yaml
    x_full = None  # [h, w, t]

    for k, i in enumerate(os.listdir(f"{path}\\images")):
        f_name = i[:-4]
        if i[-3:] in ["jpg", "png"]:
            try:  # 提取出labels
                with open(f"{path}\\labels\\{f_name}.yaml") as f:
                    y = yaml.load(f, Loader=yaml.SafeLoader)
            except:  # 无label，跳出循环
                print(f"缺失标签：{f_name}.yaml")
                continue


            try:
                os.mkdir(f"{output_path}")
                os.mkdir(f"{output_path}\\labels")
                os.mkdir(f"{output_path}\\images")
            except:
                pass


            with open(f"{output_path}\\labels\\{f_name}.yaml", "w") as f:
                yaml.dump(y, f)

            img = cv2.imread(f"{path}\\images\\{i}")  # 读取图像 [w, h, c]
            # img = USM_Shapen(img, alpha=5, sigma=5)
            # img = noise_cancelling(img, h=8)
            img = img_process(img, out_shape=img_shape)  # 转为灰度单通道图片 [w, h]


            x = np.array(img)  # 转为数组
            # 每一个x数组包含t_shape张图片，是时域上连续的一系列图片
            if x_full is None:
                x_full = np.repeat(x, t_shape, axis=2)  # 初始时，在时域通道保留相同的图像               
                cv2.imwrite(f"{output_path}\\images\\{i}", x_full)
            else:
                x_full = np.append(x_full, x, axis=2)  # 将新图片存在数组的末尾
                x_full = np.delete(x_full, obj=0, axis=2)  # 删除数组的第一张图片
                cv2.imwrite(f"{output_path}\\images\\{i}", x_full)




if __name__ == "__main__":
    paras = {
        "batch_size": 5,
        "img_shape": (320, 240),  # w, h
        # "img_shape": (80, 80),  # w, h
        "t_shape": 1,
    }
    import os
    try:
        os.mkdir(f".\\data_deal")
    except:
        pass
    for k, i in enumerate(os.listdir(f".\\datas")):
        print(f"正在处理：.\\data\\{i}")
        data_trainser(f".\\datas\\{i}", f".\\data_deal\\{i}", 
            img_shape=paras["img_shape"], t_shape=paras["t_shape"])

