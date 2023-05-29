import torch
import numpy as np
import cv2
from pathlib import Path

from tqdm import tqdm
from regression_cam1 import build_cam1_model
from regression_cam2 import build_cam2_model


def build_model():
    """ build custom model """

    model = torch.hub.load("ultralytics/yolov5", "custom", path="../../weight/yolov5/best.pt")
    cam1_regression_model = build_cam1_model()
    cam2_regression_model = build_cam2_model()
    return model, cam1_regression_model, cam2_regression_model


def get_image_list(img_path: Path) -> list:
    image_list = img_path.glob("*")
    return image_list


def data_collect(
    model,
    cam1_regression_model,
    cam2_regression_model,
    img_path_list: list,
    cam_num: int,
    destination_img_path: Path,
    destination_label_path: Path
) -> None:
    cnt_img = 436
    cnt_cow = 0
    # convert path object to list
    img_path_list = list(img_path_list)
    for img_path in tqdm(img_path_list):
        cnt_cow = 0
        try:
            # read img
            img = cv2.imread(str(img_path))
            # img = cv2.resize(img, (640, 384))

            # To get individual cow info from img
            cow_info_list = model(img)
            cow_info_list = cow_info_list.xyxy[0].cpu().tolist() # from tensor to list
            for cow_info in cow_info_list:
                # setup threshold to select image
                if cow_info[4] < 0.4:
                    continue

                # xmin, ymin, xmax, ymax, conf, class, name
                xmin, ymin, xmax, ymax = cow_info[:4]
                middle_x = (xmin + xmax) / 2
                middle_y = (ymin + ymax) / 2

                # base on camera number to select regression model
                if cam_num == 1:
                    pred_list = cam1_regression_model.predict(np.array([[middle_x, middle_y]]))
                elif cam_num == 2:
                    pred_list = cam2_regression_model.predict(np.array([[middle_x, middle_y]]))

                real_x = pred_list[0][0]
                real_y = pred_list[0][1]

                # calculate bb's width and height
                width = xmax - xmin
                height = ymax - ymin

                # write info into txt
                txt_name = f"{destination_label_path}/cow_{cnt_img}_{cnt_cow}.txt"
                img_name = f"{destination_img_path}/cow_{cnt_img}_{cnt_cow}.png"
                info = f"{real_x},{real_y},{width},{height}"
                with open(txt_name, "w") as f:
                    f.write(info)

                # crop image
                cropped_img = img[int(ymin):int(ymax), int(xmin-15):int(xmax+10)]
                cropped_img = cv2.resize(cropped_img, (224, 224), interpolation=cv2.INTER_AREA)
                cv2.imwrite(img_name, cropped_img)

                cnt_cow += 1
        except:
            continue
        cnt_img += 1


if __name__ == "__main__":
    # config path
    raw_img_path = Path("./raw-data")
    destination_img_path = "./img"
    destination_label_path = "./label"

    # build / init model
    model, cam1_regression_model, cam2_regression_model = build_model()

    # start process
    raw_img_list = get_image_list(raw_img_path)
    data_collect(
        model=model,
        cam1_regression_model=cam1_regression_model,
        cam2_regression_model=cam2_regression_model,
        img_path_list=raw_img_list,
        cam_num=2,
        destination_img_path=destination_img_path,
        destination_label_path=destination_label_path
    )
