# -*- coding: utf-8 -*-
"""
第2章SSDで予測結果を画像として描画するクラス

"""
import numpy as np
import matplotlib.pyplot as plt 
import cv2  # OpenCVライブラリ
import torch

from .ssd_model import Panorama_DataTransform


class SSDPredictShow_split8():
    """SSDでの予測と画像の表示をまとめて行うクラス"""

    def __init__(self, eval_categories, net):
        self.eval_categories = eval_categories  # クラス名
        self.net = net  # SSDネットワーク

        color_mean = (104, 117, 123)  # (BGR)の色の平均値
        input_size = 300  # 画像のinputサイズを300×300にする
        self.transform = Panorama_DataTransform(color_mean)  # 前処理クラス
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def show(self, img, data_confidence_level):
        """
        物体検出の予測結果を表示をする関数。

        Parameters
        ----------
        image_file_path:  str
            画像のファイルパス
        data_confidence_level: float
            予測で発見とする確信度の閾値

        Returns
        -------
        なし。rgb_imgに物体検出結果が加わった画像が表示される。
        """
        rgb_img, predict_bbox, pre_dict_label_index, scores = self.ssd_predict(
            img, data_confidence_level)
        
        #psudo apt-get install ros-indigo-video-stream-opencirint(rgb_img, predict_bbox, pre_dict_label_index, scores)

        return rgb_img, predict_bbox, pre_dict_label_index, scores
        #self.vis_bbox(rgb_img, bbox=predict_bbox, label_index=pre_dict_label_index,
        #              scores=scores, label_names=self.eval_categories)

    def ssd_predict(self, img, only_front=False, data_confidence_level=0.5):
        """
        SSDで予測させる関数。

        Parameters
        ----------
        image_file_path:  strt
            画像のファイルパス

        dataconfidence_level: float
            予測で発見とする確信度の閾値

        Returns
        -------
        rgb_img, true_bbox, true_label_index, predict_bbox, pre_dict_label_index, scores
        """

        # rgbの画像データを取得
        #img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]
        # cv2.imshow("image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        height, width, channels = img.shape  # 画像のサイズを取得
        # print("img is", height, width)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 画像の前処理
        phase = "val"
        img_transformed, boxes, labels = self.transform(
            img, phase, "", "")  # アノテーションが存在しないので""にする。
        img = torch.from_numpy(
            img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)

        # SSDで予測
        self.net.eval()  # ネットワークを推論モードへ
        x = img.unsqueeze(0)  # ミニバッチ化：torch.Size([1, 3, 300, 300])
        x = x.to(self.device)
        self.net = self.net.to(self.device)

        # print('img is', x.is_cuda)
        # print('net is ', next(self.net.parameters()).is_cuda)

        detections = self.net(x)
        #print(detections)
        # detectionsの形は、torch.Size([1, 21, 200, 5])  ※200はtop_kの値

        # confidence_levelが基準以上を取り出す
        predict_bbox = []
        pre_dict_label_index = []
        scores = []
        detections = detections.cpu().detach().numpy()

        # 条件以上の値を抽出
        find_index = np.where(detections[:, 0:, :, 0] >= data_confidence_level)
        #print(find_index)
        detections = detections[find_index]
        #print("detection is", detections)
        for i in range(len(find_index[1])):  # 抽出した物体数分ループを回す
            if (find_index[1][i]) > 0:  # 背景クラスでないもの
                sc = detections[i][0]  # 確信度
                bbox = detections[i][1:] * [width, height, width, height]
                # find_indexはミニバッチ数、クラス、topのtuple
                lable_ind = find_index[1][i]-1
                # （注釈）
                # 背景クラスが0なので1を引く

                # 返り値のリストに追加
                predict_bbox.append(bbox)
                pre_dict_label_index.append(lable_ind)
                scores.append(sc)
        #print(predict_bbox)
        return rgb_img, predict_bbox, pre_dict_label_index, scores

    def ssd_panorama_predict(self, img, only_front=True , data_confidence_level=0.5):
        """
        SSDで予測させる関数。

        Parameters
        ----------
        image_file_path:  strt
            画像のファイルパス

        dataconfidence_level: float
            予測で発見とする確信度の閾値

        Returns
        -------
        rgb_img, true_bbox, true_label_index, predict_bbox, pre_dict_label_index, scores
        """

        # rgbの画像データを取得
        height, width, channels = img.shape  # 画像のサイズを取得
        img = cv2.resize(img, (300*4, 300))
        ori_img = img
        #print(np.shape(img))
        #cv2.namedWindow("Image")
        #cv2.imshow("image", img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # 画像の前処理
        phase = "val"
        img = np.array(self.transform(img, phase))[0]

        if only_front == True:
            img_front1 = img[:, np.int(1 * np.shape(img)[1]/4) : np.int(2 * np.shape(img)[1]/4), :]
            img_front2 = img[:, np.int(2 * np.shape(img)[1]/4) : np.int(3 * np.shape(img)[1]/4), :]
            img = np.stack([img_front1, img_front2])
            #cv2.namedWindow("Image")
            #cv2.imshow("image", img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #img = img.reshape(2, np.shape(img)[0], int(np.shape(img)[1]/2), -1)
            #cv2.namedWindow("Image")
            #cv2.imshow("image", img[0])
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

        else:
            img = np.split(img, [300, 300*2, 300*3], 1)
            img = np.stack(img)


        img = torch.from_numpy(img[:, :, :, (2, 1, 0)]).permute(0, 3, 1, 2)
        #print("imgis ", img)
        #print("img is ", np.shape(img))
        # SSDで予測
        self.net.eval()  # ネットワークを推論モードへ

        img = img.to(self.device)
        self.net = self.net.to(self.device)

        # print('img is', img.is_cuda)
        # print('net is ', next(self.net.parameters()).is_cuda)

        detections_batch = self.net(img)
        #print(np.shape(detections))
        # detectionsの形は、torch.Size([1, 21, 200, 5])  ※200はtop_kの値

        detections_batch = detections_batch.cpu().detach().numpy()
        total_predict_bbox = []

        for detections in detections_batch:
            #print(detections)
            detections = detections[np.newaxis, :, :, :]
            # confidence_levelが基準以上を取り出す
            predict_bbox = []
            pre_dict_label_index = []
            scores = []
            #print("detections is ", np.shape(detections))
            # 条件以上の値を抽出
            find_index = np.where(detections[:, 0:, :, 0] >= data_confidence_level)
            #print(find_index)
            detections_true = detections[find_index]
            #print("detection_true is", np.shape(detections))
            for i in range(len(find_index[1])):  # 抽出した物体数分ループを回す
                if (find_index[1][i]) > 0:  # 背景クラスでないもの
                    sc = detections_true[i][0]  # 確信度
                    bbox = detections_true[i][1:] * [300, 300, 300, 300] # width, high, width, high
                    # find_indexはミニバッチ数、クラス、topのtuple
                    lable_ind = find_index[1][i]-1
                    # （注釈）
                    # 背景クラスが0なので1を引く

                    # 返り値のリストに追加
                    predict_bbox.append(bbox)
                    #print(predict_bbox)

                    pre_dict_label_index.append(lable_ind)
                    scores.append(sc)
            #print(predict_bbox)
            total_predict_bbox.append(predict_bbox)
        return ori_img, total_predict_bbox, pre_dict_label_index, scores

    def vis_bbox(self, rgb_img, bbox, label_index, scores, label_names):
        """
        物体検出の予測結果を画像で表示させる関数。

        Parameters
        ----------
        rgb_img:rgbの画像
            対象の画像データ
        bbox: list
            物体のBBoxのリスト
        label_index: list
            物体のラベルへのインデックス
        scores: list
            物体の確信度。
        label_names: list
            ラベル名の配列

        Returns
        -------
        なし。rgb_imgに物体検出結果が加わった画像が表示される。
        """

        # 枠の色の設定
        num_classes = len(label_names)  # クラス数（背景のぞく）
        colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

        # 画像の表示
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_img)
        currentAxis = plt.gca()

        # BBox分のループ
        for i, bb in enumerate(bbox):

            # ラベル名
            label_name = label_names[label_index[i]]
            color = colors[label_index[i]]  # クラスごとに別の色の枠を与える

            # 枠につけるラベル　例：person;0.72　
            if scores is not None:
                sc = scores[i]
                display_txt = '%s: %.2f' % (label_name, sc)
            else:
                display_txt = '%s: ans' % (label_name)

            # 枠の座標
            xy = (bb[0], bb[1])
            width = bb[2] - bb[0]
            height = bb[3] - bb[1]

            # 長方形を描画する
            currentAxis.add_patch(plt.Rectangle(
                xy, width, height, fill=False, edgecolor=color, linewidth=2))

            # 長方形の枠の左上にラベルを描画する
            currentAxis.text(xy[0], xy[1], display_txt, bbox={
                             'facecolor': color, 'alpha': 0.5})
        
