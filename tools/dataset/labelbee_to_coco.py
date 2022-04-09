import os
import json
import shutil
import argparse

'''
    将 labelbee多步标注 后的关键点数据进行标签转化，转至mmpose模型需要的coco格式。 数据集先划分好。
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default='/home/wanghao/Work/labelbee_json_2_MMPose_COCO_json/', help='输入labelbee json的根路径')
    parser.add_argument('--output-file', type=str, default='/home/wanghao/Work/labelbee_json_2_MMPose_COCO_json/', help='输出mmpose json文件根路径')
    parser.add_argument('--dataset-split', type=list, default=['train', 'val'], help='数据集划分，对应labelbee根路径下划分情况')
    args = parser.parse_args()


    data_set = args.dataset_split
    for n in range(len(data_set)):
        anno_path = args.input_file + data_set[n] + "/"
        anno_list  = os.listdir(anno_path)

        sort_anno_list = []
        for i in range(len(anno_list)):
            sort_anno_list.append(int(anno_list[i].split(".jpg")[0]))
        sort_anno_list = sorted(sort_anno_list)

        anno_list = []
        for i in range(len(sort_anno_list)):
            anno_list.append(str(sort_anno_list[i]) + ".jpg.json")
        print(anno_list)

        anno_info = {}
        anno_info["info"] = {"description":"For testing COCO dataset only.",
                             "year":2022,
                             "date_created":"2022/03/24"}

        anno_info["licenses"] = [{"url":"http://creativecommons.org/licenses/by-nc-sa/2.0/",
                                  "id":1,
                                  "name":"Attribution-NonCommercial-ShareAlike License"}]

        anno_info["categories"] = [{"supercategory":"person",
                                    "id":1,
                                    "name":"preson",
                                    "keypoints":["nose",
                                                 "left_eye",
                                                 "right_eye",
                                                 "left_ear",
                                                 "right_ear",
                                                 "left_shoulder",
                                                 "right_shoulder",
                                                 "left_elbow",
                                                 "right_elbow",
                                                 "left_wrist",
                                                 "right_wrist",
                                                 "left_hip",
                                                 "right_hip",
                                                 "left_knee",
                                                 "right_knee",
                                                 "left_ankle",
                                                 "right_ankle"],
                                    "skeleton":[[16, 14],
                                                [14, 12],
                                                [17, 15],
                                                [15, 13],
                                                [12, 13],
                                                [6, 12],
                                                [7, 13],
                                                [6, 7],
                                                [6, 8],
                                                [7, 9],
                                                [8, 10],
                                                [9, 11],
                                                [2, 3],
                                                [1, 2],
                                                [1, 3],
                                                [2, 4],
                                                [3, 5],
                                                [4, 6],
                                                [5, 7]]}]

        anno_info["images"] = []

        anno_info["annotations"] = []

        l = 1
        for i in range(len(anno_list)):
            with open(anno_path + anno_list[i], 'r') as f:
                current_json_content = json.load(f)
            f.close()
            img_anno = {
                        "license":1,
                        "file_name":anno_list[i].split(".json")[0],
                        "coco_url":"http://creativecommons.org/licenses/by-nc-sa/2.0/",
                        "height":current_json_content["height"],
                        "width":current_json_content["width"],
                        "date_captured":"2022/03/24",
                        "flickr_url":"http://creativecommons.org/licenses/by-nc-sa/2.0/",
                        "id": i + 1
                        }
            anno_info["images"].append(img_anno)

            for j in range(len(current_json_content["step_1"]["result"])):
                current_bbox = current_json_content["step_1"]["result"][j]

                # 凭借这个id在后面的关键点列表里寻找到对应框内的关键点
                bbox_id = current_bbox["id"]
                keypoints_list = []
                keypoints_num = 0
                area = int(current_bbox["width"]) * int(current_bbox["height"])

                current_json_content["step_2"]["result"]
                for k in range(len(current_json_content["step_2"]["result"])):
                    if current_json_content["step_2"]["result"][k]["sourceID"] == bbox_id:
                        current_bbox_keypoints = current_json_content["step_2"]["result"][k:k+17]
                        break

                for k in range(len(current_bbox_keypoints)):
                    if current_bbox_keypoints[k]["attribute"] == "1":
                        # 这个点 可视
                        keypoints_num = keypoints_num + 1
                        keypoints_list.append(int(current_bbox_keypoints[k]["x"]))
                        keypoints_list.append(int(current_bbox_keypoints[k]["y"]))
                        keypoints_list.append(2)
                    else:
                        # 这个点 不可视
                        keypoints_list.append(0)
                        keypoints_list.append(0)
                        keypoints_list.append(0)

                keypoints_anno = {
                                    "segmentation":[[]],
                                    "num_keypoints":keypoints_num,
                                    "area":area,
                                    "iscrowd":0,
                                    "keypoints":keypoints_list,
                                    "image_id": i + 1,
                                    "bbox":[int(current_bbox["x"]), int(current_bbox["y"]),
                                            int(current_bbox["width"]), int(current_bbox["height"])],
                                    "category_id":1,
                                    "id": l
                                  }
                l = l + 1
                anno_info["annotations"].append(keypoints_anno)

        with open(args.output_file + data_set[n] +  '.json', 'w') as f:
            f.write(json.dumps(anno_info, ensure_ascii=False, indent=4, separators=(',', ':')))
            f.close()

        print(str(data_set[n]) + "标签转化完成！！！！")


if __name__=='__main__':
    main()