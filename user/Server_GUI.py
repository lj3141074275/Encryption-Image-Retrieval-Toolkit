import time
import numpy as np
import argparse

def distance_IP(pa_en, pb_en, qa_en, qb_en):
    res1 = np.dot(qa_en, pa_en)
    res2 = np.dot(qb_en, pb_en)
    return res1 + res2

def re_pq_en(pa_en, pb_en, qa_en, qb_en, y_ac, y_test, topk):
    reeuc = np.zeros(len(pa_en))
    for i in range(len(pa_en)):
        dist = distance_IP(pa_en[i], pb_en[i], qa_en, qb_en)
        reeuc[i] = dist
    sort_re = np.argsort(reeuc)[::-1]
    sorted_arr = reeuc[sort_re]
    cou_ac = 0
    ap = 0.0
    for i in range(0, topk):
        if y_test[int(sort_re[i])] == y_ac:
            cou_ac = cou_ac + 1
            tt = cou_ac / (i + 1)
            ap = ap + tt
    if cou_ac > 0:
        ap = ap / cou_ac
    return cou_ac, ap, sort_re

def load_data():
    """
    加载加密特征数据和标签
    :return: 加载的数据
    """
    try:
        pa_en = np.load("datas/corel-1k_pa.npy")
        pb_en = np.load("datas/corel-1k_pb.npy")
        qa_en = np.load("datas/corel-1k_qa.npy")
        qb_en = np.load("datas/corel-1k_qb.npy")
        pca_dim = 128
        y_test = np.arange(10)
        y_test = np.repeat(y_test, 100)
        for i in range(1000):
            y_test[i] = y_test[i] + 1
        return pa_en, pb_en, qa_en, qb_en, pca_dim, y_test
    except FileNotFoundError:
        print("数据文件未找到，请检查文件路径。")
        return None, None, None, None, None, None

def main(query_count, want_top):
    """
    主函数，执行图像检索流程
    :param query_count: 想要查询图像的次数
    :param want_top: 想要排名前几的相似图像
    """
    # 输入图像 ID
    image_ids = []
    for i in range(query_count):
        while True:
            id_input = input(f"请输入第 {i + 1} 个图像的 ID: ")
            try:
                image_id = int(id_input)
                image_ids.append(image_id)
                break
            except ValueError:
                print("输入无效，请输入一个有效的整数。")
    if len(image_ids) == query_count:
        image_ids = np.array(image_ids)
        print("最终的图像 ID 数组为:", image_ids)
    else:
        print("由于输入无效，未成功生成完整的数组。")
        return
    # 加载数据
    pa_en, pb_en, qa_en, qb_en, pca_dim, y_test = load_data()
    if pa_en is None:
        return
    print("---------数据和标签预处理结束------------")

    start_time = time.time()
    cou_acc = 0
    topk = 10
    ap_acc = 0.0
    select = len(image_ids)
    for i in image_ids:
        try:
            cou, ap, sort = re_pq_en(pa_en, pb_en, qa_en[i], qb_en[i], y_test[i], y_test, topk)
            cou_acc = cou_acc + cou
            ap_acc = ap_acc + ap
            print("检索结果Top-10的图像ID是：", sort[:want_top])
        except IndexError:
            print(f"图像 ID {i} 超出数据范围，请检查输入。")
    end_time = time.time()
    elapsed_time = end_time - start_time
    if select > 0:
        ap_acc = ap_acc / select
        cou = cou_acc / (select * topk)
        print("-----------加密特征向量检索结束--------------特征维度：", pca_dim)
        print("加密特征：", len(y_test), "个，   检索特征：", int(select), "个.\n取top", topk, "得到检索准确率p@k=", cou)
        print("P @", topk, "=", cou)
        print("加密特征向量与加密查询向量--检索的花费时间：", elapsed_time, "秒")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图像检索程序")
    parser.add_argument("--query_count", type=int, help="想要查询图像的次数", default=1)
    parser.add_argument("--want_top", type=int, help="想要排名前几的相似图像", default=10)
    args = parser.parse_args()
    main(args.query_count, args.want_top)
# python Server_GUI.py --query_count 2 --want_top 10
