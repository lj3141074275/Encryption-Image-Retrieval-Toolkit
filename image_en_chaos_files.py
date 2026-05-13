'''
图像进行混沌加密解密，使用的加密图片为Corel-1k数据库。混沌加密后的数据是密文图像。
加密后的数据是存于encrypted_images_chaos_Corel-1k文件夹中，
解密后的数据是存于decrypted_images_chaos_Corel-1k文件夹中。
'''
import numpy as np
import os
import cv2

def logistic_map(x, r, length):
    """
    生成Logistic映射序列
    :param x: 初始值           :param r: 参数
    :param length: 序列长度    :return: 生成的序列
    """
    sequence = []
    for _ in range(length):
        x = r * x * (1 - x)
        sequence.append(x)
    return np.array(sequence)

def encrypt_image(image_path, output_path):
    """
    对彩色图像进行混沌加密并保存
    :param image_path: 输入图像的路径      :param output_path: 加密后图像的保存路径
    """
    # 读取彩色图像
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"无法读取图像: {image_path}，请检查文件路径和文件完整性。")
        return None
    height, width, channels = image.shape
    # Logistic映射参数
    x0 = 0.2
    r = 3.9
    length = height * width * channels
    # 生成混沌序列
    chaotic_sequence = logistic_map(x0, r, length)
    chaotic_sequence = (chaotic_sequence * 255).astype(np.uint8).reshape(height, width, channels)
    # 加密图像
    encrypted_image = cv2.bitwise_xor(image, chaotic_sequence)
    # 保存加密后的图像，使用无损格式 PNG
    cv2.imwrite(output_path.replace('.jpg', '.png'), encrypted_image)
    return chaotic_sequence

def decrypt_image(encrypted_image_path, output_path, chaotic_sequence):
    """
    对加密后的彩色图像进行解密并保存 :param encrypted_image_path: 加密图像的路径
    :param output_path: 解密后图像的保存路径  :param chaotic_sequence: 加密时使用的混沌序列
    """
    # 读取加密后的彩色图像
    encrypted_image = cv2.imread(encrypted_image_path.replace('.jpg', '.png'), cv2.IMREAD_COLOR)
    if encrypted_image is None:
        print(f"无法读取加密图像: {encrypted_image_path}，请检查文件路径和文件完整性。")
        return
    # 解密图像
    decrypted_image = cv2.bitwise_xor(encrypted_image, chaotic_sequence)
    # 检查像素值范围并裁剪
    decrypted_image = np.clip(decrypted_image, 0, 255).astype(np.uint8)
    # 保存解密后的图像，使用无损格式 PNG
    cv2.imwrite(output_path.replace('.jpg', '.png'), decrypted_image)

def process_folder(input_folder, encrypted_folder, decrypted_folder):
    """
    处理文件夹内的所有图像，进行加密和解密操作   :param input_folder: 输入图像文件夹路径
    :param encrypted_folder: 加密后图像保存文件夹路径   :param decrypted_folder: 解密后图像保存文件夹路径
    """
    # 检查文件夹是否存在，不存在则创建
    if not os.path.exists(encrypted_folder):
        os.makedirs(encrypted_folder)
    if not os.path.exists(decrypted_folder):
        os.makedirs(decrypted_folder)
    # 遍历输入文件夹内的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_image_path = os.path.join(input_folder, filename)
            encrypted_image_path = os.path.join(encrypted_folder, filename)
            decrypted_image_path = os.path.join(decrypted_folder, filename)
            # 加密图像
            chaotic_sequence = encrypt_image(input_image_path, encrypted_image_path)
            if chaotic_sequence is not None:
                # 解密图像
                decrypt_image(encrypted_image_path, decrypted_image_path, chaotic_sequence)
                print(f"图像 {filename} 加密和解密完成。")

if __name__ == "__main__":
    input_folder = "plainimages"
    encrypted_folder = "encrypted_images_chaos_Corel-1k"
    decrypted_folder = "decrypted_images_chaos_Corel-1k"
    process_folder(input_folder, encrypted_folder, decrypted_folder)
    print("文件夹内所有图像加密和解密完成。")
#92-39