'''
图像进行混沌加密解密，使用的加密图片为Corel-1k数据库。混沌加密后的数据是密文图像。
加密后的数据是存于encrypted_images_AES_Corel-1k文件夹中，
解密后的数据是存于decrypted_images_AES_Corel-1k文件夹中。
'''
import io
import os
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from PIL import Image

def encrypt_image(image_path, key):
    # 读取图片文件
    with open(image_path, 'rb') as file:
        image_data = file.read()
    # 创建 AES 加密对象
    cipher = AES.new(key, AES.MODE_EAX)
    # 加密图片数据
    ciphertext, tag = cipher.encrypt_and_digest(image_data)
    return (ciphertext, cipher.nonce, tag)

def decrypt_image(ciphertext, nonce, tag, key):
    # 创建 AES 解密对象
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    # 解密图片数据
    try:
        image_data = cipher.decrypt_and_verify(ciphertext, tag)
        return image_data
    except ValueError:
        print("Incorrect decryption")
        return None

def process_folder(input_folder, encrypted_folder, decrypted_folder, key):
    # 检查文件夹是否存在，不存在则创建
    if not os.path.exists(encrypted_folder):
        os.makedirs(encrypted_folder)
    if not os.path.exists(decrypted_folder):
        os.makedirs(decrypted_folder)
    # 遍历输入文件夹内的所有文件
    for filename in os.listdir(input_folder[:20]):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_image_path = os.path.join(input_folder, filename)
            encrypted_image_path = os.path.join(encrypted_folder, filename)
            decrypted_image_path = os.path.join(decrypted_folder, filename)
            # 加密图像
            encrypted_data = encrypt_image(input_image_path, key)
            if encrypted_data is not None:
                # 保存加密后的数据（这里可以选择保存为二进制文件）
                ciphertext, nonce, tag = encrypted_data
                with open(encrypted_image_path + '.enc', 'wb') as f:
                    f.write(nonce)
                    f.write(tag)
                    f.write(ciphertext)
                # 解密图像
                decrypted_data = decrypt_image(*encrypted_data, key)
                if decrypted_data is not None:
                    try:
                        image = Image.open(io.BytesIO(decrypted_data))
                        image.save(decrypted_image_path)
                        print(f"图像 {filename} 加密和解密完成。")
                    except Exception as e:
                        print(f"处理图像 {filename} 时出错: {e}")

if __name__ == '__main__':
    key = get_random_bytes(16)  # 生成图像加密密钥
    print("key", key)
    input_folder = "plainimages"
    encrypted_folder = "encrypted_images_AES_Corel-1k"
    decrypted_folder = "decrypted_images_AES_Corel-1k"
    process_folder(input_folder, encrypted_folder, decrypted_folder, key)
    print("文件夹内所有图像加密和解密完成。")

#71-19