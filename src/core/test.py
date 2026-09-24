import sys
import csv
import SimpleITK as sitk
from radiomics import featureextractor
import numpy as np
import nrrd
import os

print(f"Python version: {sys.version}")

def get2DSlice(image_3d, slice_index=None):
    """
    Извлекает 2D срез из 3D массива.
    Если slice_index не указан, берет средний срез.
    """
    if image_3d.ndim == 2:
        return image_3d
    
    if slice_index is None:
        slice_index = image_3d.shape[2] // 2
    
    return image_3d[:, :, slice_index]

def rgb_to_gray(rgb_array):
    """Конвертирует RGB изображение в оттенки серого"""
    if rgb_array.ndim == 3 and rgb_array.shape[2] == 3:
        return np.dot(rgb_array[..., :3], [0.2989, 0.5870, 0.1140])
    return rgb_array

def extract_radiomics_2d(image_path: str, mask_path: str, params_path: str, output_csv: str):
    print(f"1. Загрузка основного изображения: {image_path}")
    image, image_header = nrrd.read(image_path)
    print(f"   Исходная форма изображения: {image.shape}")
    
    # Если изображение 3D с каналами (RGB), берем 2D срез
    if image.ndim == 3:
        if image.shape[2] == 3:  # RGB
            print("   Обнаружено RGB изображение, конвертация в grayscale...")
            image_2d = rgb_to_gray(image)
        else:
            print(f"   Извлечение среднего 2D среза из 3D массива...")
            image_2d = get2DSlice(image)
    else:
        image_2d = image
    
    print(f"   Форма после обработки: {image_2d.shape}")
    
    print(f"2. Загрузка маски: {mask_path}")
    mask, mask_header = nrrd.read(mask_path)
    print(f"   Исходная форма маски: {mask.shape}")
    
    # Если маска 3D, берем 2D срез
    if mask.ndim == 3:
        mask_2d = get2DSlice(mask)
    else:
        mask_2d = mask
    
    # КРИТИЧНО: Выравнивание осей изображения и маски
    # Если оси перепутаны (например, изображение [636, 434], маска [434, 636])
    if image_2d.shape != mask_2d.shape:
        print(f"   ️ Формы не совпадают: изображение {image_2d.shape}, маска {mask_2d.shape}")
        print("   Выполняется транспонирование маски...")
        mask_2d = np.transpose(mask_2d)
        print(f"   Форма маски после транспонирования: {mask_2d.shape}")
    
    # Конвертация в SimpleITK для 2D анализа
    # Для 2D создаем массив с размерностью (1, height, width)
    image_2d_sitk = sitk.GetImageFromArray(image_2d.astype(np.float32)[np.newaxis, ...])
    mask_2d_sitk = sitk.GetImageFromArray(mask_2d.astype(np.uint8)[np.newaxis, ...])
    
    # Устанавливаем spacing для 2D (x, y, z)
    # Для УЗИ обычно используется spacing из метаданных или默认 1.0
    image_2d_sitk.SetSpacing((1.0, 1.0, 1.0))
    mask_2d_sitk.SetSpacing((1.0, 1.0, 1.0))
    
    print("3. Инициализация экстрактора для 2D анализа...")
    
    # Настройки для 2D УЗИ
    settings_2d = {
        'label': 1,
        'binWidth': 25,
        'resampledPixelSpacing': None,
        'interpolator': 'sitkBSpline',
        'force2D': True,              # КРИТИЧНО для 2D УЗИ
        'force2Ddimension': 0,        # Анализ по оси Z (которая у нас = 1)
        'verbose': True,
        'minimumROIDimensions': 2,    # Минимум 2D
    }
    
    extractor = featureextractor.RadiomicsFeatureExtractor(params_path, **settings_2d)
    
    print("4. Извлечение радиомических признаков (2D режим)...")
    result = extractor.execute(image_2d_sitk, mask_2d_sitk)
    
    print("5. Сохранение результатов в CSV...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Feature', 'Value'])
        
        for key, val in result.items():
            if not key.startswith('diagnostics_'):
                print(f"\t{key}: {val}")
                writer.writerow([key, val])
    
    print(f"✅ Готово! Результаты сохранены в: {output_csv}")
    return result

if __name__ == "__main__":
    IMAGE_PATH = r"C:\Users\bil\1.nrrd"
    MASK_PATH = r"C:\Users\bil\1.seg.nrrd"
    PARAMS_PATH = r"params.yaml"
    OUTPUT_CSV = r"output_2d.csv"
    
    if os.path.exists(IMAGE_PATH) and os.path.exists(MASK_PATH):
        extract_radiomics_2d(IMAGE_PATH, MASK_PATH, PARAMS_PATH, OUTPUT_CSV)
    else:
        print("❌ Ошибка: Файлы не найдены. Проверьте пути.")
