import sys
import csv
import SimpleITK as sitk
from radiomics import featureextractor
import os

print(f"Python version: {sys.version}")

def extract_radiomics_2d(image_path: str, mask_path: str, params_path: str, output_csv: str):
    print(f"1. Загрузка изображения: {image_path}")
    # SimpleITK читает .nrrd напрямую, сохраняя правильные оси (X, Y) и Spacing
    image = sitk.ReadImage(image_path)
    print(f"   Размер (SimpleITK): {image.GetSize()}")
    print(f"   Spacing: {image.GetSpacing()}")

    print(f"2. Загрузка маски: {mask_path}")
    mask = sitk.ReadImage(mask_path)
    print(f"   Размер (SimpleITK): {mask.GetSize()}")
    
    # КРИТИЧНО: PyRadiomics требует, чтобы маска была целочисленной (UInt8)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    # Проверка на совпадение размеров. Если не совпадают - ресемплим маску под изображение
    if image.GetSize() != mask.GetSize():
        print(f"   ⚠️ Размеры не совпадают! Выполняется ресемплинг маски...")
        mask = sitk.Resample(mask, image, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, mask.GetPixelID())
        print(f"   Размер маски после ресемплинга: {mask.GetSize()}")

    print("3. Инициализация экстрактора...")
    # Передаем ТОЛЬКО путь к YAML. Все настройки (включая force2D) теперь внутри него!
    # Это предотвращает конфликты настроек.
    extractor = featureextractor.RadiomicsFeatureExtractor(params_path)
    
    print("4. Извлечение радиомических признаков (2D режим)...")
    result = extractor.execute(image, mask)
    
    print("5. Сохранение результатов в CSV...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Feature', 'Value'])
        
        for key, val in result.items():
            # Фильтруем служебные диагностические сообщения для чистоты файла
            if not key.startswith('diagnostics_'):
                print(f"\t{key}: {val}")
                writer.writerow([key, val])
                
    print(f"✅ Готово! Результаты сохранены в: {output_csv}")

if __name__ == "__main__":
    # Убедитесь, что пути к файлам верные!
    IMAGE_PATH = r"C:\Users\bil\1.nrrd"
    MASK_PATH = r"C:\Users\bil\1.seg.nrrd"
    PARAMS_PATH = r"params.yaml"
    OUTPUT_CSV = r"output_2d.csv"
    
    if os.path.exists(IMAGE_PATH) and os.path.exists(MASK_PATH):
        extract_radiomics_2d(IMAGE_PATH, MASK_PATH, PARAMS_PATH, OUTPUT_CSV)
    else:
        print("❌ Ошибка: Файлы не найдены. Проверьте пути:")
        print(f"   Изображение: {os.path.exists(IMAGE_PATH)}")
        print(f"   Маска: {os.path.exists(MASK_PATH)}")
        print(f"   Params: {os.path.exists(PARAMS_PATH)}")
