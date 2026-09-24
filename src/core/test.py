import sys
import csv
import SimpleITK as sitk
from radiomics import featureextractor
import os

print(f"Python version: {sys.version}")

def extract_radiomics(image_path: str, mask_path: str, params_path: str, output_csv: str):
    print(f"1. Загрузка основного изображения: {image_path}")
    # Читаем напрямую через SimpleITK. Это СОХРАНЯЕТ Spacing, Origin и Direction из .nrrd
    image = sitk.ReadImage(image_path)
    
    # Если изображение случайно оказалось многоканальным (RGB), извлекаем первый канал
    if image.GetNumberOfComponentsPerPixel() > 1:
        print("   ⚠️ Обнаружено многоканальное изображение, извлекаем первый канал (градации серого)...")
        image = sitk.VectorIndexSelectionCast(image, 0)

    print(f"2. Загрузка маски: {mask_path}")
    mask = sitk.ReadImage(mask_path)
    
    # КРИТИЧНО: PyRadiomics требует, чтобы маска была целочисленной (UInt8)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    # Проверка геометрии: если маска и изображение не совпадают по размеру/spacing, делаем ресемплинг
    if (image.GetOrigin() != mask.GetOrigin() or 
        image.GetSpacing() != mask.GetSpacing() or 
        image.GetDirection() != mask.GetDirection()):
        print("   ⚠️ Геометрия маски не совпадает с изображением. Выполняется автоматический ресемплинг маски...")
        mask = sitk.Resample(mask, image, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, mask.GetPixelID())

    print("3. Инициализация экстрактора и загрузка параметров...")
    extractor = featureextractor.RadiomicsFeatureExtractor(params_path)
    
    print("4. Извлечение радиомических признаков...")
    result = extractor.execute(image, mask)
    
    print("5. Сохранение результатов в CSV...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Feature', 'Value'])
        
        for key, val in result.items():
            # Фильтруем длинные диагностические строки для чистоты выходного файла
            if not key.startswith('diagnostics_'):
                print(f"\t{key}: {val}")
                writer.writerow([key, val])
                
    print(f"✅ Готово! Результаты сохранены в: {output_csv}")

if __name__ == "__main__":
    # Укажите ваши реальные пути к файлам
    # Для Colab используйте пути вида: "/content/1.nrrd"
    IMAGE_PATH = r"C:\Users\bil\1.nrrd"
    MASK_PATH = r"C:\Users\bil\1.seg.nrrd"
    PARAMS_PATH = r"params.yaml"
    OUTPUT_CSV = r"output_corrected.csv"
    
    if os.path.exists(IMAGE_PATH) and os.path.exists(MASK_PATH):
        extract_radiomics(IMAGE_PATH, MASK_PATH, PARAMS_PATH, OUTPUT_CSV)
    else:
        print("❌ Ошибка: Файлы не найдены. Проверьте пути.")