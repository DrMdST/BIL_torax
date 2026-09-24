imageType:
  Original: {}

featureClass:
  firstorder: {}
  shape2D: {}      # Для УЗИ используем только 2D признаки формы
  glcm: {}
  glrlm: {}
  glszm: {}
  gldm: {}
  ngtdm: {}

setting:
  label: 1                  # ID сегмента в .seg.nrrd (обычно 1)
  binWidth: 25              # Ширина бина для УЗИ
  resampledPixelSpacing: null # Сохраняем оригинальный размер пикселя из файла
  interpolator: sitkBSpline
  force2D: true             # КРИТИЧНО: заставляет PyRadiomics считать признаки как для 2D
  force2Ddimension: 0       # Ось, вдоль которой применяется 2D (ось Z)
  minimumROIDimensions: 2   # Разрешаем анализ даже если маска плоская
  verbose: true
