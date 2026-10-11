# Загрузка на GitHub (kb8-cloud), плоская структура

Все файлы лежат в одной папке. Загружаем всё содержимое папки в корень репозитория.

1. Распаковать `datagen_flat.zip`.
2. На GitHub: Add file → Upload files → перетащить ВСЁ содержимое папки (включая папку .github).
3. Commit changes.

Порядок запуска:
1. Actions → calib-openings → Run workflow. Скачать артефакт opening-pool-calibrated, загрузить opening_pool_calibrated.txt в корень репозитория.
2. Actions → datagen → Run workflow. Запомнить ID запуска из адреса.
3. Actions → tune-hce2 → datagen_run_id = этот ID. Скачать hce2_weights.txt, загрузить в корень как hce2_v1.txt.
4. Actions → match → a_source = build_2_dev.cpp, a_weights = hce2_v1.txt, b_source = build_1.2.cpp, b_weights пусто.
