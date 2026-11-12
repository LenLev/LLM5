# RL для вызова функций

Модель: Qwen/Qwen3.5-2B. Только RL (GRPO + LoRA), без SFT.
Обучение: ToolACE. Проверка: исходный BFCL evaluator и отдельная валидация ToolACE.

## Файлы

- `prepare_data.py`: чтение ToolACE, первая пара диалога, разделение по диалогам, EDA.
- `rewards.py`: безопасный разбор вызовов и две награды; `python3 rewards.py` проверяет основные случаи.
- `train.py`: три независимых эксперимента, seed 42; параметры сохраняются в `runs/*/config.json`.
- `merge.py`: объединение LoRA с исходной моделью перед оценкой.
- `evaluate.py`, `bfcl.jsonl`: оригинальные файлы преподавателя, без изменений.
- `validate.py`: оценка на 128 отложенных примерах ToolACE.
- `logs/`: вывод команд. `runs/*/metrics.jsonl`: метрики каждого шага.
- `results/*/metrics.json`, `predictions.jsonl`: результаты BFCL.

## Воспроизведение

Запускать из этой папки. На сервере выбран **физический GPU 1**.
В контейнере он может называться cuda:0: это локальная нумерация единственной доступной карты.

```bash
sudo docker build -t rlvr-homework .
sudo docker run --rm -it --gpus device=1 --ipc=host \
  -v "$PWD":/workspace -v rlvr-hf-cache:/cache rlvr-homework
python3 prepare_data.py
python3 rewards.py
python3 evaluate.py Qwen/Qwen3.5-2B --output results/base
bash run_all.sh
bash run_validation.sh
python3 check_submission.py
```

## Наблюдение в VS Code

Откройте терминал в удалённой папке и выполните:

```bash
python3 watch.py
```

Все запуски начинают обучение с одной и той же ревизии исходной модели.
RL-1: точное совпадение, learning rate 1e-5.
RL-2: частичная награда, learning rate 1e-5.
RL-3: частичная награда, learning rate 3e-5.
Бюджет: 100 шагов каждого запуска; выбирается последний шаг, без подбора по BFCL.
Ревизии данных и модели сохранены в EDA и конфигурациях.

Полные логи можно смотреть через `tail -F logs/rl1.log` (или rl2/rl3).
`audit.py` проверяет отсутствие пересечений и сохраняет SHA256 исходных файлов.
`analyze.py` строит графики из результатов; `make_report.py` создаёт PDF (matplotlib и reportlab).
В `diagnostics/` сохранена прерванная техническая проверка парсера; она не входит в итоговые эксперименты.

## Результат

BFCL overall: исходная модель **68.93%**, RL-1 **68.80%**, RL-2 **71.47%**, RL-3 **75.33%**.
Лучший запуск — RL-3: **+6.40 п.п.** к исходной модели (один seed, 100 шагов).
Полные объединённые веса на ВМ: `/home/edu/rlhf_lena_20261112/checkpoints/rl3`.
Все три набора весов находятся в `checkpoints/`, адаптеры — в `runs/*/adapter/`.
Большие веса не включаются в архив для GitHub; код и настройки позволяют их воспроизвести.
