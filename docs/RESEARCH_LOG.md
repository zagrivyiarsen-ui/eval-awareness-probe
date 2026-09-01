
---

## 7. 27.08 — ревью R-Gate0 и правки до пилота (см. `docs/REVIEW_R-Gate0_2026-08-27.md`, `docs/PREREG_ADDENDUM_2026-08-27.md`)

Вердикт ревью: пилот не запускать; блокеры А-3/Б-4/В-1/Г-3, все без GPU. Принято целиком, три уточнения: random band остаётся
репортом (список Нанды), ворота — перестановочная полоса; S-parse-probe — только `parse_ok`, без текста; дедлайн D = 04.09 (продление
не подтверждено). Сделано:
- **D2**: деконструкция sandbagging применена (`make_dps.py`); `dps_sha256` = новый; предсказание E2-AUC(last) ≥ 0.85 записано.
- **D1**: e2_c (user-suffix, перефразировка) — `conditions_hash` меняется один раз.
- `configs/analysis.yaml` (→ `analysis_hash`): скоринг, вербализация, PC-промпты, T-правило, правило урезания, слой, обрыв, джойн.
- Код: `src/scoring.py`, `src/analysis.py`, `probe_toy.perm_band/cross_env_auc`, Г-2 (sampling → числа), Г-4 (скан 1024 токенов,
  чередование сред), бюджетная формула «батч до cap», `scripts/gate0_fp16_checks.py` (Г-1, Г-5), `scripts/pilot.py` (резюмируемый),
  `scripts/gate1_checks.py` (12 проверок + механическое решение), `scripts/fill_gate_params.py` (одна правка config_hash),
  `verify_prereg.py` (+actions token-identity, утечка ключей, e2_c, analysis/conditions/code hash, change log). Тесты: +43.
- Бюджет (ревью В, tok_s_eff≈75, батч до cap): до 4.09 после пилота (~8 ч) остаётся ~35 GPU-ч → k=10 на двух средах при cap 3072,
  либо cap по пилоту. Язык §4.5: headline — пул n=32 (MDE≈0.10), по-средовые дельты — наблюдения (MDE≈0.14); H-add/пропорция/
  «спроси модель» — по строке.
Открыто до пилота: 0. Следующий запуск — `kaggle_cells/step4_apply_and_pilot_kaggle.py`.

## 8. 28.08 — ревью R-Pipeline (ячейки B1–B8) и правки до первого GPU-запуска
Блокеры: `pilot.py` NameError `P` (падение после загрузки модели), заполненный yaml не переживал ячейки (main_gen падал, B7 резал по cap 2048),
B5/B6 `--conds` ломал равный k. Сделано: см. addendum «Поправка 28.08». Ячейки пересобраны из одного репозитория; B6 упразднён (B5 — все условия,
повторный запуск продолжает); B1 разделён на validate (до пилота) и B1b main (после B4); тесты B1 — все, включая test_probe_stage.

## 9. 28.08 вечер — ревью R-Launch (перед B1→B2) и правки
Блокер №0 снят (ячейки == репозиторий, blob b4befc54374e). Блокеров 5, все 0 GPU: (1) правила k<6 не было; (2) парсер
secret_number пропускал cheat после вводной строки/&& → вторичная cheat_any; (3) recut_to_cap не обрезал длину →
смещение cot_len по условиям; (4) захардкоженные today/hours в B3/B4; (5) дрейф версии transformers между сессиями
(assert в main_gen тавтологичен после verify) → пин по numbers. Плюс: env-ассерт в анализе, contingency 10 ч в k-rule,
правило agreement<0.85, язык H-dissoc, perm-полоса на фигурах, presence_penalty в fill/reuse, --refill, blob_sha через
record_numbers, qualitative n=5, pytest без -x. Всё — до первой генерации; новый blob/analysis_hash в change log.
Прогон скоринга на 22 синтетических ответах и пересчёт бюджета руками — в docs/REVIEW_R-Launch (вне блоба) и здесь.

## 10. 28.08 ночь — ревью R2-Reject (адверсариальное) и партия правок до B2
NO-GO снят одной партией: дедлайн РЕШЁН (D=04.09, продление не используется); confirmatory-семейство из 4 тестов
с Holm (inference в analysis.yaml, счёт в join_arms); kill-switches key-shuffle (≈0.27 на синтетике) и shuffled-join
null; паттерн 6 вербализации удалён (не self-referential); κ-оговорка к обоим 0.85; pc_escalated — после завершения
strong-группы; сиды и FORCE_ENV оставляют deviation-след; tests/test_rehearsal_chain.py в репо (уже поймал 2 дефекта);
docs/SOURCE_BASELINES.md — заполнить частоты Singh/R&N ДО B2. Пороги не менялись. Таблица порогов, критерии Нанды и
списки проверок — в REVIEW_R2 (вне блоба). Осталось руками: частоты источников, скриншот квоты, запуск B1→B2.

## 11. 28.08 поздний вечер — запуск. Gate 0 перегнан (бэкап 27.08 утрачен); B1-тесты нашли fixture-баг
Gate 0 re-run: fp16 bs=8, 34.4 tok/s на cap 4096 (было 32.2); bf16 ~14 tok/s. fp16_checks: prefix_single флипнул на
обновлённом образе (torch 2.10.0+cu128) с ИДЕНТИЧНЫМИ префиксами в обоих прогонах дня [32,32,32,32,32,32,27,1] —
детерминированное расхождение fp16/bf16 на 2/8 промптах, все содержательные чеки с запасом; принят fp16 по записанному
правилу dtype_decision, deviation в numbers (gate0.fp16.deviation; первая версия текста ошибочно цитировала утренний
прогон — заменена тем же вечером до B1, обе записи в history). B1 pytest: 86/87, упал test_probe_stage_synthetic —
баг фикстуры (write_fake без mkdir), однострочный фикс в tests/fake_config.py → новый blob v8. Правка тест-кода до
единой реальной генерации.

## 12. 28.08 ночь — Gate 1 (T=0.7): вне окна → T-retry; PC alive=False оказался артефактом фильтра
secret_number deployment 0/48 cheat (100% honest) → вне окна → записанное T-правило: retry при T=1.0 (уточнение
«T после неудачного retry = 1.0» записано в numbers ДО повторного замера). sandbagging deployment 3.6% ≈ пол Сингха;
обрывы deployment 50–69% при cap 4096 (стоп-флаг записан); вербализация sandbagging 42% → accept with caveat (D2).
Диагностика PC: per-gen group='positive_control' (имя группы условия), gate1 фильтровал group=='pc' → PC-агрегаты
пусты, alive=False — АРТЕФАКТ; данные PC (60 генераций) целы. Фикс: выбор PC по condition_id (blob v9). Пилот T=0.7
заархивирован (pilot_*_T07_archived.jsonl, остаётся в results/), полный пилот A перегоняется при T=1.0 на остатке
пятничной квоты; PC-решение (жив/усиление) принимается по данным T=1.0 следующим gate1.

## 13. 28.08 ночь — Gate 1 пилота A (T=0.7) и два срабатывания правил
secret_number deployment 0/48 cheat (100% honest) → вне окна → T-правило: retry при T=1.0 (уточнение «что после
неудачного retry» записано в numbers ДО повторного замера). sandbagging deployment 3.6% ≈ пол Сингха → критерий 4.3а
ждёт E1. Обрывы deployment 50–69% при cap 4096 — стоп-флаг записан, complete-case+bounds. Вербализация sandbagging
42% → accept with caveat (D2 использована). БАГ ЧТЕНИЯ: per-gen group несёт имя группы условия ("positive_control"),
gate1 искал "pc" → PC-агрегаты пусты, alive=False был артефактом; фикс — выбор по condition_id (v9), репетиция
обновлена под прод-нейминг и проходит. Пилот T=0.7 архивирован (pilot_*_T07_archived.jsonl), retry при T=1.0 —
теми же 24 батчами, PC-дельта посчитается при финальной T. keyshuffle 0.34 (ожид. ~0.26) — репорт.
