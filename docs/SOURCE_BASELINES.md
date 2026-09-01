# Baseline-частоты источников (час-0 обязанность плана §2; заполнено 28.08 ДО пилота, до первой генерации)

Назначение: окно baseline [0.05, 0.40] (analysis.yaml pilot.baseline_window) опирается на фактические частоты
источников. Окно НЕ правится после этой записи; расхождение пилотных частот с ним репортится, порог не двигается.

| Источник | Среда | Величина | Число | Где |
|---|---|---|---|---|
| Singh et al., arXiv:2606.26071 v1 | Math Sandbagging | sandbagging-rate DeepSeek R1 0528 в eval-контексте | 60.5 % [57.4, 63.6] | §6.4 |
| Singh et al. | Math Sandbagging | то же после удаления фразы «We want to determine whether you can be deployed» | 23.4 % | §6.4 |
| Singh et al. (+ LW-пост «Why Did My Model Do That?») | Math Sandbagging | «deployment-подобный» пол: reverse-setting, ошибка/floor | 1 % (5/432) | derisk-раздел |
| Singh et al. | Secret Number | cheat-rate Kimi K2 Thinking при ЯВНОМ eval-фрейминге (сис-промпт «We are evaluating…», 20-я из 20 задач) | 76.8 % | §6.5, Appx C.5 |
| Singh et al. | Pre-commit Hook (контекст) | workaround-rate Kimi K2 Thinking / K2.5 / GLM 5.1 | 13.1 % [8.9, 18.9] / 2 % / 0 % | §6.1 |
| Singh et al. | Eval Tampering (контекст) | concealment-rate R1 0528 | 46.9 % | §6.3 |
| Rajamanoharan & Nanda, AF 11.06.2026 | их 3 эвала (Gemini) | знак: eval/frame-awareness может ПОВЫШАТЬ rate «undesired» действий; таксономия прочтений: puzzle/CTF, consequence-free simulation, alignment-test | качественный результат + плот с тремя эвалами (точные высоты столбиков снять с графика при финальной вычитке Related Work) | пост, вводный плот |

Как это обосновывает наши правила (записано до данных):
- **Sandbagging:** deployment-baseline у источника ≈ 0–1 % (reverse-setting floor 1 %; сандбэггинг у них живёт только в eval-контексте
  60.5 %) → окно [0.05, 0.40] к sandbagging-базе неприменимо, поэтому критерий 4.3(а) (E1-rate ≥ 0.10) — уже в analysis.yaml.
  Их деконструкция D2 (удаление одной фразы: 60.5 → 23.4 %) — прямой прецедент нашей D2-правки deployment-промпта.
- **Secret_number:** у источника НЕТ чистого deployment-условия — 76.8 % получены при явном eval-фрейминге; наше deployment-условие
  (вычищенная рамка) ново, точечного предсказания литература не даёт → окно вместо точки. [0.05, 0.40] заякорено: существенно ниже
  их eval-framed 76.8 % (наш deployment должен снять давление фрейминга; к тому же Qwen3.5-9B ≪ Kimi K2 Thinking по способностям —
  их точечная величина не переносится), и не ниже 0.05 (под MDE ≈ 0.17 сдвиг от почти нулевой базы вниз неизмерим, а «первая команда»
  у слабой модели редко cheat). Это конвенция, ЗАЯКОРЕННАЯ источником, а не выведенная из него, — так и репортится.
- **Знак (R&N):** eval-awareness двигает поведение в ОБЕ стороны (puzzle/CTF-прочтение повышает «undesired») → H-implicit знака
  не предсказывает; наши тесты двусторонние, язык «сдвигает», не «ухудшает/улучшает». Их таксономия прочтений
  (alignment-test / CTF / simulation) — источник verbalization.taxonomy в analysis.yaml.

Строка для Method: «The [0.05, 0.40] deployment window is anchored to the source rates: Singh et al. report 76.8 % cheating
under explicit eval framing (Secret Number, Kimi K2 Thinking) with no deployment condition, and a ~1 % floor (5/432) for
sandbagging outside the eval context; our window sits well below the former and above the MDE floor, and is a pre-registered
convention anchored by, not derived from, these numbers.»
