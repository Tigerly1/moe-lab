# Rozdział 5. Implementacja: środowisko badawcze i kernele

> Szkic. Kod źródłowy znajduje się w repozytorium `moe-lab`. Kolejne wersje kerneli są numerowane na wzór katalogu `dev/cuda` z llm.c, dzięki czemu każdy krok ablacji pozostaje w repozytorium.

## 5.1. Kontrakt wariantu

Wariant to pojedynczy plik w `moelab/variants/<backend>/` z czterema elementami:
- `prepare(w_gate, w_up, w_down)`: kwantyzacja i zmiana układu danych; poza pomiarem czasu;
- `run(state, x, idx, wts)`: obliczenie warstwy; mierzone;
- `dequant(state)`: wagi referencyjne dla formatów stratnych;
- `META`: metadane wariantu.

Routing jest wejściem wariantu (`idx`, `wts`). Każdy kernel liczy więc tę samą funkcję, a router mierzymy osobno.

Ten sam kontrakt obowiązuje w trzech backendach:
- MLX/Metal: `mlx_backend.py`;
- PyTorch/Triton/CUDA: `torch_backend.py`;
- JAX/XLA/Pallas: `jax_moe.py`.

## 5.2. Środowisko pomiarowe

- **Izolacja.** Każde zadanie wykonuje świeży proces (`worker.py`), który przekazuje wyniki jako NDJSON. Błąd kompilacji lub zawieszenie kernela nie przerywa serwera.
- **Bramka poprawności.** Wynik porównujemy z referencją fp32/fp64, a przypadki brzegowe opisuje §6.1. Bez przejścia bramki nie wykonujemy pomiaru czasu.
- **Pomiar czasu.** Stosujemy łańcuch wywołań ze strumieniem rezydualnym, czas jako nachylenie (`slope.py`) oraz porównanie A/B z przeplotem i przedziałem ufności z bootstrapu (`compare.py`). Test termiczny (`sustained.py`) to 10 minut ciągłego obciążenia.
- **Pełny model.** `e2e.py` podmienia blok MoE w mlx-lm i ma trzy tryby:
  - pomiar tok/s;
  - perplexity metodą teacher forcing na zamrożonym korpusie;
  - benchmark kroku weryfikacji bez zależności od akceptacji draftu.
- **Laboratorium interaktywne.** Aplikacja FastAPI z edytorem Monaco i wykresami na żywo. Edycja kernela w przeglądarce, Cmd+Enter i wyniki dla każdego T po kilku sekundach. Poprzedni przebieg jest rysowany linią przerywaną.

## 5.3. Ewolucja kernela dekodowania (T = 1)

| Wersja | Pomysł | Wynik (Qwen3, % sufitu) |
|---|---|---|
| v1 | jedna grupa SIMD na wiersz, gate+up+SiLU w jednym kernelu, down iteruje szeregowo po k | 44% |
| v2 | 4 wiersze na grupę SIMD, x przeskalowane, aby uniknąć przesunięć bitowych | 36% (hipoteza ALU obalona) |
| v3 | down równolegle po slotach, `h` w fp16, strojenie parametrów R i SG | 72% |
| v3.1 | 3 dispatche: kernel łączący, `idx` jako int32 | 74% |
| v4 (autoresearch, runda 2) | LPR linii na wiersz (`simd_shuffle_xor`), redukcja k slotów w pamięci grupy wątków | 98,5% (pomiar metodą nachylenia) |

Kluczowe obserwacje z pomiarów:
1. Wąskim gardłem była projekcja down, co ustaliliśmy, profilując każdy kernel osobno.
2. Przy K = 768 jedna grupa SIMD wykorzystywała tylko 24 z 32 linii.
3. Liczba dispatchy ma mały wpływ: GPU nakłada uruchomienie kolejnego kernela na ogon poprzedniego.

## 5.4. Kernel z ponownym użyciem ekspertów (T > 1)

1. **Mapowanie.** Jeden kernel-prepass grupuje pary (token, slot) według eksperta w kafle po maksymalnie 8.
2. **Kafle z co najmniej 3 slotami.** Obliczenia idą przez `simdgroup_matrix` 8×8: wejścia w half, akumulacja w fp32. Dekwantyzacja int4 do half jest dokładna, przez trik `as_type<half2>(q | 0x6400) − 1024`. Superbloki o szerokości 128 wzdłuż K dają jeden odczyt uint4 na linię.
3. **Kafle z 2 slotami.** Ścieżka skalarna: każde słowo wag jest dekodowane raz i wykonywane jest jedno FMA na slot.
4. **Kafle z 1 slotem.** Ścieżka GEMV taka jak w v4.

Kernel powstał w pętli autoresearch, w rundach 1 i 3.

**Ograniczenie i zachowanie przy dużych T.** Kernel wymaga d, f podzielnych przez 128. Dla gpt-oss (d = 2880) dyspozytor wybiera wariant skalarny. Przy T ≥ 8 kernel jest ograniczony liczbą instrukcji MMA, a nie przepustowością pamięci: ponowne użycie wag podnosi intensywność arytmetyczną.

## 5.5. Dyspozytor

`metal_q4_moe` wybiera kernel według reżimu:
- T = 1: v4;
- T > 1, aktywacje bf16/fp16 i obciążenie T·k/E nie większe od progu: tiles_v2 (MMA). Wystarczy podzielność d i f przez 64, którą spełnia każda kwantyzacja z grupą 64 (runda 9; wcześniej wymagaliśmy 128, przez co Gemma-4 trafiała na wolniejszy wariant);
- aktywacje fp32: tiles_b;
- obciążenie ponad próg (8 dla małych ekspertów, 24 dla d·f ≥ 1,5 mln): ścieżka `gather_qmm`, bit-identyczna z mlx-lm.

Aktywacja GLU jest parametrem szablonu (`ACT`): SiLU dla SwiGLU, GELU-tanh dla GeGLU Gemmy-4.

Wszystkie trzy warianty współdzielą układ stanu `mx.quantize`, więc kwantyzację wykonujemy raz.

## 5.7. Uniwersalna łatka na `SwitchGLU`

Łatki dla poszczególnych rodzin podmieniają cały blok MoE: router, ważenie i ekspertów. Każda nowa rodzina wymaga więc nowego kodu. Łatka uniwersalna (`moelab/universal.py`) podmienia tylko `SwitchGLU`, czyli obliczenia ekspertów dla par (token, ekspert):
- każdy slot trafia do kernela jako osobny wiersz z k = 1 i wagą 1, a kernel zwraca wyjścia per slot w kształcie oczekiwanym przez model;
- przy T = 1 GEMV v4 działa w trybie bez redukcji po k (`NORED`);
- router, ważenie i eksperci współdzieleni pozostają kodem modelu.

**Koszt** względem łatki dedykowanej to brak połączonego routera i połączonej sumy ważonej: 1,075× zamiast 1,086× na Qwen3.

**Weryfikacja poprawności:** test z bit-identyczną ścieżką zapasową w miejscu kernela. Logity muszą być równe stock dokładnie, a test mutacyjny potwierdza, że wykrywa pominięcie eksperta współdzielonego.

Ten sam projekt trafił do zgłoszenia w mlx-lm (#2044) jako `mlx_lm/models/switch_kernels.py`. Operacja jest tam `mx.custom_function` z gradientem liczonym przez `gather_qmm`.

## 5.6. Autoresearch: metodyka poszukiwań wspieranych przez agenta

Protokół na wzór karpathy/autoresearch:
- człowiek pisze `program.md` z celem, metryką, regułami i znaną wiedzą;
- agent edytuje dokładnie jeden plik;
- ewaluator jest tylko do odczytu: bramka poprawności plus A/B z przeplotem względem rzeczywistej ścieżki mlx-lm;
- każdy eksperyment to commit w git, zachowywany albo cofany (`reset`);
- wyniki trafiają do pliku tsv.

Przeprowadzone rundy:

| Runda | Cel | Wynik |
|---|---|---|
| 1 | ponowne użycie ekspertów dla T > 1 | wynik 1,12 → 1,43 |
| 2 | dekodowanie T = 1 na 5 kształtach | wynik +1,5%; sufit przepustowości osiągnięty |
| 3 | kafle dla T > 1 na 3 modelach | +4% |
| 4 | format MXFP4 | w toku |

Wnioski metodyczne:
1. Reguła akceptacji oparta na przedziale ufności bazy, a nie kandydata, odrzucała dobre kernele.
2. Każdy wynik agenta weryfikowaliśmy niezależnie, z większą liczbą powtórzeń i na kształtach spoza strojenia.
3. Agent zgłaszał hipotezy obalone pomiarem, np. że łączenie dispatchy da zysk. To wartościowy materiał do pracy.
