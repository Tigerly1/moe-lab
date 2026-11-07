# Rozdział 6. Ewaluacja eksperymentalna

> Szkic, stan na 2026-11-07. Źródła liczb: `docs/RESULTS.md` oraz `runs/*.json`. Wszystkie pomiary pochodzą z Apple M4 Pro (20 rdzeni GPU, 48 GB), na zasilaniu sieciowym. Wersje: MLX 0.32.4 i mlx-lm 0.31.3. Typ danych: bf16, wagi int4 affine z grupą 64.

## 6.1. Metodyka pomiaru

1. **Poprawność przed szybkością.** Każdy wariant porównujemy z referencją fp32 policzoną na tych samych zdekwantyzowanych wagach. Kryterium to względny błąd Frobeniusa poniżej 5·10⁻³ (fp16) i 3·10⁻² (bf16). Zestaw testów obejmuje:
   - przypadki brzegowe: T = 1, T niepodzielne przez kafel, routing one-hot i Zipfa, rzeczywiste ślady routingu;
   - test całego modelu: logity mlx-lm z podmienionym kernelem dla T = 1, 2, 3 i 8.

   Razem 167 testów.
2. **Uczciwy punkt odniesienia.** Bazą jest rzeczywista ścieżka mlx-lm, czyli `SwitchGLU` ze skompilowanym `swiglu` i kombinacją z Qwen3.

   Pierwsza wersja bazy odtwarzała tę ścieżkę ręcznie. Była o trzy dispatche wolniejsza i zawyżała przyspieszenie z 1,17× do 1,25×. Wykrył to niezależny audyt (§7.1).
3. **Czas na wywołanie.** Mierzymy łańcuch zależnych wywołań ze strumieniem rezydualnym (x ← x + y), z jednym `eval` na łańcuch.

   Czas jednego wywołania wyznaczamy jako nachylenie prostej czasu całkowitego względem długości łańcucha (10, 20, 40 i 80 wywołań). Wyraz wolny prostej to stały koszt pojedynczego `eval`, wynoszący 130–340 µs. Nachylenie od niego nie zależy.
4. **Statystyka.** Warianty A i B przeplatamy w ramach jednego procesu. Wykonujemy 3–5 niezależnych uruchomień procesu po 6 rund. Przyspieszenie liczymy jako stosunek median, z 95-procentowym przedziałem ufności metodą bootstrapu (próbkowanie w parach).
5. **Sufit sprzętowy.** Przepustowość samego odczytu zmierzyliśmy osobnym kernelem: 240–250 GB/s, wobec 273 GB/s w specyfikacji.

## 6.2. RQ2: dekodowanie (T = 1) na granicy przepustowości

| Kształt (d×f, E, k) | Dolne ograniczenie przy 250 GB/s | mlx-lm | Nasz kernel (v4) | Przyspieszenie |
|---|---|---|---|---|
| Qwen3-30B-A3B (2048×768, 128, 8) | 85,1 µs | 107,4 µs (79%) | 86,4 µs (98,5%) | 1,244× |
| gpt-oss-20b (2880×2880, 32, 4) | 224,2 µs | 248,9 µs (90%) | 225,8 µs (99%) | 1,10–1,12× |
| Mixtral-8x7B (4096×14336, 8, 2) | 793,2 µs | 801,8 µs (99%) | 771,1 µs (103%) | 1,04× |

W nawiasach podano odsetek dolnego ograniczenia. Przy T = 1 nasz kernel czyta pamięć z prędkością bliską sprzętowemu limitowi. Dalsze przyspieszenie przy T = 1 wymaga zmniejszenia liczby przesyłanych bajtów, a nie lepszego kernela.

**Mechanizm.** Ablacja zmienia wyłącznie wymiar K. Około połowa przewagi wynika z tego, że szybka ścieżka `qmv` w MLX wymaga K % 512 = 0. Qwen3 (K = 768 w projekcji down) i gpt-oss (K = 2880) tego warunku nie spełniają, więc MLX używa wolniejszej ścieżki.

Pozostała część przewagi pochodzi z trzech źródeł:
- fuzja gate+up+SiLU;
- pełne wykorzystanie 32 linii przy K = 768;
- redukcja k slotów w pamięci grupy wątków.

## 6.3. RQ1 / RQ3: T > 1 i ponowne użycie ekspertów

Przyspieszenie liczymy względem mlx-lm na kształtach Qwen3:

| T | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| ślady rzeczywiste, kafle + MMA | 1,16 | 1,22 | 1,36 | 1,64 | 1,87 | 2,36 |
| routing jednostajny, kafle + MMA | 1,16 | 1,14 | 1,12 | 1,15 | 1,22 | 1,44 |

Zysk rośnie z T zgodnie z modelem z rozdziału 4. Przy rzeczywistym routingu T kolejnych tokenów dotyka U(T) ≪ T·k ekspertów. Kafel czyta wagi eksperta raz i mnoży je przez wszystkie tokeny kafla na macierzach 8×8 grupy SIMD.

## 6.3a. RQ1: punkt przejścia względem mlx-lm

Przyspieszenie liczymy względem mlx-lm na kształtach Qwen3 (int4):

| T | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|
| ślady rzeczywiste | 1,98 | 2,45 | 1,61 | 1,34 | 1,12 | 1,00 |
| routing jednostajny | 1,33 | 1,55 | 1,93 | 1,36 | 1,09 | 0,99 |

Nasze kernele wygrywają do T = 256. Przy T = 512 osiągamy parytet z posortowanym, grupowanym GEMM-em w MLX. Model analityczny z rozdziału 4 przewiduje, że warstwa przestaje być ograniczona pamięcią przy T* ≈ 157 dla routingu jednostajnego. Zmierzony przebieg jest z tym zgodny: zysk z ponownego użycia wag maleje, gdy warstwa staje się ograniczona obliczeniami.

W pełnym modelu fragment 32 tokenów, czyli krótki prefill lub mały batch, przetwarzamy 1,86× szybciej.

## 6.4. Pełny model (Qwen3-30B-A3B-4bit, mlx-lm)

| Tryb | mlx-lm (tok/s) | Nasz kernel (tok/s) | Zysk |
|---|---|---|---|
| dekodowanie, 512 tokenów | 91,5 | 99,9 | +9,2% |
| krok weryfikacji, 4 tokeny | 124,4 | 148,6 | +19,4% |
| krok weryfikacji, 7 tokenów | 128,4 | 167,6 | +30,5% |
| krok weryfikacji, 16 tokenów | 193,8 | 271,8 | +40,3% |

**Jakość.** Perplexity metodą teacher forcing na zamrożonym korpusie 1000 tokenów:
- mlx-lm: 21,17;
- nasz kernel: 21,15;
- eksperci w fp32: 21,02.

Każda para zgadza się co do najbardziej prawdopodobnego tokenu (top-1) w 95,8–96,2% pozycji. To poziom szumu numerycznego bf16, a nie degradacja jakości.

## 6.4a. Format MXFP4 (format ekspertów gpt-oss)

Porównujemy nasz kernel ze ścieżką mlx-lm skwantyzowaną w trybie `mxfp4`. Format zapisuje wagi jako FP4 E2M1 ze skalą E8M0 na każde 32 wagi, co daje 4,25 bitu na wagę.

| Kształt | T = 1 | T = 4 | T = 8 | T = 16 | T = 32 |
|---|---|---|---|---|---|
| gpt-oss-20b | 1,16 | 1,29 | 1,50* | 1,60* | – |
| Qwen3-30B-A3B (ślady) | 1,27 | 1,53 | 1,73* | 2,00* | 2,34* |
| gpt-oss-120b (spoza strojenia) | 1,17 | 1,14 | 1,16* | 1,24* | – |
| Mixtral-8x7B (spoza strojenia) | 1,05 | 1,39 | 1,77* | – | – |
| DeepSeek-V2-Lite (spoza strojenia) | 1,18 | 1,25 | 1,34* | 1,38* | – |

\* Pomiar po poprawkach przepełnienia skali (§7.6); w rundzie 5 skala E8M0 jest stosowana w fp32 do częściowej sumy każdej 32-elementowej grupy.

**Koszt dekodowania.** Wartości FP4 dekodujemy dokładnie, jedną operacją bitową dla pary wartości do typu `half2`. Kosztuje to tyle samo co dekodowanie int4 affine. Format nie ma członu bias i nie wymaga przeskalowania x, więc kernel MXFP4 jest szybszy od kernela int4 mniej więcej w stosunku liczby bajtów (4,25 wobec 4,5 bitu).

**Wniosek dla RQ4.** Te same decyzje projektowe przenoszą się między formatami kwantyzacji bez zmian:
- GEMV z pełnym wykorzystaniem linii przy T = 1;
- współdzielenie odczytów wag między slotami tego samego eksperta przy T > 1.

## 6.4b. Drugi pełny model: gpt-oss-20b (MXFP4)

| Tryb | mlx-lm (tok/s) | Nasz kernel (tok/s) | Zysk |
|---|---|---|---|
| dekodowanie, 512 tokenów | 75,5 | 81,5 | +7,9% |
| krok weryfikacji, 4 tokeny | 125,9 | 160,4 | +27,4% |
| krok weryfikacji, 16 tokenów | 174,8 | 296,4 | +69,5% |

**Jakość (teacher forcing).** Nasz kernel jest bliższy referencji fp32 niż ścieżka mlx-lm:
- zgodność top-1 z fp32: 92,3% wobec 90,5%;
- perplexity: 179,6 wobec 181,2 (fp32: 172,4).

**Różnice względem Qwen3:**
- obciążenia (bias) dla każdego eksperta;
- przycięta funkcja SwiGLU (alpha = 1,702, limit 7, +1 na gałęzi liniowej);
- skale E8M0 z rzeczywistego checkpointu. To one ujawniły błąd opisany w §7.6.

## 6.5. Do uzupełnienia

- [ ] Ten sam zestaw pomiarów na GPU NVIDIA: Triton fused_moe i Marlin-MoE przez Modal (RQ4).
- [ ] Drugi układ Apple, np. M1/M2/M3 (§7.3).
- [ ] Wykresy w formacie pracy (matplotlib, PDF), wygenerowane z `runs/*.json`.
