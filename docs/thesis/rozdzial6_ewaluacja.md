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
| ślady rzeczywiste | 1,95 | 2,43 | 1,53 | 1,29 | 1,10 | 0,98 |
| routing jednostajny | 1,27 | 1,53 | 1,87 | 1,36 | 1,09 | 0,99 |

Tabela pokazuje stan sprzed rundy 7: kernele wygrywały do T = 256 na kształcie Qwen3. Szerszy przegląd (§6.7) wykazał jednak, że o zwycięzcy nie decyduje T, lecz **obciążenie** `L = T·k/E`, czyli średnia liczba tokenów na eksperta. Qwen3 ma E = 128 i k = 8, więc nawet przy T = 256 obciążenie wynosi tylko 16; modele z mniejszą liczbą ekspertów osiągają ten próg znacznie wcześniej. Model analityczny z rozdziału 4 przewiduje przejście warstwy w reżim obliczeniowy przy T* ≈ 157 dla routingu jednostajnego. Zmierzony przebieg jest z tym zgodny: zysk z ponownego użycia wag maleje, gdy warstwa staje się ograniczona obliczeniami.

W pełnym modelu fragment 32 tokenów, czyli krótki prefill lub mały batch, przetwarzamy 1,83× szybciej.

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

## 6.4c. Trzeci pełny model: DeepSeek-V2-Lite (eksperci współdzieleni, MLA)

| Tryb | mlx-lm (tok/s) | Nasz kernel (tok/s) | Zysk |
|---|---|---|---|
| dekodowanie (identyczne wyjście, 225/225 tokenów) | 126,9 | 135,9 | +7,1% |
| krok weryfikacji, 4 tokeny | 184,8 | 210,0 | +13,7% |
| krok weryfikacji, 16 tokenów | 257,4 | 343,6 | +33,5% |
| krok weryfikacji, 32 tokeny | 324,8 | 531,0 | +63,5% |

**Jakość.** Zgodność top-1 z referencją fp32 wynosi 97,87% dla naszego kernela i 97,70% dla mlx-lm. Perplexity: 41,40 dla naszego kernela, 41,35 dla mlx-lm i 41,26 dla fp32.

**Wniosek dla RQ4.** Te same kernele dają zysk w trzech różnych architekturach MoE:
- Qwen3-MoE;
- gpt-oss: MXFP4, biasy, przycięta funkcja SwiGLU;
- DeepSeek-V2: eksperci współdzieleni, MLA.

## 6.5. Gdzie zysk jest praktyczny: równoległe zapytania

Dekodowanie pojedynczej rozmowy przyspiesza o 7–9% (z połączonym routerem z §6.6: 10–17%). Największy praktyczny zysk pojawia się jednak wtedy, gdy Mac obsługuje kilka zapytań naraz: serwer `mlx_lm.server` z wieloma użytkownikami, agent z równoległymi wywołaniami, ewaluacja na zbiorze. Każdy krok dekodowania przesyła wtedy przez warstwę MoE B tokenów z B różnych sekwencji. Stock mlx-lm aż do 64 slotów czyta wagi ekspertów osobno dla każdej pary (token, slot), a nasze kernele grupują sloty według eksperta.

**Benchmark `BatchGenerator`** (generator wsadowy używany przez serwer mlx-lm; łączna przepustowość, kernele + połączony router, mlx-lm 0.32.2, `runs/r8/latest/`):

| B | Qwen3-30B-A3B | gpt-oss-20b | DeepSeek-V2-Lite |
|---|---|---|---|
| 1 | 1,16 | 1,09 | 1,12 |
| 8 | 1,23 | 1,43 | 1,27 |
| 32 | **1,59** (222 → 352 tok/s) | **1,60** (224 → 359) | **1,54** (312 → 481) |

Dwa niezależne uruchomienia na mlx-lm 0.31.3 dają te same stosunki z dokładnością ±0,05.

**Prawdziwy serwer HTTP.** Uruchamiamy `moelab.serve`, czyli `mlx_lm.server` z naszymi kernelami, i porównujemy go z serwerem stock. Klienci wysyłają równolegle zapytania OpenAI `/v1/completions`, 256 tokenów każde (`runs/r8/server_*.json`):

| klienci | Qwen3-30B-A3B | gpt-oss-20b | DeepSeek-V2-Lite |
|---|---|---|---|
| 8 | 1,16×, mediana odpowiedzi 17,0 → 14,6 s | 1,35×, 19,3 → 14,2 s | 1,18×, 14,0 → 11,9 s |
| 32 | **1,38×**, 52,2 → 37,8 s | **1,38×**, 51,8 → 37,3 s | **1,32×**, 40,8 → 30,9 s |

Zysk na serwerze jest niższy niż w benchmarku, bo serwer dodaje prefill, tokenizację i HTTP, a tych etapów kernele nie przyspieszają.

**Poprawność w trybie wsadowym.** Dla jednego kroku dekodowania przy B = 32 największy względny błąd logitów względem prawdy z ekspertami fp32 wynosi 0,19 dla stock i 0,07 dla naszych kerneli; zgodność argmax jest taka sama (93,8%). Wczesne rozchodzenie się wyjść zachłannych przy B ≥ 8 wynika więc z wrażliwości modelu na szum bf16, a nie z błędu kernela.

![Przepustowość przy równoległych zapytaniach](fig/batch_scaling.pdf)

![Serwer mlx-lm i llama.cpp](fig/server_llamacpp.pdf)

**Porównanie z llama.cpp** (Qwen3-30B-A3B, GGUF Q4_0 o tej samej liczbie 4,5 bita na wagę, ustawienia domyślne):

| | llama.cpp | stock mlx-lm | nasze kernele |
|---|---|---|---|
| 1 strumień (czyste dekodowanie) | 82 tok/s | 90 | 105 |
| serwer, 8 klientów | **144** | 121 | 142 |
| serwer, 32 klientów | 150 | 156 | **222** |

Bez naszych kerneli MLX przy 8 klientach ustępuje llama.cpp o 16%. Z nimi wyrównuje przy 8 klientach i wyprzedza llama.cpp o 48% przy 32. Zastrzeżenia: oba systemy działają z ustawieniami domyślnymi, porównanie obejmuje jeden model, a formaty kwantyzacji różnią się szczegółami.

## 6.6. Połączony router

W Qwen3 router (bramka, softmax, argpartition, take, normalizacja) kosztuje 1,49 ms na token. Sama bramka to tylko 0,26 ms; resztę stanowi 4–5 małych wywołań kerneli na warstwę. Jeden kernel Metal (`moelab/router.py`) wybiera top-k bezpośrednio na logitach i liczy wagi. Tak samo projektuje bramkę SGLang. Kernel podnosi zysk w dekodowaniu jednego strumienia z 1,09× do 1,17× (Qwen3), z 1,08× do 1,10× (gpt-oss) i z 1,07× do 1,12× (DeepSeek). Przy B ≥ 4 koszt routera rozkłada się na batch i jego wkład jest pomijalny.

Zmienia się rozstrzyganie remisów. W 9% par (token, warstwa) Qwen3 dwa eksperty mają dokładnie równy logit bf16. Stock i nasz kernel wybierają wtedy różne, równie poprawne zbiory top-k. Perplexity różni się o mniej niż 0,9%. Pierwotna interpretacja, że stock myli się w tych przypadkach, okazała się fałszywa (§7.11).

## 6.7. Szeroki i wąski przegląd poprawności i wydajności

- **Wąskie przypadki brzegowe:**
  - wszystkie granice dyspozytora;
  - wymiary niepodzielne przez 128, k = 1, k = E, E = 1;
  - zerowe i nienormalizowane wagi;
  - wszystkie tokeny do tych samych ekspertów;
  - niezależność od kolejności slotów i od składu batcha;
  - aktywacje 100×;
  - NaN w logitach routera.
  
  Każdy przypadek uruchamiamy też z wyłączoną ścieżką awaryjną, żeby na pewno wykonywał się nasz kernel.
- **Szeroki przegląd:** 480 losowych kształtów (d 512–5120, f 256–4096, E 8–256, k 1–8, T 1–128, trzy rodzaje routingu). Wszystkie są poprawne.
- **Wydajność:** kernele wygrywają do obciążenia 24 na ekspertach rozmiaru Qwen3-30B i większych, do 8 na mniejszych oraz do 4 dla int4 przy d = 2880 (ścieżka skalarna). Powyżej tych progów dyspozytor przełącza się na ścieżkę mlx-lm, bit w bit identyczną z oryginałem, więc wynik jest wtedy równy punktowi odniesienia. Przykładowo kształt gpt-oss w int4 przy T = 128 poprawił się z 0,51× do 0,98–1,00×.
![Zysk w funkcji obciążenia; progi dyspozytora](fig/load_policy.pdf)

- **Pozostałe straty:**
  - wszystkie k ekspertów otrzymuje każdy token (tylko syntetycznie, 0,76×; w rzeczywistych śladach nie występuje);
  - eksperci mniejsi niż w jakimkolwiek badanym modelu (0,93–0,95×).
- Każde oznaczenie z szybkiego przesiewu było ponownie mierzone pełną procedurą. Kilka rzekomych strat (0,51×, 0,55×, 0,78×) okazało się wygranymi 1,05–1,31×.

## 6.8. Długi kontekst

Rozbicie czasu kroku dekodowania (Qwen3, dwa czyste uruchomienia, `runs/r8/longctx_*_clean*.json`) pokazuje, że zysk na samej warstwie MoE nie zależy od kontekstu: 1,14–1,17× przy 1k i 1,13–1,14× przy 16k (z routerem: 1,27× w obu przypadkach). Zysk całego modelu maleje z 1,13× do 1,07× wyłącznie z powodu prawa Amdahla, bo rośnie udział atencji. Wcześniejszy pojedynczy pomiar sugerujący spadek zysku MoE do 1,04× nie dał się odtworzyć (§7.10).

## 6.9. Aktualność punktu odniesienia

Główne wyniki zmierzyliśmy ponownie na najnowszych wersjach (mlx-lm 0.32.2, MLX 0.32.4). MLX 0.32.4 zawiera już ulepszenie `gather_qmm` z PR #4572. Stosunki zgadzają się z pomiarami na mlx-lm 0.31.3.

## 6.10. Do uzupełnienia

- [ ] Ten sam zestaw pomiarów na GPU NVIDIA: Triton fused_moe i Marlin-MoE przez Modal (RQ4).
- [ ] Drugi układ Apple, np. M2 (`scripts/portability_m2.sh`, §7.3).
- [x] Wykresy dla §6.5 i §6.7 (`python -m moelab.figures`).
